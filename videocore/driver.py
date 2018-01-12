"""VideoCore driver."""

import os
import struct
import mmap

import numpy as np

import rpi_vcsm.VCSM

from videocore.mailbox import MailBox
from videocore.assembler import assemble

DEFAULT_MAX_THREADS = 1024
DEFAULT_DATA_AREA_SIZE = 32 * 1024 * 1024
DEFAULT_CODE_AREA_SIZE = 1024 * 1024

class DriverError(Exception):
    'Exception related to QPU driver'

class Array(np.ndarray):
    def __new__(cls, *args, **kwargs):
        address = kwargs.pop('address')
        obj = super(Array, cls).__new__(cls, *args, **kwargs)
        obj.address = address
        return obj

    def addresses(self):
        return np.arange(
            self.address,
            self.address + self.nbytes,
            self.itemsize,
            np.uint32
            ).reshape(self.shape)

class Memory(object):
    def __init__(self, vcsm, size, cache_mode = rpi_vcsm.CACHE_NONE):
        self.size = size
        self.vcsm = vcsm
        self.handle = None   # vcsm handle which corresponds to a memory area
        self.busaddr = None  # bus address used for QPU
        self.usraddr = None  # user virtual address used in user CPU program
        self.buffer = None   # mmap object of the memory area
        try:
            (self.handle, self.busaddr, self.usraddr, self.buffer) = \
                    vcsm.malloc_cache(size, cache_mode, 'py-videocore')
            if self.handle == 0:
                raise DriverError('Failed to allocate QPU device memory')
        except:
            self.close()
            raise

    def close(self):
        if self.handle:
            self.vcsm.free(self.handle, self.buffer)
        self.handle = None
        self.busaddr = None
        self.usraddr = None
        self.buffer = None

class Program(object):
    def __init__(self, code_addr, code):
        self.address = code_addr
        self.code    = code

class Driver(object):
    def __init__(self,
            data_area_size = DEFAULT_DATA_AREA_SIZE,
            code_area_size = DEFAULT_CODE_AREA_SIZE,
            max_threads    = DEFAULT_MAX_THREADS,
            cache_mode     = rpi_vcsm.CACHE_NONE
            ):
        self.mailbox = MailBox()
        self.mailbox.enable_qpu(1)
        self.vcsm = rpi_vcsm.VCSM.VCSM()
        self.memory  = None
        try:
            self.data_area_size = data_area_size
            self.code_area_size = code_area_size
            self.max_threads = max_threads

            self.code_area_base = 0
            self.data_area_base = self.code_area_base + self.code_area_size
            self.msg_area_base  = self.data_area_base + self.data_area_size

            self.code_pos = self.code_area_base
            self.data_pos = self.data_area_base

            total = data_area_size + code_area_size + max_threads * 64
            self.memory = Memory(self.vcsm, total, cache_mode=cache_mode)

            self.message = Array(
                    address = self.memory.busaddr + self.msg_area_base,
                    buffer  = self.memory.buffer,
                    offset = self.msg_area_base,
                    shape = (self.max_threads, 2),
                    dtype = np.uint32)
        except:
            self.close()
            raise

    def close(self):
        if self.memory:
            self.memory.close()
        self.memory = None
        self.mailbox.enable_qpu(0)
        self.mailbox.close()
        self.mailbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()
        return exc_type is None

    def copy(self, arr):
        new_arr = Array(
                shape   = arr.shape,
                dtype   = arr.dtype,
                address = self.memory.busaddr + self.data_pos,
                buffer  = self.memory.buffer,
                offset  = self.data_pos
                )
        if self.data_pos + new_arr.nbytes > self.msg_area_base:
            raise DriverError('Array too large')
        self.data_pos += new_arr.nbytes
        new_arr[:] = arr
        return new_arr

    def alloc(self, *args, **kwargs):
        arr = Array(
                *args,
                address = self.memory.busaddr + self.data_pos,
                buffer  = self.memory.buffer,
                offset  = self.data_pos,
                **kwargs)
        if self.data_pos + arr.nbytes > self.msg_area_base:
            raise DriverError('Array too large')
        self.data_pos += arr.nbytes
        return arr

    def array(self, *args, **kwargs):
        arr = np.array(*args, copy = False, **kwargs)
        new_arr = self.alloc(arr.shape, arr.dtype)
        new_arr[:] = arr
        return new_arr

    def program(self, program, *args, **kwargs):
        if hasattr(program, '__call__'):
            program = assemble(program, *args, **kwargs)
        code = memoryview(program).tobytes()
        if self.code_pos + len(code) > self.data_area_base:
            raise DriverError('Program too long')
        code_addr = self.memory.busaddr + self.code_pos
        self.memory.buffer[self.code_pos:self.code_pos+len(code)] = code
        self.code_pos += len(code)
        return Program(code_addr, code)

    def execute(self, n_threads, program, uniforms = None, timeout = 10000):
        if not (1 <= n_threads and n_threads <= self.max_threads):
            raise DriverError('n_threads exceeds max_threads')
        if uniforms is not None:
            if not isinstance(uniforms, Array):
                uniforms = self.array(uniforms, dtype = 'u4')
            self.message[:n_threads, 0] = uniforms.addresses().reshape(n_threads, -1)[:, 0]
        else:
            self.message[:n_threads, 0] = 0

        self.message[:n_threads, 1] = program.address

        r = self.mailbox.execute_qpu(n_threads, self.message.address, 0, timeout)
        if r > 0:
            raise DriverError('QPU execution timeout')
