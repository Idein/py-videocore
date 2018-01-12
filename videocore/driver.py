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
        vcsm = kwargs.pop('vcsm')
        address = kwargs.pop('address')  # bus address
        usraddr = kwargs.pop('usraddr')  # user virtual address
        obj = super(Array, cls).__new__(cls, *args, **kwargs)
        obj.vcsm = vcsm
        obj.address = address
        obj.usraddr = usraddr
        return obj

    def addresses(self):
        return np.arange(
            self.address,
            self.address + self.nbytes,
            self.itemsize,
            np.uint32
            ).reshape(self.shape)

    # Mark the content of CPU cache as invalid i.e. the content must be fetched
    # from memory when user read from the location.
    def invalidate(self):
        self.vcsm.invalidate(self.usraddr, self.nbytes)

    # Write the content of CPU cache to memory.
    def clean(self):
        self.vcsm.clean(self.usraddr, self.nbytes)

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
    def __init__(self, code_addr, usraddr, code, size):
        self.address = code_addr
        self.usraddr = usraddr
        self.code    = code
        self.size    = size

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

        if cache_mode in [rpi_vcsm.CACHE_HOST, rpi_vcsm.CACHE_BOTH]:
            self.is_cacheop_needed = True
        else:
            self.is_cacheop_needed = False

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
                    vcsm = self.vcsm,
                    address = self.memory.busaddr + self.msg_area_base,
                    usraddr = self.memory.usraddr + self.msg_area_base,
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
                vcsm    = self.vcsm,
                shape   = arr.shape,
                dtype   = arr.dtype,
                address = self.memory.busaddr + self.data_pos,
                usraddr = self.memory.usraddr + self.data_pos,
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
                vcsm    = self.vcsm,
                address = self.memory.busaddr + self.data_pos,
                usraddr = self.memory.usraddr + self.data_pos,
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
        usraddr = self.memory.usraddr + self.code_pos
        self.memory.buffer[self.code_pos:self.code_pos+len(code)] = code
        self.code_pos += len(code)
        return Program(code_addr, usraddr, code, len(code))

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

        if self.is_cacheop_needed:
            uniforms.clean()
            self.vcsm.clean(program.usraddr, program.size)
            self.message.clean()
        r = self.mailbox.execute_qpu(n_threads, self.message.address, 0, timeout)
        if self.is_cacheop_needed:
            uniforms.invalidate()
            self.vcsm.invalidate(program.usraddr, program.size)
            self.message.invalidate()
        if r > 0:
            raise DriverError('QPU execution timeout')
