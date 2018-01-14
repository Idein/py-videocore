"""VideoCore driver."""

import os
import struct
import mmap
from math import ceil

import numpy as np

import rpi_vcsm.VCSM

from videocore.mailbox import MailBox
from videocore.assembler import assemble

DEFAULT_MAX_THREADS = 12
DEFAULT_DATA_AREA_SIZE = 32 * 1024 * 1024
DEFAULT_CODE_AREA_SIZE = 1024 * 1024

class DriverError(Exception):
    'Exception related to QPU driver'

class Array(np.ndarray):
    def __new__(cls, *args, **kwargs):
        vcsm = kwargs.pop('vcsm')
        address = kwargs.pop('address')  # bus address
        usraddr = kwargs.pop('usraddr')
        buffer = kwargs.pop('buffer')
        offset = kwargs.pop('offset')

        try:
            obj = super(Array, cls).__new__(cls, *args, buffer = buffer,
                                            offset = offset, **kwargs)
        except TypeError as e:
            raise DriverError('Array too large: {0}'.format(e))

        obj.vcsm = vcsm
        obj.address = address
        obj.usraddr = usraddr
        obj.buffer = buffer
        obj.offset = offset
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

class Mempool(object):
    def __init__(self, size, **kwargs):
        self.size = size
        self.vcsm = kwargs.pop('vcsm')
        cache_mode = kwargs.pop('cache_mode')

        self.start_pos = {}
        self.cur_pos = {}
        total = 0
        for (n, s) in size.items():
            self.start_pos[n] = self.cur_pos[n] = total
            total += s
        self.total = total

        try:
            self.memory = Memory(self.vcsm, total, cache_mode=cache_mode)
        except:
            self.close()
            raise

    def close(self):
        self.vcsm = None
        self.size = None
        if self.memory:
            self.memory.close()
        self.memory = None
        self.start_pos = None
        self.cur_pos = None

    def alloc(self, name, *args, **kwargs):
        pos = self.cur_pos[name]
        arr = Array(
                *args,
                vcsm = self.vcsm,
                address = self.memory.busaddr + pos,
                usraddr = self.memory.usraddr + pos,
                buffer  = self.memory.buffer,
                offset  = pos,
                **kwargs)
        if (pos - self.start_pos[name]) + arr.nbytes > self.size[name]:
            raise DriverError('Array too large')
        self.cur_pos[name] += arr.nbytes
        return arr

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

        if cache_mode in [rpi_vcsm.CACHE_HOST, rpi_vcsm.CACHE_BOTH]:
            self.is_cacheop_needed = True
        else:
            self.is_cacheop_needed = False

        self.max_threads = max_threads
        message_area_size = max_threads * 2 * 4

        try:
            # Non-cached memory area for code and message.
            self.ctlmem = Mempool({'message': message_area_size,
                                   'code': code_area_size},
                                  vcsm = self.vcsm,
                                  cache_mode = rpi_vcsm.CACHE_NONE)
            # Memory area for uniforms and data with cache mode cache_mode.
            self.datmem = Mempool({'data': data_area_size},
                                  vcsm = self.vcsm, cache_mode = cache_mode)

            self.message = self.ctlmem.alloc('message',
                                             shape = (self.max_threads, 2),
                                             dtype = np.uint32)

        except:
            self.close()
            raise

    def close(self):
        if self.ctlmem:
            self.ctlmem.close()
        self.ctlmem = None
        if self.datmem:
            self.datmem.close()
        self.datmem = None
        self.mailbox.enable_qpu(0)
        self.mailbox.close()
        self.mailbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()
        return exc_type is None

    def copy(self, arr):
        new_arr = self.alloc(shape = arr.shape, dtype = arr.dtype)
        new_arr[:] = arr
        return new_arr

    def alloc(self, *args, **kwargs):
        return self.datmem.alloc('data', *args, **kwargs)

    def array(self, *args, **kwargs):
        arr = np.array(*args, copy = False, **kwargs)
        new_arr = self.alloc(arr.shape, arr.dtype)
        new_arr[:] = arr
        return new_arr

    def program(self, program, *args, **kwargs):
        if hasattr(program, '__call__'):
            program = assemble(program, *args, **kwargs)
        code = memoryview(program).tobytes()
        arr = self.ctlmem.alloc('code', shape = int(ceil(len(code) / 8.0)),
                                dtype = np.uint64)
        arr.buffer[arr.offset:arr.offset+len(code)] = code
        return Program(arr.address, arr.usraddr, code, len(code))

    def execute(self, n_threads, program, uniforms = None, timeout = 10000):
        if not (1 <= n_threads and n_threads <= self.max_threads):
            raise DriverError('n_threads exceeds max_threads')

        message = self.message

        if uniforms is not None:
            if not isinstance(uniforms, Array):
                uniforms = self.array(uniforms, dtype = 'u4')
            message[:n_threads, 0] = uniforms.addresses().reshape(n_threads, -1)[:, 0]
        else:
            message[:n_threads, 0] = 0

        message[:n_threads, 1] = program.address

        if self.is_cacheop_needed:
            uniforms.clean()
        r = self.mailbox.execute_qpu(n_threads, message.address, 0, timeout)
        if self.is_cacheop_needed:
            uniforms.invalidate()
        if r > 0:
            raise DriverError('QPU execution timeout')
