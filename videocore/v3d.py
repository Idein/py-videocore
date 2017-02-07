"""V3D registers.

This code is based on https://github.com/Terminus-IMRC/libvc4v3d
"""

import os
import ctypes
import mmap

V3D_OFFSET_FROM_PERI = 0x00c00000
V3D_LENGTH = 0x00f20 - 0x00000 + 32 / 8


class V3DRegisters(object):
    def __init__(self):
        self.v3d_base = self._mmap_v3d_region()

    def close():
        self.v3d_base.close()

    def _mmap_v3d_region(self):
        fd = os.open('/dev/mem', os.O_RDWR|os.O_SYNC)
        base = mmap.mmap(fd, 
            V3D_LENGTH,
            mmap.MAP_SHARED,
            mmap.PROT_READ|mmap.PROT_WRITE,
            offset = self._get_v3d_addr()
            )
        os.close(fd)
        return base
            
    def _get_v3d_addr(self):
        lib = ctypes.cdll.LoadLibrary("libbcm_host.so")
        return lib.bcm_host_get_peripheral_address() + V3D_OFFSET_FROM_PERI
