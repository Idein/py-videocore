import os, mmap
from ctypes import cdll
import numpy as np

class Register(object):
    def __init__(self, addr, mask, width):
        self.addr = (0xc00000 + addr) >> 2
        self.mask = mask
        self.width = width

# V3D Register Address Map
IDENT0  = Register(0x0000, 0xffffffff, 32)
IDENT1  = Register(0x0004, 0xffffffff, 32)
IDENT2  = Register(0x0008, 0xffffffff, 32)
IDENT3  = Register(0x000c, 0xffffffff, 32)
SCRATCH = Register(0x0010, 0xffffffff, 32)
L2CACTL = Register(0x0020, 0xffffffff, 32)
SLCACTL = Register(0x0024, 0xffffffff, 32)
INTCTL  = Register(0x0030, 0xffffffff, 32)
INTENA  = Register(0x0034, 0xffffffff, 32)
INTDIS  = Register(0x0038, 0xffffffff, 32)
CT0CS   = Register(0x0100, 0xffffffff, 32)
CT1CS   = Register(0x0104, 0xffffffff, 32)
CT0EA   = Register(0x0108, 0xffffffff, 32)
CT1EA   = Register(0x010c, 0xffffffff, 32)
CT0CA   = Register(0x0110, 0xffffffff, 32)
CT1CA   = Register(0x0114, 0xffffffff, 32)
CT00RA0 = Register(0x0118, 0xffffffff, 32)
CT01RA0 = Register(0x011c, 0xffffffff, 32)
CT0LC   = Register(0x0120, 0xffffffff, 32)
CT1LC   = Register(0x0124, 0xffffffff, 32)
CT0PC   = Register(0x0128, 0xffffffff, 32)
CT1PC   = Register(0x012c, 0xffffffff, 32)
PCS     = Register(0x0130, 0x0000013f,  9)
BFC     = Register(0x0134, 0x000000ff,  8)
RFC     = Register(0x0138, 0x000000ff,  8)
BPCA    = Register(0x0300, 0xffffffff, 32)
BPCS    = Register(0x0304, 0xffffffff, 32)
BPOA    = Register(0x0308, 0xffffffff, 32)
BPOS    = Register(0x030c, 0xffffffff, 32)
BXCF    = Register(0x0310, 0x00000003,  2)
SQRSV0  = Register(0x0410, 0xffffffff, 32)
SQRSV1  = Register(0x0414, 0xffffffff, 32)
SQCNTL  = Register(0x0418, 0x0000000f,  4)
SQCSTAT = Register(0x041c, 0xffffffff, 32)
SRQPC   = Register(0x0430, 0xffffffff, 32)
SRQUA   = Register(0x0434, 0xffffffff, 32)
SRQUL   = Register(0x0438, 0x00000fff, 12)
SRQCS   = Register(0x043c, 0x00ffffbf, 24)
VPACNTL = Register(0x0500, 0xffffffff, 32)
VPMBASE = Register(0x0504, 0xffffffff, 32)
PCTRC   = Register(0x0670, 0x0000ffff, 16)
PCTRE   = Register(0x0674, 0x8000ffff, 32)
PCTR0   = Register(0x0680, 0xffffffff, 32)
PCTRS0  = Register(0x0684, 0x0000001f,  5)
PCTR1   = Register(0x0688, 0xffffffff, 32)
PCTRS1  = Register(0x068c, 0x0000001f,  5)
PCTR2   = Register(0x0690, 0xffffffff, 32)
PCTRS2  = Register(0x0694, 0x0000001f,  5)
PCTR3   = Register(0x0698, 0xffffffff, 32)
PCTRS3  = Register(0x069c, 0x0000001f,  5)
PCTR4   = Register(0x06a0, 0xffffffff, 32)
PCTRS4  = Register(0x06a4, 0x0000001f,  5)
PCTR5   = Register(0x06a8, 0xffffffff, 32)
PCTRS5  = Register(0x06ac, 0x0000001f,  5)
PCTR6   = Register(0x06b0, 0xffffffff, 32)
PCTRS6  = Register(0x06b4, 0x0000001f,  5)
PCTR7   = Register(0x06b8, 0xffffffff, 32)
PCTRS7  = Register(0x06bc, 0x0000001f,  5)
PCTR8   = Register(0x06c0, 0xffffffff, 32)
PCTRS8  = Register(0x06c4, 0x0000001f,  5)
PCTR9   = Register(0x06c8, 0xffffffff, 32)
PCTRS9  = Register(0x06cc, 0x0000001f,  5)
PCTR10  = Register(0x06d0, 0xffffffff, 32)
PCTRS10 = Register(0x06d4, 0x0000001f,  5)
PCTR11  = Register(0x06d8, 0xffffffff, 32)
PCTRS11 = Register(0x06dc, 0x0000001f,  5)
PCTR12  = Register(0x06e0, 0xffffffff, 32)
PCTRS12 = Register(0x06e4, 0x0000001f,  5)
PCTR13  = Register(0x06e8, 0xffffffff, 32)
PCTRS13 = Register(0x06ec, 0x0000001f,  5)
PCTR14  = Register(0x06f0, 0xffffffff, 32)
PCTRS14 = Register(0x06f4, 0x0000001f,  5)
PCTR15  = Register(0x06f8, 0xffffffff, 32)
PCTRS15 = Register(0x06fc, 0x0000001f,  5)
DBCFG   = Register(0x0e00, 0xffffffff, 32)
DBSCS   = Register(0x0e04, 0xffffffff, 32)
DBSCFG  = Register(0x0e08, 0xffffffff, 32)
DBSSR   = Register(0x0e0c, 0xffffffff, 32)
DBSDR0  = Register(0x0e10, 0xffffffff, 32)
DBSDR1  = Register(0x0e14, 0xffffffff, 32)
DBSDR2  = Register(0x0e18, 0xffffffff, 32)
DBSDR3  = Register(0x0e1c, 0xffffffff, 32)
DBQRUN  = Register(0x0e20, 0xffffffff, 32)
DBQHLT  = Register(0x0e24, 0xffffffff, 32)
DBQSTP  = Register(0x0e28, 0xffffffff, 32)
DBQITE  = Register(0x0e2c, 0xffffffff, 32)
DBQITC  = Register(0x0e30, 0xffffffff, 32)
DBQGHC  = Register(0x0e34, 0xffffffff, 32)
DBQGHG  = Register(0x0e38, 0xffffffff, 32)
DBQGHH  = Register(0x0e3c, 0xffffffff, 32)
DBGE    = Register(0x0f00, 0xffffffff, 32)
FDBG0   = Register(0x0f04, 0xffffffff, 32)
FDBGB   = Register(0x0f08, 0xffffffff, 32)
FDBGR   = Register(0x0f0c, 0xffffffff, 32)
FDBGS   = Register(0x0f10, 0xffffffff, 32)
ERRSTAT = Register(0x0f20, 0xffffffff, 32)

_PCREGS = [
    PCTR0 , PCTR1 , PCTR2 , PCTR3 ,
    PCTR4 , PCTR5 , PCTR6 , PCTR7 ,
    PCTR8 , PCTR9 , PCTR10, PCTR11,
    PCTR12, PCTR13, PCTR14, PCTR15,
]

_PCSREGS = [
    PCTRS0 , PCTRS1 , PCTRS2 , PCTRS3 ,
    PCTRS4 , PCTRS5 , PCTRS6 , PCTRS7 ,
    PCTRS8 , PCTRS9 , PCTRS10, PCTRS11,
    PCTRS12, PCTRS13, PCTRS14, PCTRS15,
]

class RegisterMapping(object):
    def __init__(self, driver, library_path = '/opt/vc/lib'):
        self.lib = cdll.LoadLibrary('{}/libbcm_host.so'.format(library_path))
        self.peri = None

    def __enter__(self):
        self.lib.bcm_host_init()
        peri_addr = self.lib.bcm_host_get_peripheral_address()
        peri_size = self.lib.bcm_host_get_peripheral_size()
        page_size = os.sysconf("SC_PAGE_SIZE")
        assert(not (peri_addr & (page_size - 1)))
        fd = os.open('/dev/mem', os.O_RDWR)
        self.peri = mmap.mmap(fd, peri_size, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE, offset = peri_addr)
        self.peri_arr = np.frombuffer(self.peri, dtype=np.uint32)
        os.close(fd)
        return self

    def __exit__(self, type, value, traceback):
        self.peri.close()
        self.peri = None
        self.lib.bcm_host_deinit()

    def read(self, register):
        return self.peri_arr[register.addr]

    def write(self, register, value):
        self.peri_arr[register.addr] = value

class PerformanceCounter(object):
    def __init__(self, regmap, pcs):
        self.regmap = regmap
        self.pcs = pcs

    def __enter__(self):
        self.regmap.write(PCTRE, 0)
        self.regmap.write(PCTRC, 0xffff)
        for reg in _PCREGS:
            self.regmap.write(reg, 0)
        for reg in _PCSREGS:
            self.regmap.write(reg, 0)
        for i, pc in enumerate(self.pcs):
            self.regmap.write(_PCSREGS[i], pc)
        self.regmap.write(PCTRE, 0x80000000 | ((1 << len(self.pcs)) - 1))
        return self

    def result(self):
        return [ self.regmap.read(_PCREGS[i]) for i in range(len(self.pcs)) ]

    def __exit__(self, type, value, traceback):
        self.regmap.write(PCTRE, 0)
        self.regmap.write(PCTRC, 0xffff)
