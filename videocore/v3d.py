"""V3D registers.

This code is based on https://github.com/Terminus-IMRC/libvc4v3d
"""

import os
import ctypes
import mmap
import struct

V3D_OFFSET_FROM_PERI = 0x00c00000
V3D_LENGTH = 0x00f20 - 0x00000 + 32 / 8
V3D_REGISTER_TABLE = {
  'TVER': (0x00000, 24, 31, 'r'),
  'IDSTR_V': (0x00000, 0, 7, 'r'),
  'IDSTR_3': (0x00000, 8, 15, 'r'),
  'IDSTR_D': (0x00000, 16, 23, 'r'),
  'VPMSZ':(0x00004, 28, 31, 'r'),
  'HDRT': (0x00004, 24, 27, 'r'),
  'NSEM': (0x00004, 16, 23, 'r'),
  'TUPS': (0x00004, 12, 15, 'r'),
  'QUPS': (0x00004, 8, 11, 'r'),
  'NSLC': (0x00004, 4, 7, 'r'),
  'REV': (0x00004, 0, 3, 'r'),
  'TLBDB': (0x00008, 8, 11, 'r'),
  'TLBSZ': (0x00008, 4, 7, 'r'),
  'VRISZ': (0x00008, 0, 3, 'r'),
  'SCRATCH': (0x00010, 0, 31, 'rw'),
  'L2CACTL': (0x00020, 0, 2, 'rw'),
  'L2CCLR': (0x00020, 2, 2, 'w'),
  'L2CDIS': (0x00020, 1, 1, 'w'),
  'L2CENA': (0x00020, 0, 0, 'rw'),
  'T1CCS0': (0x00024, 27, 27, 'w'),
  'T1CCS1': (0x00024, 26, 26, 'w'),
  'T1CCS2': (0x00024, 25, 25, 'w'),
  'T1CCS3': (0x00024, 24, 24, 'w'),
  'T0CCS0': (0x00024, 19, 19, 'w'),
  'T0CCS1': (0x00024, 18, 18, 'w'),
  'T0CCS2': (0x00024, 17, 17, 'w'),
  'T0CCS3': (0x00024, 16, 16, 'w'),
  'UCCS0': (0x00024, 11, 11, 'w'),
  'UCCS1': (0x00024, 10, 10, 'w'),
  'UCCS2': (0x00024, 9, 9, 'w'),
  'UCCS3': (0x00024, 8, 8, 'w'),
  'ICCS0': (0x00024, 3, 3, 'w'),
  'ICCS1': (0x00024, 2, 2, 'w'),
  'ICCS2': (0x00024, 1, 1, 'w'),
  'ICCS3': (0x00024, 0, 0, 'w'),
  'INT_SPILLUSE': (0x00030, 3, 3, 'rw'),
  'INT_OUTTOMEM': (0x00030, 2, 2, 'rw'),
  'INT_FLDONE': (0x00030, 1, 1, 'rw'),
  'INT_FRDONE': (0x00030, 0, 0, 'rw'),
  'EI_SPILLUSE': (0x00034, 3, 3, 'rw'),
  'EI_OUTTOMEM': (0x00034, 2, 2, 'rw'),
  'EI_FLDONE': (0x00034, 1, 1, 'rw'),
  'EI_FRDONE': (0x00034, 0, 0, 'rw'),
  'DI_SPILLUSE': (0x00038, 3, 3, 'rw'),
  'DI_OUTTOMEM': (0x00038, 2, 2, 'rw'),
  'DI_FLDONE': (0x00038, 1, 1, 'rw'),
  'DI_FRDONE': (0x00038, 0, 0, 'rw'),
  'CT0CS_CTRSTA': (0x00100, 15, 15, 'w'),
  'CT0CS_CTSEMA': (0x00100, 12, 14, 'r'),
  'CT0CS_CTRTSD': (0x00100, 8, 9, 'r'),
  'CT0CS_CTRUN': (0x00100, 5, 5, 'rw'),
  'CT0CS_CTSUBS': (0x00100, 4, 4, 'rw'),
  'CT0CS_CTERR': (0x00100, 3, 3, 'r'),
  'CT0CS_CTMODE': (0x00100, 0, 0, 'r'),
  'CT1CS_CTRSTA': (0x00104, 15, 15, 'w'),
  'CT1CS_CTSEMA': (0x00104, 12, 14, 'r'),
  'CT1CS_CTRTSD': (0x00104, 8, 9, 'r'),
  'CT1CS_CTRUN': (0x00104, 5, 5, 'rw'),
  'CT1CS_CTSUBS': (0x00104, 4, 4, 'rw'),
  'CT1CS_CTERR': (0x00104, 3, 3, 'r'),
  'CT1CS_CTMODE': (0x00104, 0, 0, 'r'),
  'CT0EA_CTLEA': (0x00108, 0, 31, 'rw'),
  'CT1EA_CTLEA': (0x0010c, 0, 31, 'rw'),
  'CT0CA_CTLCA': (0x00110, 0, 31, 'rw'),
  'CT1CA_CTLCA': (0x00114, 0, 31, 'rw'),
  'CT00RA0_CTLRA': (0x00118, 0, 31, 'r'),
  'CT01RA0_CTLRA': (0x0011c, 0, 31, 'r'),
  'CT0LC_CTLLCM': (0x00120, 16, 31, 'rw'),
  'CT0LC_CTLSLCS': (0x00120, 0, 15, 'rw'),
  'CT1LC_CTLLCM': (0x00124, 16, 31, 'rw'),
  'CT1LC_CTLSLCS': (0x00124, 0, 15, 'rw'),
  'CT0PC_CTLPC': (0x00128, 0, 31, 'r'),
  'CT1PC_CTLPC': (0x0012c, 0, 31, 'r'),
  'BMOOM': (0x00130, 8, 8, 'r'),
  'RMBUSY': (0x00130, 3, 3, 'r'),
  'RMACTIVE': (0x00130, 2, 2, 'r'),
  'BMBUSY': (0x00130, 1, 1, 'r'),
  'BMACTIVE': (0x00130, 0, 0, 'r'),
  'BMFCT': (0x00134, 0, 7, 'rw'),
  'RMFCT': (0x00138, 0, 7, 'rw'),
  'BMPCA': (0x00300, 0, 31, 'r'),
  'BMPRS': (0x00304, 0, 31, 'r'),
  'BMPOA': (0x00308, 0, 31, 'rw'),
  'BMPOS': (0x0030c, 0, 31, 'rw'),
  'CLIPDISA': (0x00310, 1, 1, 'rw'),
  'FWDDISA': (0x00310, 0, 0, 'rw'),
  'QPURSV7': (0x00410, 28, 31, 'rw'),
  'QPURSV6': (0x00410, 24, 27, 'rw'),
  'QPURSV5': (0x00410, 20, 23, 'rw'),
  'QPURSV4': (0x00410, 16, 19, 'rw'),
  'QPURSV3': (0x00410, 12, 15, 'rw'),
  'QPURSV2': (0x00410, 8, 11, 'rw'),
  'QPURSV1': (0x00410, 4, 7, 'rw'),
  'QPURSV0': (0x00410, 0, 3, 'rw'),
  'QPURSV15': (0x00414, 28, 31, 'rw'),
  'QPURSV14': (0x00414, 24, 27, 'rw'),
  'QPURSV13': (0x00414, 20, 23, 'rw'),
  'QPURSV12': (0x00414, 16, 19, 'rw'),
  'QPURSV11': (0x00414, 12, 15, 'rw'),
  'QPURSV10': (0x00414, 8, 11, 'rw'),
  'QPURSV9': (0x00414, 4, 7, 'rw'),
  'QPURSV8': (0x00414, 0, 3, 'rw'),
  'CSRBL': (0x00418, 2, 3, 'rw'),
  'VSRBL': (0x00418, 0, 1, 'rw'),
  'QPURQPC': (0x00430, 0, 31, 'w'),
  'QPURQUA': (0x00434, 0, 31, 'rw'),
  'QPURQUL': (0x00438, 0, 11, 'rw'),
  'QPURQCC': (0x0043c, 16, 23, 'rw'),
  'QPURQCM': (0x0043c, 8, 15, 'rw'),
  'QPURQERR': (0x0043c, 7, 7, 'rw'),
  'QPURQL': (0x0043c, 0, 5, 'rw'),
  'VPATOEN': (0x00500, 13, 13, 'rw'),
  'VPALIMEN': (0x00500, 12, 12, 'rw'),
  'VPABATO': (0x00500, 9, 11, 'rw'),
  'VPARATO': (0x00500, 6, 8, 'rw'),
  'VPABALIM': (0x00500, 3, 5, 'rw'),
  'VPARALIM': (0x00500, 0, 2, 'rw'),
  'VPMURSV': (0x00504, 0, 4, 'rw'),
  'CTCLR': (0x00670, 0, 15, 'w'),
  'CTEN': (0x00674, 0, 31, 'rw'),
  'PCTR0': (0x00680, 0, 31, 'rw'),
  'PCTRS0': (0x00684, 0, 4, 'rw'),
  'PCTR1': (0x00688, 0, 31, 'rw'),
  'PCTRS1': (0x0068c, 0, 4, 'rw'),
  'PCTR2': (0x00690, 0, 31, 'rw'),
  'PCTRS2': (0x00694, 0, 4, 'rw'),
  'PCTR3': (0x00698, 0, 31, 'rw'),
  'PCTRS3': (0x0069c, 0, 4, 'rw'),
  'PCTR4': (0x006a0, 0, 31, 'rw'),
  'PCTRS4': (0x006a4, 0, 4, 'rw'),
  'PCTR5': (0x006a8, 0, 31, 'rw'),
  'PCTRS5': (0x006ac, 0, 4, 'rw'),
  'PCTR6': (0x006b0, 0, 31, 'rw'),
  'PCTRS6': (0x006b4, 0, 4, 'rw'),
  'PCTR7': (0x006b8, 0, 31, 'rw'),
  'PCTRS7': (0x006bc, 0, 4, 'rw'),
  'PCTR8': (0x006c0, 0, 31, 'rw'),
  'PCTRS8': (0x006c4, 0, 4, 'rw'),
  'PCTR9': (0x006c8, 0, 31, 'rw'),
  'PCTRS9': (0x006cc, 0, 4, 'rw'),
  'PCTR10': (0x006d0, 0, 31, 'rw'),
  'PCTRS10': (0x006d4, 0, 4, 'rw'),
  'PCTR11': (0x006d8, 0, 31, 'rw'),
  'PCTRS11': (0x006dc, 0, 4, 'rw'),
  'PCTR12': (0x006e0, 0, 31, 'rw'),
  'PCTRS12': (0x006e4, 0, 4, 'rw'),
  'PCTR13': (0x006e8, 0, 31, 'rw'),
  'PCTRS13': (0x006ec, 0, 4, 'rw'),
  'PCTR14': (0x006f0, 0, 31, 'rw'),
  'PCTRS14': (0x006f4, 0, 4, 'rw'),
  'PCTR15': (0x006f8, 0, 31, 'rw'),
  'PCTRS15': (0x006fc, 0, 4, 'rw'),
  'IPD2_FPDUSED': (0x00f00, 20, 20, 'r'),
  'IPD2_VALID': (0x00f00, 19, 19, 'r'),
  'MULIP2': (0x00f00, 18, 18, 'r'),
  'MULIP1': (0x00f00, 17, 17, 'r'),
  'MULIP0': (0x00f00, 16, 16, 'r'),
  'VR1_B': (0x00f00, 2, 2, 'r'),
  'VR1_A': (0x00f00, 1, 1, 'r'),
  'EZREQ_FIFO_ORUN': (0x00f04, 17, 17, 'r'),
  'EZVAL_FIFO_ORUN': (0x00f04, 15, 15, 'r'),
  'DEPTHO_ORUN': (0x00f04, 14, 14, 'r'),
  'DEPTHO_FIFO_ORUN': (0x00f04, 13, 13, 'r'),
  'REFXY_FIFO_ORUN': (0x00f04, 12, 12, 'r'),
  'ZCOEFF_FIFO_FULL': (0x00f04, 11, 11, 'r'),
  'XYRELW_FIFO_ORUN': (0x00f04, 10, 10, 'r'),
  'XYRELO_FIFO_ORUN': (0x00f04, 7, 7, 'r'),
  'FIXZ_ORUN': (0x00f04, 6, 6, 'r'),
  'XYFO_FIFO_ORUN': (0x00f04, 5, 5, 'r'),
  'QBSZ_FIFO_ORUN': (0x00f04, 4, 4, 'r'),
  'QBFR_FIFO_ORUN': (0x00f04, 3, 3, 'r'),
  'XYRELZ_FIFO_FULL': (0x00f04, 2, 2, 'r'),
  'WCOEFF_FIFO_FULL': (0x00f04, 1, 1, 'r'),
  'XYFO_FIFO_OP_READY': (0x00f08, 28, 28, 'r'),
  'QXYF_FIFO_OP_READY': (0x00f08, 27, 27, 'r'),
  'RAST_BUSY': (0x00f08, 26, 26, 'r'),
  'EZ_XY_READY': (0x00f08, 25, 25, 'r'),
  'EZ_DATA_READY': (0x00f08, 23, 23, 'r'),
  'ZRWPE_READY': (0x00f08, 7, 7, 'r'),
  'ZRWPE_STALL': (0x00f08, 6, 6, 'r'),
  'EDGES_CTRLID': (0x00f08, 3, 5, 'r'),
  'EDGES_ISCTRL': (0x00f08, 2, 2, 'r'),
  'EDGES_READY': (0x00f08, 1, 1, 'r'),
  'EDGES_STALL': (0x00f08, 0, 0, 'r'),
  'FIXZ_READY': (0x00f0c, 30, 30, 'r'),
  'RECIPW_READY': (0x00f0c, 28, 28, 'r'),
  'INTERPRW_READY': (0x00f0c, 27, 27, 'r'),
  'INTERPZ_READY': (0x00f0c, 24, 24, 'r'),
  'XYRELZ_FIFO_LAST': (0x00f0c, 23, 23, 'r'),
  'XYRELZ_FIFO_READY': (0x00f0c, 22, 22, 'r'),
  'XYNRM_LAST': (0x00f0c, 21, 21, 'r'),
  'XYNRM_READY': (0x00f0c, 20, 20, 'r'),
  'EZLIM_READY': (0x00f0c, 19, 19, 'r'),
  'DEPTHO_READY': (0x00f0c, 18, 18, 'r'),
  'RAST_LAST': (0x00f0c, 17, 17, 'r'),
  'RAST_READY': (0x00f0c, 16, 16, 'r'),
  'XYFO_FIFO_READY': (0x00f0c, 14, 14, 'r'),
  'ZO_FIFO_READY': (0x00f0c, 13, 13, 'r'),
  'XYRELO_FIFO_READY': (0x00f0c, 11, 11, 'r'),
  'WCOEFF_FIFO_READY': (0x00f0c, 7, 7, 'r'),
  'XYRELW_FIFO_READY': (0x00f0c, 6, 6, 'r'),
  'ZCOEFF_FIFO_READY': (0x00f0c, 5, 5, 'r'),
  'REFXY_FIFO_READY': (0x00f0c, 4, 4, 'r'),
  'DEPTHO_FIFO_READY': (0x00f0c, 3, 3, 'r'),
  'EZVAL_FIFO_READY': (0x00f0c, 2, 2, 'r'),
  'EZREQ_FIFO_READY': (0x00f0c, 1, 1, 'r'),
  'QXYF_FIFO_READY': (0x00f0c, 0, 0, 'r'),
  'ZO_FIFO_IP_STALL': (0x00f10, 28, 28, 'r'),
  'RECIPW_IP_STALL': (0x00f10, 25, 25, 'r'),
  'INTERPW_IP_STALL': (0x00f10, 22, 22, 'r'),
  'XYRELZ_FIFO_IP_STALL': (0x00f10, 18, 18, 'r'),
  'INTERPZ_IP_STALL': (0x00f10, 17, 17, 'r'),
  'DEPTHO_FIFO_IP_STALL': (0x00f10, 16, 16, 'r'),
  'EZLIM_IP_STALL': (0x00f10, 15, 15, 'r'),
  'XYNRM_IP_STALL': (0x00f10, 14, 14, 'r'),
  'EZREQ_FIFO_OP_VALID': (0x00f10, 13, 13, 'r'),
  'QXYF_FIFO_OP_VALID': (0x00f10, 12, 12, 'r'),
  'QXYF_FIFO_OP_LAST': (0x00f10, 11, 11, 'r'),
  'QXYF_FIFO_OP1_DUMMY': (0x00f10, 10, 10, 'r'),
  'QXYF_FIFO_OP1_LAST': (0x00f10, 9, 9, 'r'),
  'QXYF_FIFO_OP1_VALID': (0x00f10, 8, 8, 'r'),
  'EZTEST_ANYQVALID': (0x00f10, 7, 7, 'r'),
  'EZTEST_ANYQF': (0x00f10, 6, 6, 'r'),
  'EZTEST_QREADY': (0x00f10, 5, 5, 'r'),
  'EZTEST_VLF_OKNOVALID': (0x00f10, 4, 4, 'r'),
  'EZTEST_STALL': (0x00f10, 3, 3, 'r'),
  'EZTEST_IP_VLFSTALL': (0x00f10, 2, 2, 'r'),
  'EZTEST_IP_PRSTALL': (0x00f10, 1, 1, 'r'),
  'EZTEST_IP_QSTALL': (0x00f10, 0, 0, 'r'),
  'L2CARE': (0x00f20, 15, 15, 'r'),
  'VCMBE': (0x00f20, 14, 14, 'r'),
  'VCMRE': (0x00f20, 13, 13, 'r'),
  'VCDI': (0x00f20, 12, 12, 'r'),
  'VCDE': (0x00f20, 11, 11, 'r'),
  'VDWE': (0x00f20, 10, 10, 'r'),
  'VPMEAS': (0x00f20, 9, 9, 'r'),
  'VPMEFNA': (0x00f20, 8, 8, 'r'),
  'VPMEWNA': (0x00f20, 7, 7, 'r'),
  'VPMERNA': (0x00f20, 6, 6, 'r'),
  'VPMERR': (0x00f20, 5, 5, 'r'),
  'VPMEWR': (0x00f20, 4, 4, 'r'),
  'VPAERRGL': (0x00f20, 3, 3, 'r'),
  'VPAEBRGL': (0x00f20, 2, 2, 'r'),
  'VPAERGS': (0x00f20, 1, 1, 'r'),
  'VPAEABB': (0x00f20, 0, 0, 'r')
}

class V3DRegisters(object):
    def __init__(self):
        self.base = self._mmap_v3d_region()

    def close(self):
        self.base.close()

    def read_word(self, offs):
        return struct.unpack('I4', self.base[offs:offs+4])[0]

    def write_word(self, offs, v):
        self.base[offs:offs+4] = struct.pack('I4', v)

    def read(self, name):
        if name not in V3D_REGISTER_TABLE:
            raise Exception('Unknown V3D register {}'.format(name))
        offs, frm, to, prop = V3D_REGISTER_TABLE[name]
        if 'r' not in prop:
            raise Exception('{} is not readable'.format(name))
        v = self.read_word(offs)
        return (v >> frm) & ((1 << (to - frm + 1)) - 1)

    def write(self, name, v):
        if name not in V3D_REGISTER_TABLE:
            raise Exception('Unknown V3D register {}'.format(name))
        offs, frm, to, prop = V3D_REGISTER_TABLE[name]
        if 'w' not in prop:
            raise Exception('{} is not writable'.format(name))
        t = self.read_word(offs)
        w = to - frm + 1
        if v < 0 or v >= (1 << w):
            raise Exception('{} is not fit for {}-bit(s) register {}'.format(v, w, name))
        mask = ~(((1 << w) - 1) << frm)
        self.write_word(offs, (v << frm) | (t & mask))

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
