from ctypes import Structure, c_ulong, string_at, byref, sizeof, c_char, c_size_t, POINTER
from struct import pack, unpack

class AssembleError(Exception):
    'Exception related to QPU assembler'

#============================== Encoding tables ===============================

# Signaling bits.
_SIGNAL = {
    name: code
    for code, name in enumerate([
        'breakpoint', 'no signal', 'thread switch', 'thread end',
        'wait scoreboard', 'unlock scoreboard', 'last thread switch',
        'load coverage', 'load color', 'load color and thread end',
        'load tmu0', 'load tmu1', 'load alpha', 'alu small imm', 'load',
        'branch'
    ])}

# Add ALU instructions
_ADD_INSN = {
    name: code
    for code, name in enumerate([
        'nop', 'fadd', 'fsub', 'fmin', 'fmax', 'fminabs', 'fmaxabs', 'ftoi',
        'itof', '', '', '', 'iadd', 'isub', 'shr', 'asr', 'ror', 'shl', 'imin',
        'imax', 'band', 'bor', 'bxor', 'bnot', 'clz', '', '', '', '', '',
        'v8adds', 'v8subs'
    ]) if name
    }

# Mul ALU instructions
_MUL_INSN = {
    name: code
    for code, name in enumerate([
        'nop', 'fmul', 'imul24', 'v8muld', 'v8min', 'v8max', 'v8adds', 'v8subs'
    ])}


# Branch instructions
_BRANCH_INSN = {
    name: code
    for code, name in enumerate([
        'jzs', 'jzc', 'jzs_any', 'jzc_any', 'jns', 'jnc', 'jns_any', 'jnc_any',
        'jcs', 'jcc', 'jcs_any', 'jcc_any', '', '', '', 'jmp'
    ]) if name
    }


# Small immediate values.
_SMALL_IMM = {
    value: code
    for code, value in enumerate(
        [repr(i) for i in range(16)] +       # 0,1,2,..,15
        [repr(i) for i in range(-16, 0)] +   # -16,-15,..,-1
        [repr(2.0**i) for i in range(8)] +   # 1.0,2.0,..,128.0
        [repr(2.0**i) for i in range(-8, 0)] # 1.0/256.0,1.0/128.0,..,1.0/2.0
    )}
_SMALL_IMM['0.0'] = 0     # 0.0 and 0 have same code

# Condition codes.
_COND = {
    name: code
    for code, name in enumerate([
        'never', 'always', 'zs', 'zc', 'ns', 'nc', 'cs', 'cc'
    ])}


# Input multiplexers.
# 'A' specifies raddr_a. 'B' specifies raddr_b.
_INPUT_MUXES = {
    'r0': 0, 'r1': 1, 'r2': 2, 'r3': 3, 'r4': 4, 'r5': 5, 'A': 6, 'B': 7
    }

# Packing. See Register.pack.
_PACK = {
    op: code
    for code, op in enumerate([
        'nop', '16a', '16b', 'rep 8', '8a', '8b', '8c', '8d', '32 sat',
        '16a sat', '16b sat', 'rep 8 sat', '8a sat', '8b sat', '8c sat',
        '8d sat'
    ])}

# Unpacking. See Register.unpack.
_UNPACK = {
    op: code
    for code, op in enumerate([
        'nop', '16a', '16b', 'rep 8d', '8a', '8b', '8c', '8d'
    ])}

# Mul ALU packing. See MulEmitter._emit.
_MUL_PACK = {
    'nop': 0,
    'rep 8': 3,
    '8a': 4 ,
    '8b': 5,
    '8c': 6,
    '8d': 7
    }

def rev(d):
  return {v:k for k, v in d.items()}

_SIGNAL_REV = rev(_SIGNAL)
_ADD_INSN_REV = rev(_ADD_INSN)
_MUL_INSN_REV = rev(_MUL_INSN)
_BRANCH_INSN_REV = rev(_BRANCH_INSN)
_COND_REV = rev(_COND)


#=================================== Register =================================


# Flags to specify locations of registers.
_REG_AR = 1 << 3   # Regfile A read location
_REG_BR = 1 << 2   # Regfile B read location
_REG_AW = 1 << 1   # Regfile A write location
_REG_BW = 1 << 0   # Regfile B write location

class Register(object):
    """QPU Registers.

    This class implements general purpuse registers, register mapped I/O
    locations and accumulators.
    """

    def __init__(self, name, addr, spec, pack=0, unpack=0, pm=False):
        self.name = name
        self.addr = addr
        self.spec = spec
        self.pack_bits = pack
        self.unpack_bits = unpack
        self.pm_bit = pm

    def __str__(self):
        return self.name

    def pack(self, op):
        """Regfile-A pack.

        Call in write locations like this.

        >>> iadd(ra1.pack('16a'), ra0, rb0)

        In case of this example, QPU converts the result of ra0 + rb0 to int16
        then stores it to lower 16 bits of ra1.  The numbers and characters
        (abcd) in operation codes specifies location of registers where the
        packed bits to be stored. 'a' is for the least bits and 'd' for the
        highest.

        Operation codes with 'sat' suffix instruct to perform *saturation
        arithmetic*.

        :param op: One of the following strings to specify unpack operation.
            * 'nop': no operation
            * '16a', '16b':
                Convert float32 to float16 for floating-point arithmetic, int32
                to int16 for others.
            * '8a', '8b', '8c', '8d':
                Convert int32 to uint8.
            * 'rep 8':
                Convert int32 to uint8 then replicate it 4 times accross word.
            * '32 sat', '16a sat', '16b sat', 'rep 8 sat', '8a sat', '8b sat',
              '8c sat', '8d sat7:
                Saturation arithmetic version of former codes.

        See section 3 of the reference guide for details.
        """

        if (self.name in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5'] or
            not (self.spec & _REG_AW)):
            raise AssembleError(
                'Packing is only available for regfile-A write locations'
                )

        return Register(self.name, self.addr, _REG_AW, pack=_PACK[op], pm=0)

    def unpack(self, op):
        """Regfile-A unpack and r4 unpack.

        Call in read locations like this.

        >>> iadd(r0, ra0.unpack('16a'), rb0)

        In case of this example, QPU converts int16, lower 16 bits of ra0, to
        int32 before the addition.  The numbers and characters (abcd) in
        operation codes indicate bits of registers to be converted.  'a' is for
        the least bits and 'd' for the highest.

        :param op: One of the following strings to indicate unpack operation.
            * 'nop': no operation
            * '16a', '16b':
                Convert float16 to float32 for floating-point arithmetic, int16
                to int32 for others.
            * '8a', '8b', '8c', '8d':
                Convert uint8 to float32 in range [0.0, 1.0] for floating-point
                arithmetic, uint8 to int32 for others.
            * 'rep 8d':
                Replicate MS byte 4 times accores word.

        See section 3 of the reference guide for details.
        """

        if self.name != 'r4' and not (self.spec & _REG_AR):
            raise AssembleError(
                'Unpacking is only available for regfile-A read locations'
                ' and accumulator r4'
                )

        if self.name == 'r4':
            spec = self.spec
            pm = 1
        else:
            spec = _REG_AR
            pm = 0

        return Register(self.name, self.addr, spec, unpack=_UNPACK[op], pm=pm)

# Table of registers

# There are 32 general purpose registers in each regfile A and B.
GENERAL_PURPOSE_REGISTERS = {
    name: Register(
        name, addr,
        {'a': _REG_AR|_REG_AW,'b': _REG_BR|_REG_BW}[regfile]
        )
    for regfile in ['a', 'b']
    for addr in range(32)
    for name in ['r' + regfile + str(addr)]
    }

IO_REGISTERS = {
    name: Register(name, addr, spec)
    for name, addr, spec in [
        ('uniform'           , 32 , _REG_AR|_REG_BR),
        ('varying_read'      , 35 , _REG_AR|_REG_BR),
        ('tmu_noswap'        , 36 , _REG_AW|_REG_BW),
        ('r5_pix0'           , 37 , _REG_AW),
        ('broadcast'         , 37 , _REG_BW),
        ('host_interrupt'    , 38 , _REG_AW|_REG_BW),
        ('element_number'    , 38 , _REG_AR),
        ('qpu_number'        , 38 , _REG_BR),
        ('null'              , 39 , _REG_AR|_REG_BR|_REG_AW|_REG_BW),
        ('uniforms_address'  , 40 , _REG_AW),
        ('x_pixel_coord'     , 41 , _REG_AR),
        ('y_pixel_coord'     , 41 , _REG_BR),
        ('quad_x'            , 41 , _REG_AW),
        ('quad_y'            , 41 , _REG_BW),
        ('ms_flags'          , 42 , _REG_AR|_REG_AW),
        ('rev_flag'          , 42 , _REG_BR|_REG_BW),
        ('tlb_stencil_setup' , 43 , _REG_AW|_REG_BW),
        ('tlb_z'             , 44 , _REG_AW|_REG_BW),
        ('tlb_color_ms'      , 45 , _REG_AW|_REG_BW),
        ('tlb_color_all'     , 46 , _REG_AW|_REG_BW),
        ('tlb_alpha_mask'    , 47 , _REG_AW|_REG_BW),
        ('vpm'               , 48 , _REG_AR|_REG_BR|_REG_AW|_REG_BW),
        ('vpm_ld_busy'       , 49 , _REG_AR),
        ('vpm_st_busy'       , 49 , _REG_BR),
        ('vpmvcd_rd_setup'   , 49 , _REG_AW),
        ('vpmvcd_wr_setup'   , 49 , _REG_BW),
        ('vpm_ld_wait'       , 50 , _REG_AR),
        ('vpm_st_wait'       , 50 , _REG_BR),
        ('vpm_ld_addr'       , 50 , _REG_AW),
        ('vpm_st_addr'       , 50 , _REG_BW),
        ('mutex'             , 51 , _REG_AR|_REG_BR|_REG_AW|_REG_BW),
        ('sfu_recip'         , 52 , _REG_AW|_REG_BW),
        ('sfu_recipsqrt'     , 53 , _REG_AW|_REG_BW),
        ('sfu_exp2'          , 54 , _REG_AW|_REG_BW),
        ('sfu_log2'          , 55 , _REG_AW|_REG_BW),
        ('tmu0_s'            , 56 , _REG_AW|_REG_BW),
        ('tmu0_t'            , 57 , _REG_AW|_REG_BW),
        ('tmu0_r'            , 58 , _REG_AW|_REG_BW),
        ('tmu0_b'            , 59 , _REG_AW|_REG_BW),
        ('tmu1_s'            , 60 , _REG_AW|_REG_BW),
        ('tmu1_t'            , 61 , _REG_AW|_REG_BW),
        ('tmu1_r'            , 62 , _REG_AW|_REG_BW),
        ('tmu1_b'            , 63 , _REG_AW|_REG_BW)
    ]}

ACCUMULATORS = {
    name: Register(name, addr, spec)
    for name, addr, spec in [
        ('r0', 32, _REG_AW|_REG_BW),
        ('r1', 33, _REG_AW|_REG_BW),
        ('r2', 34, _REG_AW|_REG_BW),
        ('r3', 35, _REG_AW|_REG_BW),
        ('r4', -1, 0),
        ('r5', -1, 0),
    ]}

REGISTERS = {}
REGISTERS.update(GENERAL_PURPOSE_REGISTERS)
REGISTERS.update(IO_REGISTERS)
REGISTERS.update(ACCUMULATORS)

#============================ Default Arguments ============================


_r0 = REGISTERS['r0']
__d = _ADD_INSN

_ADD_DEFAULT_ARGS = {
    __d['nop']  : [_r0, _r0, _r0],
    __d['itof'] : [_r0],
    __d['ftoi'] : [_r0],
    __d['bnot'] : [_r0],
    __d['clz']  : [_r0]
}

__d = _MUL_INSN
_MUL_DEFAULT_ARGS = {
    __d['nop'] : [_r0, _r0, _r0]
}


#============================ Instruction encoding ============================


class Insn(Structure):
    'Instruction encoding.'

    def to_bytes(self):
        'Encode instruction to string.'
        return string_at(byref(self), sizeof(self))

    @classmethod
    def from_bytes(self, buf):
        'Decode string (or buffer object of length 64 bit) to instruction.'
        bytes, = unpack('Q', buf)
        sig = bytes >> 60
        if sig == _SIGNAL['branch']:
            return BranchInsn.from_buffer_copy(buf)
        elif sig == _SIGNAL['load']:
            if (bytes >> 57) & 0x7 == 4:
                return SemaInsn.from_buffer_copy(buf)
            else:
                return LoadInsn.from_buffer_copy(buf)
        else:
            return AluInsn.from_buffer_copy(buf)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            all(getattr(self, f) == getattr(other, f)
                for f, _, _ in self._fields_ if f != 'dontcare'
                ))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return '{class_name}({fields})'.format(
            class_name=self.__class__.__name__,
            fields =', '.join(
                    f + '=' + "0x%x" % getattr(self, f)
                    for f, _, _ in reversed(self._fields_) if f != 'dontcare'
                ))

class AluInsn(Insn):
    _fields_ = [ (f, c_ulong, n) for f, n in [
        ('mul_b', 3), ('mul_a', 3), ('add_b', 3), ('add_a', 3), ('raddr_b', 6),
        ('raddr_a', 6), ('op_add', 5), ('op_mul', 3), ('waddr_mul', 6),
        ('waddr_add', 6), ('ws', 1), ('sf', 1), ('cond_mul', 3),
        ('cond_add', 3), ('pack', 4), ('pm', 1), ('unpack', 3), ('sig', 4)
        ]]

class BranchInsn(Insn):
    _fields_ = [ (f, c_ulong, n) for f, n in [
        ('immediate', 32), ('waddr_mul', 6), ('waddr_add', 6), ('ws', 1),
        ('raddr_a', 5), ('reg', 1), ('rel', 1), ('cond_br', 4),
        ('dontcare', 4), ('sig', 4)
        ]]

class LoadInsn(Insn):
    _fields_ = [ (f, c_ulong, n) for f, n in [
        ('immediate', 32), ('waddr_mul', 6), ('waddr_add', 6), ('ws', 1),
        ('sf', 1), ('cond_mul', 3), ('cond_add', 3), ('pack', 4), ('pm', 1),
        ('unpack', 3), ('sig', 4)
        ]]

class SemaInsn(Insn):
    _fields_ = [ (f, c_ulong, n) for f, n in [
        ('semaphore', 4), ('sa', 1), ('dontcare', 27), ('waddr_mul', 6),
        ('waddr_add', 6), ('ws', 1), ('sf', 1), ('cond_mul', 3),
        ('cond_add', 3), ('pack', 4), ('pm', 1), ('unpack', 3), ('sig', 4)
        ]]

class RawInsn(Insn):
    _fields_ = [ (f, c_ulong, n) for f, n in [('raw1', 32), ('raw2', 32)] ]

