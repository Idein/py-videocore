"""VideoCore IV QPU assembler.

This module implements an assembly language for VideoCore IV QPU and its
assembler. We took an approach that the language is implemented as an internal
DSL of Python language.

The QPU instruction set is described in the section 3 of the following document
`VideoCore(R) IV 3D Architecture Reference Guide
<https://www.broadcom.com/docs/support/videocore/VideoCoreIV-AG100-R.pdf>`__
"""

from functools import partial
from ctypes import Structure, c_ulong, string_at, byref, sizeof
from struct import pack, unpack
import inspect
import ast

import numpy


class _partialmethod(partial):
    'A descriptor for methods behaves like :py:class:`functools.partial.`'
    def __get__(self, obj, type):
        return partial(self.func, obj,
                       *(self.args or ()), **(self.keywords or {}))

class AssembleError(Exception):
    'Exception related to QPU assembler'


#=================================== Register =================================

# Flags to specify locations of registers.
_REG_AR = 1 << 3   # Regfile A read location
_REG_BR = 1 << 2   # Regfile B read location
_REG_AW = 1 << 1   # Regfile A write location
_REG_BW = 1 << 0   # Regfile B write location

# See Register.unpack.
_UNPACK = {
    op: code for code, op in enumerate([
        'nop', '16a', '16b', 'rep 8d', '8a', '8b', '8c', '8d'
        ])
    }

# See Register.pack.
_PACK = {
    op: code for code, op in enumerate([
        'nop', '16a', '16b', 'rep 8', '8a', '8b', '8c', '8d', '32 sat',
        '16a sat', '16b sat', 'rep 8 sat', '8a sat', '8b sat', '8c sat',
        '8d sat'
        ])
    }

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

        In case of this example, QPU converts int16, lower 16 bits of ra1, to
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
        ('host_interrupt'    , 38 , _REG_AW|_REG_BW),
        ('element_number'    , 38 , _REG_AR),
        ('qpu_number'        , 38 , _REG_BR),
        ('null'              , 39 , _REG_AR|_REG_BR|_REG_AW|_REG_BW),
        ('uniforms_address'  , 40 , _REG_AW|_REG_BW),
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
        ('mutex_acquire'     , 51 , _REG_AR|_REG_BR),
        ('mutex_release'     , 51 , _REG_AW|_REG_BW),
        ('sfu_recip'         , 52 , _REG_AW|_REG_BW),
        ('sfu_recipsqrt'     , 53 , _REG_AW|_REG_BW),
        ('sfu_exp'           , 54 , _REG_AW|_REG_BW),
        ('sfu_log'           , 55 , _REG_AW|_REG_BW),
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
        ('r5', 37, _REG_AW|_REG_BW),
    ]}

REGISTERS = {}
REGISTERS.update(GENERAL_PURPOSE_REGISTERS)
REGISTERS.update(IO_REGISTERS)
REGISTERS.update(ACCUMULATORS)


#============================ Instruction encoding ============================

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
                for f, _, _ in self._fields_)
            )

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return (
            self.__class__.__name__ + '(' +
            ', '.join(
                f + '=' + hex(getattr(self, f))
                for f, _, _ in reversed(self._fields_)
                ) +
            ')')

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

# Encoding of small immediate values.
_SMALL_IMMED = {
    value: code
    for code, value in enumerate(
        [repr(i) for i in range(16)] +       # 0,1,2,..,15
        [repr(i) for i in range(-16, 0)] +   # -16,-15,..,-1
        [repr(2.0**i) for i in range(8)] +   # 1.0,2.0,..,128.0
        [repr(2.0**i) for i in range(-8, 0)] # 1.0/256.0,1.0/128.0,..,1.0/2.0
    )}
_SMALL_IMMED['0.0'] = 0     # 0.0 and 0 have same code

def pack_small_imm(val):
    """Pack 'val' to 6-bit array for ALU instruction with small immediates.

    >>> pack_small_imm(1)
    1
    >>> pack_small_imm(-2)
    30
    >>> pack_small_imm(1.0)
    32
    >>> pack_small_imm(1.0/256.0)
    40
    >>> pack_small_imm(1.2)
    Traceback (most recent call last):
    ...
    AssembleError: Immediate operand 1.2 is not allowed
    """

    code = _SMALL_IMMED.get(repr(val))
    if code is None:
        raise AssembleError('Immediate operand {} is not allowed'.format(val))
    return code

def pack_imm(val):
    """ Pack 'val' to 32-bit array for load and branch instructions.

    This function packs 'val' to 32-bit array for the immediate field of load and branch
    instructions. It returns the packed value with 'unpack' bits.

    >>> pack_imm(1)
    (1, 0)
    >>> pack_imm(-3)
    (4294967293L, 0)
    >>> pack_imm(1.3)
    (1067869798, 0)
    >>> pack_imm([3, 3, 1, 1, 0, 2, 3, 3, 1, 3, 0, 2, 2, 1, 0, 1])
    (3344495557L, 3)
    >>> pack_imm([-2,  1,  1,  1, -2,  0,  0,  1, -1,  1, -1, -2,  1,  1,  1, -1])
    (2293330415L, 1)
    >>> pack_imm('hello')
    Traceback (most recent call last):
    ...
    AssembleError: Unsupported immediate value hello
    >>> pack_imm([4,2,3])
    Traceback (most recent call last):
    ...
    AssembleError: 4 is not a 2-bit unsigned value
    >>> pack_imm([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    Traceback (most recent call last):
    ...
    AssembleError: Too many values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    """

    if isinstance(val, float):
        return unpack('L', pack('f', val))[0], 0x0
    elif isinstance(val, (int, long)):
        fmt = 'l' if val < 0 else 'L'
        return unpack('L', pack(fmt, val))[0], 0x0
    elif not isinstance(val, (list, tuple, numpy.ndarray)):
        raise AssembleError('Unsupported immediate value {}'.format(val))

    # per-element immediate
    values = list(val)
    if len(values) > 16:
        raise AssembleError('Too many values {}'.format(val))

    values.extend([0] * (16-len(values)))
    signed = any(map(lambda x: x < 0, values))
    high = 0
    low  = 0
    for i in range(16):
        high <<= 1
        low  <<= 1
        v = values[i]
        if (signed and (v >= 2 or v < -2)) or (not signed and v >= 4):
            raise AssembleError('{} is not a 2-bit {} value'.format(v, 'signed' if signed else 'unsigned'))
        high |= (v & 0x2) >> 1
        low  |= v & 0x1
    return (high << 16) | low, 2*(not signed) + 1

ADD_INSTRUCTIONS = ['nop', 'fadd', 'fsub', 'fmin', 'fmax', 'fminabs', 'fmaxabs', 'ftoi', 'itof',
    '', '', '', 'iadd', 'isub', 'shr', 'asr', 'ror', 'shl', 'imin', 'imax', 'band', 'bor',
    'bxor', 'bnot', 'clz', '', '', '', '', '', 'v8adds', 'v8subs'
    ]

MUL_INSTRUCTIONS = ['nop', 'fmul', 'mul24', 'v8muld', 'v8min', 'v8max', 'v8adds', 'v8subs']

BRANCH_INSTRUCTIONS = ['jz', 'jnz', 'jz_any', 'jnz_any', 'jn', 'jnn', 'jn_any', 'jnn_any',
        'jc', 'jnc', 'jc_any', 'jnc_any', '', '', '', 'jmp']

ACCUMURATOR_CODES =  {'r0': 0, 'r1': 1, 'r2': 2, 'r3': 3, 'r4': 4, 'r5': 5}
INPUT_MUX_REGFILE_A = 6
INPUT_MUX_REGFILE_B = 7

def locate_read_operands(add1 = REGISTERS['r0'], add2 = REGISTERS['r0'],
        mul1 = REGISTERS['r0'], mul2 = REGISTERS['r0']):
    """Locate read operands of add and mul instructions properly.
    
    >>> locate_read_operands()
    [0, 0, 0, 0, 39, 39, False, 0, 0]
    >>> locate_read_operands(r1, r2, r3, r4)
    [1, 2, 3, 4, 39, 39, False, 0, 0]
    >>> locate_read_operands(ra1, rb6)
    [6, 7, 0, 0, 1, 6, False, 0, 0]
    >>> locate_read_operands(rb6, ra1)
    [7, 6, 0, 0, 1, 6, False, 0, 0]
    >>> locate_read_operands(r0, ra1, rb2, r3)
    [0, 6, 7, 3, 1, 2, False, 0, 0]
    >>> locate_read_operands(ra1, ra1)
    [6, 6, 0, 0, 1, 39, False, 0, 0]
    >>> locate_read_operands(rb1, rb1)
    [7, 7, 0, 0, 39, 1, False, 0, 0]
    >>> locate_read_operands(ra1, rb1, ra1, rb1)
    [6, 7, 6, 7, 1, 1, False, 0, 0]
    >>> locate_read_operands(ra1, 1.0)
    [6, 7, 0, 0, 1, 32, True, 0, 0]
    >>> locate_read_operands(r0, 1.0, ra1, 1.0)
    [0, 7, 6, 7, 1, 32, True, 0, 0]
    >>> locate_read_operands(ra1, ra2)
    Traceback (most recent call last):
    ...
    AssembleError: Too many regfile A operand ra2
    >>> locate_read_operands(rb1, rb2)
    Traceback (most recent call last):
    ...
    AssembleError: Too many regfile B operand rb2
    >>> locate_read_operands(1.0, rb1)
    Traceback (most recent call last):
    ...
    AssembleError: Too many regfile B operand rb1
    >>> locate_read_operands(ra1.pack('16a'), rb1)
    Traceback (most recent call last):
    ...
    AssembleError: Packing of read operand
    >>> locate_read_operands(r4.unpack('16a'))
    [4, 0, 0, 0, 39, 39, False, 1, True]
    >>> locate_read_operands(ra1, rb1.unpack('16b'))
    Traceback (most recent call last):
    ...
    AssembleError: Unpacking is not supported for the register rb1
    >>> locate_read_operands(ra1.unpack('16a'), ra1.unpack('16a'))
    [6, 6, 0, 0, 1, 39, False, 1, False]
    >>> locate_read_operands(ra1.unpack('16a'), ra1.unpack('16b'))
    Traceback (most recent call last):
    ...
    AssembleError: Multiple unpacking
    """

    operands = [add1, add2, mul1, mul2]
    mux     = [None, None, None, None]
    raddr_a = None
    raddr_b = None
    immed   = False
    unpack  = 0
    pm      = 0

    for opd in operands:
        if not isinstance(opd, Register):
            continue

        if opd.pack_bits:
            raise AssembleError('Packing of read operand')
        if opd.unpack_bits:
            if unpack != 0 and opd.unpack_bits != unpack:
                raise AssembleError('Multiple unpacking')
            unpack = opd.unpack_bits
            pm     = opd.pm_bit

    for i, opd in enumerate(operands):
        # When opd is an accumurator register, raddr_a and raddr_b is not used for it.
        if isinstance(opd, Register) and opd.name in ACCUMURATOR_CODES:
            mux[i]  = ACCUMURATOR_CODES[opd.name]

    if all(map(lambda x: x is not None, mux)):
        return mux + [REGISTERS['null'].addr]*2 + [False, unpack, pm]

    # Locate operands whose regfile is uniquely specified.
    for i, opd in enumerate(operands):
        if mux[i] is not None: continue

        if not isinstance(opd, Register):
            imm_value = pack_small_imm(opd)
            if raddr_b is not None and not (immed and raddr_b == imm_value):
                raise AssembleError('Too many regfile B operand {}'.format(opd))
            raddr_b = imm_value
            mux[i]  = INPUT_MUX_REGFILE_B
            immed   = True
        elif (opd.spec & _REG_AR) and not (opd.spec & _REG_BR):
            if raddr_a is not None and raddr_a != opd.addr:
                raise AssembleError('Too many regfile A operand {}'.format(opd))
            raddr_a = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_A
        elif not (opd.spec & _REG_AR) and (opd.spec & _REG_BR):
            if raddr_b is not None and raddr_b != opd.addr:
                raise AssembleError('Too many regfile B operand {}'.format(opd))
            raddr_b = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_B

    # Locate remaining operands.
    for i, opd in enumerate(operands):
        if mux[i] is not None: continue

        if not (opd.spec & (_REG_AR | _REG_BR)):
            raise AssembleError('{} can not be used as a read operand'.format(opd))

        if raddr_a is None and opd.spec & _REG_AR:
            raddr_a = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_A
        elif raddr_b is None and opd.spec & _REG_BR:
            raddr_b = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_B
        else:
            raise AssembleError('Too many regfile operand {}'.format(opd))

    if raddr_a is None:
        raddr_a = REGISTERS['null'].addr
    if raddr_b is None:
        raddr_b = REGISTERS['null'].addr

    return mux + [raddr_a, raddr_b] + [immed, unpack, pm]

def locate_write_operands(add_dst = REGISTERS['null'], mul_dst = REGISTERS['null']):
    """Locate write operands of add and mul instructions properly.

    >>> locate_write_operands(ra1, rb2)
    (1, 2, False, 0, False)
    >>> locate_write_operands(rb2, ra1)
    (2, 1, True, 0, False)
    >>> locate_write_operands(ra1, ra2)
    Traceback (most recent call last):
    ...
    AssembleError: ra1 and ra2 are not proper combination of destination registers
    >>> locate_write_operands(ra1.pack('16a'))
    (1, 39, False, 1, False)
    >>> locate_write_operands(ra1.pack('16a'), rb1.pack('8888 mul'))
    Traceback (most recent call last):
    ...
    AssembleError: Too many packing
    """
    pack = 0
    pm   = False
    if add_dst.pack_bits:
        if mul_dst.pack_bits:
            raise AssembleError('Too many packing')
        pack = add_dst.pack_bits
    elif mul_dst.pack_bits:
        if add_dst.pack_bits:
            raise AssembleError('Too many packing')
        pack = mul_dst.pack_bits
        pm = True

    if add_dst.spec & _REG_AW and mul_dst.spec & _REG_BW:
        return add_dst.addr, mul_dst.addr, False, pack, pm
    elif mul_dst.spec & _REG_AW and add_dst.spec & _REG_BW:
        return add_dst.addr, mul_dst.addr, True, pack, pm
    raise AssembleError('{} and {} are not proper combination of destination registers'.format(add_dst, mul_dst))

INSTRUCTION_ALIASES = []
def syntax_sugar(f):
    INSTRUCTION_ALIASES.append(f.__name__)
    return f

_MUL_PACK = {
    'nop': 0,
    'rep 8 mul': 3,
    '8a mul': 4 ,
    '8b mul': 5,
    '8c mul': 6,
    '8d mul': 7
    }

class MulInsnEmitter(object):
    def __init__(self, asm, op_add, add_dst, add_opd1, add_opd2, sig, set_flags):
        self.asm       = asm
        self.op_add    = op_add
        self.add_dst   = add_dst
        self.add_opd1  = add_opd1
        self.add_opd2  = add_opd2
        self.sig       = sig
        self.set_flags = set_flags

    def assemble(self, op_mul,
            mul_dst  = REGISTERS['null'],
            mul_opd1 = REGISTERS['r0'],
            mul_opd2 = REGISTERS['r0'],
            rotate   = 0,
            pack = 'nop'
            ):

        mul_pack = _MUL_PACK[pack]

        add_a, add_b, mul_a, mul_b, raddr_a, raddr_b, immed, unpack, read_pm =\
                locate_read_operands(self.add_opd1, self.add_opd2, mul_opd1, mul_opd2)

        waddr_add, waddr_mul, write_swap, regA_pack, _ =\
                locate_write_operands(self.add_dst, mul_dst)

        if mul_pack and regA_pack:
            raise AssembleError('Multiple pack operationss')

        write_pm = (mul_pack != 0)
        pack = mul_pack or regA_pack

        if unpack and pack and read_pm != write_pm:
            raise AssembleError('Invalid combination of packing and unpacking')
        elif unpack and not pack:
            pm = read_pm
        elif pack and not unpack:
            pm = write_pm
        else:
            pm = 0

        if immed or rotate:
            if self.sig != 'no signal':
                raise AssembleError('Signal {} can not be used with ALU small immediate instruction'.format(sig))
            self.sig = 'alu small imm'
        sig_bits = _SIGNAL[self.sig]

        if rotate:
            if immed:
                raise AssembleError('Rotate operation can not be used with ALU small immediate instruction')
            if not (0 <= mul_a and mul_a < 3 and 0 <= mul_b and mul_b < 3):
                raise AssembleError('Rotate operation is only be available when both of mul ALU inputs are taken from r0-r3')
            if rotate == REGISTERS['r5']:
                raddr_b = 48
            else:
                if not (1 <= rotate and rotate <= 15):
                    raise AssembleError('Invalid rotation value {}'.format(rotate))
                raddr_b = 48 + rotate
        self.asm.emit(AluInsn(
            sig       = sig_bits,
            unpack    = unpack,
            pm        = pm,
            pack      = pack,
            sf        = self.set_flags,
            ws        = write_swap,
            cond_add  = 1,
            cond_mul  = 1,
            op_add    = self.op_add,
            op_mul    = op_mul,
            waddr_add = waddr_add,
            waddr_mul = waddr_mul,
            raddr_a   = raddr_a,
            raddr_b   = raddr_b,
            add_a     = add_a,
            add_b     = add_b,
            mul_a     = mul_a,
            mul_b     = mul_b
            ), increment=False)

for opcode, name in enumerate(MUL_INSTRUCTIONS):
    setattr(MulInsnEmitter, name, _partialmethod(MulInsnEmitter.assemble, opcode))

class Assembler(object):
    REGISTERS = REGISTERS

    def __init__(self):
        self.insns  = []    # instruction words
        self.pc     = 0     # program counter
        self.labels = {}
        self.backpatch_list = []

    def emit(self, insn, increment=True):
        if increment:
            self.insns.append(insn.to_bytes())
            self.pc += 8
        else:
            self.insns[-1] = insn.to_bytes()

    def get_insn(self, pc):
        return Insn.from_bytes(self.insns[pc/8])

    def set_insn(self, pc, insn):
        self.insns[pc/8] = insn.to_bytes()

    def backpatch(self):
        for insn_pc, label in self.backpatch_list:
            if label not in self.labels:
                raise AssembleError('Undefined label {}'.format(label))
            insn = self.get_insn(insn_pc)
            assert(isinstance(insn, BranchInsn))
            insn.immediate = self.labels[label] - (insn_pc + 4*8)
            self.set_insn(insn_pc, insn)
        self.backpatch_list = []

    def getcode(self):
        self.backpatch()
        return ''.join(self.insns)

    @syntax_sugar
    def ldi(self, *args, **kwargs):
        reg1 = args[0]
        if len(args) == 2:
            reg2 = REGISTERS['null']
            imm  = args[1]
        else:
            reg2 = args[1]
            imm  = args[2]

        if not (reg1.spec & _REG_AW and reg2.spec & _REG_BW):
            reg1, reg2 = reg2, reg1
            if not (reg1.spec & _REG_AW):
                raise AssembleError('{} is not a write register of regfile A'.format(reg1))
            if not (reg2.spec & _REG_BW):
                raise AssembleError('{} is not a write register of regfile B'.format(reg2))

        imm, unpack = pack_imm(imm)
        self.emit(LoadInsn(sig = 0xE, unpack = unpack, pm = 0, pack = 0, cond_add = 1,
            cond_mul = 1, sf = 0, ws = 0, waddr_add = reg1.addr, waddr_mul = reg2.addr,
            immediate = imm,))

    def add_insn(self, name, opcode, 
            dst  = REGISTERS['null'], # destination opdister
            opd1 = REGISTERS['r0'],   # operand 1
            opd2 = REGISTERS['r0'],   # operand 2
            sig  = 'no signal',
            set_flags = True          # if True Z,N,C flags will be set
            ):

        add_a, add_b, mul_a, mul_b, raddr_a, raddr_b, immed, unpack, read_pm = \
                locate_read_operands(opd1, opd2)

        if immed:
            if sig != 'no signal':
                raise AssembleError('Signal {} can not be used with ALU small immediate instruction'.format(sig))
            sig = 'alu small imm'
        sig_bits = _SIGNAL[sig]

        if name == 'nop':
            set_flags = False

        waddr_add, waddr_mul, write_swap, pack, write_pm = \
                locate_write_operands(dst, self.REGISTERS['null'])

        if unpack and pack and read_pm != write_pm:
            raise AssembleError('Invalid combination of packing and unpacking')
        elif unpack and not pack:
            pm = read_pm
        elif pack and not unpack:
            pm = write_pm
        else:
            pm = 0

        self.emit(AluInsn(sig = sig_bits, unpack = unpack, pm = pm, pack = pack,
            cond_add = 1, cond_mul = 1, sf = set_flags, ws = write_swap,
            op_add = opcode, waddr_add = waddr_add, waddr_mul = waddr_mul,
            raddr_a = raddr_a, raddr_b = raddr_b, add_a = add_a, add_b = add_b,
            mul_a = mul_a, mul_b = mul_b))

        return MulInsnEmitter(self, op_add = opcode, add_dst = dst,
            add_opd1 = opd1, add_opd2 = opd2, sig = sig, set_flags = set_flags)

    def mul_insn(self, name, *args, **kwargs):
        return getattr(self.nop(), name)(*args, **kwargs)

    def branch_insn(self, cond_br, target = 0, reg = None, link = REGISTERS['null']):
        if isinstance(target, basestring):
            self.backpatch_list.append((self.pc, target))
            imm = 0
            relative = True
        elif isinstance(target, int):
            imm = target
            relative = False
        else:
            raise AssembleError('Invalid branch target {}'.format(target))

        if reg:
            if not (reg.spec & _REG_AR):
                raise AssembleError('Must be regfile A register {}'.format(reg))
            raddr_a = reg.addr
            use_reg = True
        else:
            raddr_a = 0
            use_reg = False

        waddr_add, waddr_mul, write_swap, pack, pm = locate_write_operands(link)
        if pack or pm:
            raise AssembleError('Packing can not be used with branch instruction')

        self.emit(BranchInsn(
            sig = 0xF, cond_br = cond_br, rel = relative, reg = use_reg,
            raddr_a = raddr_a, ws = write_swap, waddr_add = waddr_add, waddr_mul = waddr_mul,
            immediate = imm
            ))

    @syntax_sugar
    def label(self, name):
        if name in self.labels:
            raise AssembleError('Duplicated labels {}'.format(name))
        self.labels[name] = self.pc

    def sema_insn(self, sa, sema_id):
        if not (0 <= sema_id and sema_id <= 15):
            raise AssembleError('Semaphore id must be in range (1..15)')

        self.emit(SemaInsn(
            sig = 0xE, unpack = 4, pm = 0, pack = 0, cond_add = 1, cond_mul = 1, sf = 0, ws = 0,
            waddr_add = 0, waddr_mul = 0, sa = sa, semaphore = sema_id))

    @syntax_sugar
    def sema_up(self, sema_id):
        self.sema_insn(1, sema_id)

    @syntax_sugar
    def sema_down(self, sema_id):
        self.sema_insn(0, sema_id)

    @syntax_sugar
    def mov(self, dst, src, **kwargs):
        return self.bor(dst, src, src, **kwargs)

    @syntax_sugar
    def read(self, src):
        return self.mov(self.REGISTERS['null'], src)

    @syntax_sugar
    def write(self, dst):
        return self.mov(dst, self.REGISTERS['null'])

    @syntax_sugar
    def setup_vpm_write(self, mode = '32bit horizontal', stride = 1, Y = 0, **kwargs):
        modes      = mode.split()
        size       = {'8bit': 0, '16bit': 1, '32bit': 2}[modes.pop(0)]
        laned      = {'packed': 0, 'laned': 1}[modes.pop(0)] if size != 2 else 0
        horizontal = {'vertical': 0, 'horizontal': 1}[modes.pop(0)]
        if horizontal:
            addr = Y << 2 | kwargs.get('B', 0) if size == 0 else \
                   Y << 1 | kwargs.get('H', 0) if size == 1 else \
                   Y
        else:
            X = kwargs.get('X', 0)
            addr = (Y & 0x30) << 6 | X << 2 | kwargs.get('B', 0) if size == 0 else \
                   (Y & 0x30) << 5 | X << 1 | kwargs.get('H', 0) if size == 1 else \
                   (Y & 0x30) << 4 | X
        self.ldi(self.REGISTERS['vpmvcd_wr_setup'],
                stride<<12|horizontal<<11|laned<<10|size<<8|addr)

    @syntax_sugar
    def setup_vpm_read(self, nrows, mode = '32bit horizontal', Y = 0, stride = 1, **kwargs):
        modes      = mode.split()
        size       = {'8bit': 0, '16bit': 1, '32bit': 2}[modes.pop(0)]
        laned      = {'packed': 0, 'laned': 1}[modes.pop(0)] if size != 2 else 0
        horizontal = {'vertical': 0, 'horizontal': 1}[modes.pop(0)]
        if horizontal:
            addr = Y << 2 | kwargs.get('B', 0) if size == 0 else \
                   Y << 1 | kwargs.get('H', 0) if size == 1 else \
                   Y
        else:
            X = kwargs['X']
            addr = (Y & 0x30) << 6 | X << 2 | kwargs.get('B', 0) if size == 0 else \
                   (Y & 0x30) << 5 | X << 1 | kwargs.get('H', 0) if size == 1 else \
                   (Y & 0x30) << 4 | X
        self.ldi(self.REGISTERS['vpmvcd_rd_setup'],
                nrows<<20|stride<<12|horizontal<<11|laned<<10|size<<8|addr)

    @syntax_sugar
    def setup_dma_store(self, nrows, mode = '32bit horizontal', Y = 0, X = 0, ncols = 16,
            offset = 0):
        modes = mode.split()
        modew = 0x4 | offset if modes[0] == '8bit' else \
                0x2 | offset if modes[0] == '16bit' else \
                0
        horizontal = { 'horizontal': 1, 'vertical': 0 }[modes[1]]
        addr = Y<<4|X
        self.ldi(self.REGISTERS['vpmvcd_wr_setup'],
                0x80000000|nrows<<23|ncols<<16|horizontal<<14|addr<<3|modew)

    @syntax_sugar
    def setup_dma_load(self, nrows, mode = '32bit horizontal', Y = 0, X = 0, ncols = 16,
            offset = 0, vpitch = 1, mpitch = 3):
        modes = mode.split()
        modew = 0x4 | offset if modes[0] == '8bit' else \
                0x2 | offset if modes[0] == '16bit' else \
                0
        vertical = { 'horizontal': 0, 'vertical': 1 }[modes[1]]
        addr = Y<<4|X
        self.ldi(self.REGISTERS['vpmvcd_rd_setup'],
                0x80000000|modew<<28|mpitch<<24|ncols<<20|nrows<<16|vpitch<<12|vertical<<11|addr)

    @syntax_sugar
    def start_dma_store(self, reg):
        return self.mov(self.REGISTERS['vpm_st_addr'], reg)

    @syntax_sugar
    def start_dma_load(self, reg):
        return self.mov(self.REGISTERS['vpm_ld_addr'], reg)

    @syntax_sugar
    def wait_dma_store(self):
        return self.bor(self.REGISTERS['null'], self.REGISTERS['vpm_st_wait'],
                        self.REGISTERS['vpm_st_wait'])

    @syntax_sugar
    def wait_dma_load(self):
        return self.read(self.REGISTERS['vpm_ld_wait'])

    @syntax_sugar
    def interrupt(self):
        return self.write(self.REGISTERS['host_interrupt'])

    @syntax_sugar
    def exit(self):
        self.interrupt()
        self.nop(sig = 'thread end')
        self.nop()
        self.nop()

for opcode, name in enumerate(ADD_INSTRUCTIONS):
    if name:
        setattr(Assembler, name, _partialmethod(Assembler.add_insn, name, opcode))
        INSTRUCTION_ALIASES.append(name)

for name in MUL_INSTRUCTIONS:
    if name not in ADD_INSTRUCTIONS:
        setattr(Assembler, name, _partialmethod(Assembler.mul_insn, name))
        INSTRUCTION_ALIASES.append(name)

for cond_br, name in enumerate(BRANCH_INSTRUCTIONS):
    if name:
        setattr(Assembler, name, _partialmethod(Assembler.branch_insn, cond_br))
        INSTRUCTION_ALIASES.append(name)

SETUP_ASM_LOCALS = ast.parse(
    '\n'.join(map('{0} = asm.{0}'.format, INSTRUCTION_ALIASES)) + '\n' + 
    '\n'.join(map('{0} = asm.REGISTERS[\'{0}\']'.format, REGISTERS))
    )

def qpucode(f):
    args, _, _, _ = inspect.getargspec(f)

    if 'asm' not in args:
        raise AssembleError('A function decorated with @qpucode must have a parameter named \'asm\'') 

    tree = ast.parse(inspect.getsource(f))

    fundef = tree.body[0]
    fundef.body = SETUP_ASM_LOCALS.body + fundef.body

    # Must remove @qpucode decorator to avoid inifinite recursion.
    fundef.decorator_list = filter(lambda d: d.id != 'qpucode', fundef.decorator_list)

    code = compile(tree, '<qpucode>', 'exec')
    scope = {}
    exec(code, f.__globals__, scope)
    return scope[f.__name__]

def assemble(f, *args, **kwargs):
    asm = Assembler()
    f(asm, *args, **kwargs)
    return asm.getcode()
