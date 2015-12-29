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
import numbers

import numpy


class _partialmethod(partial):
    'A descriptor for methods behaves like :py:class:`functools.partial.`'
    def __get__(self, obj, type):
        return partial(self.func, obj,
                       *(self.args or ()), **(self.keywords or {}))

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
                    f + '=' + hex(getattr(self, f))
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

#============================ Instruction emitter =============================

class Emitter(object):
    'Base class of instruction emitters.'

    def __init__(self, asm):
        self.asm = asm

    def _encode_write_operands(self,
            add_dst=REGISTERS['null'], mul_dst=REGISTERS['null']):
        """Encode waddr_add, waddr_mul, write_swap, pack and pm from given two
        destination registers.
        """

        assert(add_dst.unpack_bits == 0)
        assert(mul_dst.unpack_bits == 0)

        if add_dst.pack_bits != 0 and mul_dst.pack_bits != 0:
            raise AssembleError(
                'Conflict packing of two registers: {} {}'.format(
                    add_dst, mul_dst))

        pack_bits = add_dst.pack_bits or mul_dst.pack_bits

        if add_dst.spec & _REG_AW and mul_dst.spec & _REG_BW:
            return add_dst.addr, mul_dst.addr, False, pack_bits
        elif mul_dst.spec & _REG_AW and add_dst.spec & _REG_BW:
            return add_dst.addr, mul_dst.addr, True, pack_bits

        raise AssembleError(
            'Invalid combination of destination registers: {} {}'.format(
                add_dst, mul_dst))

    def _encode_read_operands(self,
            add_a=REGISTERS['r0'], add_b=REGISTERS['r0'],
            mul_a=REGISTERS['r0'], mul_b=REGISTERS['r0']):
        """Encode input muxes, raddr_a, raddr_b, unpack from given four source
        registers.
        """

        operands = [add_a, add_b, mul_a, mul_b]
        muxes = [None, None, None, None]
        unpack_bits = 0
        pm_bit = 0
        raddr_a = None
        raddr_b = None
        small_imm = None

        # Encode unpacking.
        for opd in operands:
            if not isinstance(opd, Register):
                continue
            if opd.unpack_bits:
                if unpack_bits == 0:
                    unpack_bits = opd.unpack_bits
                    pm_bit = opd.pm_bit
                elif (opd.unpack_bits != unpack_bits or opd.pm_bit != pm_bit):
                    raise AssembleError('Conflict of unpacking')

        # Assign input muxes for accumulators.
        for i, opd in enumerate(operands):
            if isinstance(opd, Register) and opd.name in ACCUMULATORS:
                muxes[i] = _INPUT_MUXES[opd.name]

        if all(m is not None for m in muxes):
            null_addr = REGISTERS['null'].addr
            return [muxes, null_addr, null_addr, False, unpack_bits, pm_bit]

        # Locate operands which have to be regfile B register.
        for i, opd in enumerate(operands):
            if muxes[i] is not None or not isinstance(opd, Register):
                continue
            if opd.spec & _REG_BR and not (opd.spec & _REG_AR):
                if raddr_b is None:
                    raddr_b = opd.addr
                    muxes[i] = _INPUT_MUXES['B']
                elif raddr_b == opd.addr:
                    muxes[i] = _INPUT_MUXES['B']
                else:
                    raise AssembleError('Too many regfile B source operand')

        # Locate small immediates.
        for i, opd in enumerate(operands):
            if muxes[i] is not None or isinstance(opd, Register):
                continue

            imm = _SMALL_IMM[repr(opd)]
            if small_imm is None:
                small_imm = imm
                muxes[i] = _INPUT_MUXES['B']
            elif small_imm == imm:
                muxes[i] = _INPUT_MUXES['B']
            else:
                raise AssembleError('Too many immediates')

        # Check of raddr_b conflict.
        if small_imm is not None and raddr_b is not None:
            raise AssembleError(
                'Conflict of regfile B source operand and immedaite value'
                )
        if small_imm is not None:
            raddr_b = small_imm

        # Locate operands which have to be regfile A register.
        for i, opd in enumerate(operands):
            if muxes[i] is not None:
                continue
            if opd.spec & _REG_AR and not (opd.spec & _REG_BR):
                if raddr_a is None:
                    raddr_a = opd.addr
                    muxes[i] = _INPUT_MUXES['A']
                elif raddr_a == opd.addr:
                    muxes[i] = _INPUT_MUXES['A']
                else:
                    raise AssembleError('Too many regfile A source operand')

        # Locate remaining operands.
        for i, opd in enumerate(operands):
            if muxes[i] is not None: continue

            if not (opd.spec & (_REG_AR | _REG_BR)):
                raise AssembleError('{} can not be a read operand'.format(opd))

            if raddr_a is None:
                raddr_a = opd.addr
                muxes[i] = _INPUT_MUXES['A']
            elif small_imm is None and raddr_b is None:
                raddr_b = opd.addr
                muxes[i] = _INPUT_MUXES['B']
            else:
                raise AssembleError('Failed to locate operand {}'.format(opd))

        if raddr_a is None:
            raddr_a = REGISTERS['null'].addr
        if raddr_b is None:
            raddr_b = REGISTERS['null'].addr

        use_small_imm = (small_imm is not None)
        return [muxes, raddr_a, raddr_b, use_small_imm, unpack_bits, pm_bit]

class AddEmitter(Emitter):
    'Emitter of Add ALU instructions.'

    def _emit(self, op_add, dst=REGISTERS['null'], opd1=REGISTERS['r0'],
            opd2=REGISTERS['r0'], sig='no signal', set_flags=True, **kwargs):

        muxes, raddr_a, raddr_b, use_imm, unpack, read_pm = \
                self._encode_read_operands(opd1, opd2)

        if use_imm:
            if sig != 'no signal' and sig != 'alu small imm':
                raise AssembleError(
                        '\'{}\' can not be used with immediate'.format(sig))
            sig = 'alu small imm'
        sig_bits = _SIGNAL[sig]

        if op_add == _ADD_INSN['nop']:
            set_flags = False

        waddr_add, waddr_mul, write_swap, pack = \
                self._encode_write_operands(dst)

        pm = 0
        if unpack and pack:
            if read_pm != 0:
                raise AssembleError('Conflict of packing and unpacking')
            pm = read_pm
        elif unpack and not pack:
            pm = read_pm
        elif pack and not unpack:
            pm = 0

        cond_add = _COND[kwargs.get('cond', 'always')]
        cond_mul = _COND['never']

        insn = AluInsn(
                sig=sig_bits, unpack=unpack, pm=pm, pack=pack,
                sf=set_flags, ws=write_swap, cond_add=cond_add,
                cond_mul=cond_mul, op_add=op_add, op_mul=_MUL_INSN['nop'],
                waddr_add=waddr_add, waddr_mul=waddr_mul, raddr_a=raddr_a,
                raddr_b=raddr_b, add_a=muxes[0], add_b=muxes[1],
                mul_a=muxes[2], mul_b=muxes[3]
                )

        self.asm._emit(insn)

        # Create MulEmitter which holds arguments of Add ALU for dual
        # issuing.
        return MulEmitter(
                self.asm, op_add=op_add, add_dst=dst, add_opd1=opd1,
                add_opd2=opd2, cond_add=cond_add, sig=sig, set_flags=set_flags,
                increment=False)

class MulEmitter(Emitter):
    """ Emitter of Mul ALU instructions.

    This object receives arguments for Add ALU instruction at its construction
    to implement dual-issue of Add and Mul ALU.
    """
    def __init__(self, asm, op_add=_ADD_INSN['nop'], add_dst=REGISTERS['null'],
                 add_opd1=REGISTERS['r0'], add_opd2=REGISTERS['r0'],
                 cond_add=_COND['never'], sig='no signal', set_flags=False,
                increment=True):
        self.asm = asm
        self.op_add = op_add
        self.add_dst = add_dst
        self.add_opd1 = add_opd1
        self.add_opd2 = add_opd2
        self.cond_add = cond_add
        self.sig = sig
        self.set_flags = set_flags
        self.increment = increment

    def _emit(self, op_mul, mul_dst=REGISTERS['null'], mul_opd1=REGISTERS['r0'],
            mul_opd2=REGISTERS['r0'], rotate=0, pack='nop', **kwargs):

        mul_pack = _MUL_PACK[pack]

        muxes, raddr_a, raddr_b, use_imm, unpack, read_pm = \
                self._encode_read_operands(self.add_opd1, self.add_opd2,
                                           mul_opd1, mul_opd2)

        waddr_add, waddr_mul, write_swap, regA_pack = \
                self._encode_write_operands(self.add_dst, mul_dst)

        if mul_pack and regA_pack:
            raise AssembleError('Multiple pack operationss')

        write_pm = (mul_pack != 0)
        pack = mul_pack or regA_pack

        pm = 0
        if unpack and pack:
            if read_pm != write_pm:
                raise AssembleError('Conflict of packing and unpacking')
            pm = read_pm
        elif unpack and not pack:
            pm = read_pm
        elif pack and not unpack:
            pm = write_pm

        sig = kwargs.get('sig', 'no signal')
        if self.sig != 'no signal':
            if sig != 'no signal':
                raise AssembleError('Conflict of signals')
            sig = self.sig

        if use_imm or rotate:
            if sig != 'no signal' and sig != 'alu small imm':
                raise AssembleError(
                        '\'{}\' can not be used with immediate'.format(sig))
            sig = 'alu small imm'
        sig_bits = _SIGNAL[sig]

        if rotate:
            if not (0<=muxes[2] and muxes[2]<3 and 0<=muxes[3] and muxes[3]<3):
                raise AssembleError('Rotate operation is only available when'
                                    ' inputs are taken from r0-r3')

            if rotate == REGISTERS['r5']:
                raddr_b = 48
            else:
                raddr_b = 48 + rotate%16

        cond_add = self.cond_add
        cond_mul = _COND[kwargs.get('cond', 'always')]

        insn = AluInsn(
                sig=sig_bits, unpack=unpack, pm=pm, pack=pack,
                sf=self.set_flags, ws=write_swap, cond_add=cond_add,
                cond_mul=cond_mul, op_add=self.op_add, op_mul=op_mul,
                waddr_add=waddr_add, waddr_mul=waddr_mul, raddr_a=raddr_a,
                raddr_b=raddr_b, add_a=muxes[0], add_b=muxes[1],
                mul_a=muxes[2], mul_b=muxes[3]
                )
        self.asm._emit(insn, increment=self.increment)

class LoadEmitter(Emitter):
    'Emitter of load instructions.'

    def _encode_imm(self, val):
        if isinstance(val, float):
            return unpack('L', pack('f', val))[0], 0
        elif isinstance(val, numbers.Number):
            fmt = 'l' if val < 0 else 'L'
            return unpack('L', pack(fmt, val))[0], 0
        elif isinstance(val, (list, tuple, numpy.ndarray)):
            return self._encode_per_element_imm(list(val))
        raise AssembleError('Unsupported immediate value {}'.format(val))

    def _encode_per_element_imm(self, values):
        if len(values) > 16:
            raise AssembleError('Too many immediate values {}'.format(values))

        values.extend([0] * (16-len(values)))
        unsigned = all(map(lambda x: x >= 0, values))
        high = 0
        low  = 0
        for i in reversed(range(16)):
            high <<= 1
            low  <<= 1
            v = values[i]

            if (not unsigned and (v >= 2 or v < -2)) or (unsigned and v >= 4):
                raise AssembleError('{} is not a 2-bit {}signed value'.format(
                        v, ['', 'un'][unsigned]))
            high |= (v & 0x2) >> 1
            low  |= v & 0x1

        return (high << 16) | low, 2*unsigned + 1

    def _emit(self, *args, **kwargs):
        """Load immediate.

        Store ``value`` to the register ``a``.
        >>> ldi(a, value)

        You can use two destination registers.  ``value`` will be stored to
        both register ``a`` and ``b``.
        >>> ldi(a, b, value)

        Available immediate values:

        * signed and unsigned integers.
        * floating point numbers.
        * List of 2-bit signed and unsigned integers. Its maximum length is 16.

        The third behaves exceptionally. Values of the list will be stored to
        each SIMD element one by one. When the length of the list is shorter
        than 16, 0s will are stored for remaining elements.
        """

        reg1 = args[0]
        if len(args) == 2:
            reg2 = REGISTERS['null']
            imm = args[1]
        else:
            reg2 = args[1]
            imm = args[2]

        waddr_add, waddr_mul, write_swap, pack = \
                self._encode_write_operands(reg1, reg2)

        imm, unpack = self._encode_imm(imm)

        cond_add = cond_mul = _COND[kwargs.get('cond', 'always')]
        set_flags = kwargs.get('set_flags', False)

        insn = LoadInsn(
                sig=0xe, unpack=unpack, pm=0, pack=pack, cond_add=cond_add,
                cond_mul=cond_mul, sf=set_flags, ws=write_swap,
                waddr_add=waddr_add, waddr_mul=waddr_mul, immediate=imm
                )

        self.asm._emit(insn)

class Label(object):
    def __init__(self, asm, name):
        self.name = name
        self.pinned = True

class LabelEmitter(Emitter):
    'Emitter to provide L.<label name> syntax.'

    def __getitem__(self, name):
        label = Label(self.asm, name)
        self.asm._add_label(label)
        return label

    def __getattr__(self, name):
        return self[name]

class BranchEmitter(Emitter):
    'Emitter of branch instructions.'

    def _emit(self, cond_br, target=None, reg=None, absolute=False,
              link=REGISTERS['null']):

        if target is None:
            imm = 0
        elif isinstance(target, Label):
            target.pinned = False
            self.asm._add_backpatch_item(target.name)
            imm = 0
        elif isinstance(target, int):
            imm = target
        else:
            raise AssembleError('Invalid branch target: {}'.format(target))

        if reg:
            if (not (reg.spec & _REG_AR) or
                reg.name not in GENERAL_PURPOSE_REGISTERS):
                raise AssembleError(
                    'Must be general purpose regfile A register {}'.format(reg)
                    )
            assert(reg.addr < 32)
            raddr_a = reg.addr
            use_reg = True
        else:
            raddr_a = 0
            use_reg = False

        waddr_add, waddr_mul, write_swap, pack = \
                self._encode_write_operands(link)

        if pack:
            raise AssembleError('Packing is not available for link register')

        insn = BranchInsn(
            sig=0xF, cond_br=cond_br, rel=not absolute, reg=use_reg,
            raddr_a=raddr_a, ws=write_swap, waddr_add=waddr_add,
            waddr_mul=waddr_mul, immediate=imm
            )

        self.asm._emit(insn)

class SemaEmitter(Emitter):
    'Emitter of semaphore instructions.'

    def _emit(self, sa, sema_id):
        if not (0 <= sema_id and sema_id <= 15):
            raise AssembleError('Semaphore id must be in range (0..15)')

        null_addr = REGISTERS['null'].addr
        insn = SemaInsn(
            sig=0xE, unpack=4, pm=0, pack=0, cond_add=1, cond_mul=1, sf=0,
            ws=0, waddr_add=null_addr, waddr_mul=null_addr, sa=sa,
            semaphore=sema_id)

        self.asm._emit(insn)


#================================= Assembler ==================================


class Assembler(object):
    'QPU Assembler.'

    _REGISTERS = REGISTERS

    def __init__(self):
        self._instructions = []
        self._program_counter = 0
        self._labels = []
        self._backpatch_list = []    # list of (instruction index, label)

        self._add = AddEmitter(self)
        self._mul = MulEmitter(self)
        self._load = LoadEmitter(self)
        self._branch = BranchEmitter(self)
        self._sema = SemaEmitter(self)
        self.L = LabelEmitter(self)

    def _emit(self, insn, increment=True):
        """Emit new instruction ``insn`` if increment is True else replace the
        last instruction with ``insn``.
        """

        if increment:
            self._instructions.append(insn)
            self._program_counter += 8
        else:
            self._instructions[-1] = insn

    def _emit_add(self, *args, **kwargs):
        return self._add._emit(*args, **kwargs)

    def _emit_mul(self, *args, **kwargs):
        return self._mul._emit(*args, **kwargs)

    def _emit_load(self, *args, **kwargs):
        return self._load._emit(*args, **kwargs)

    def _emit_branch(self, *args, **kwargs):
        return self._branch._emit(*args, **kwargs)

    def _emit_sema(self, *args, **kwargs):
        return self._sema._emit(*args, **kwargs)

    def _add_label(self, label):
        self._labels.append((label, self._program_counter))

    def _fix_labels(self):
        new_labels = []
        label_dict = {}
        for label, pc in self._labels:
            if not label.pinned:
                continue
            if label.name in label_dict:
                raise AssembleError('Duplicated label: {}'.format(label.name))
            label_dict[label.name] = pc
            new_labels.append((label, pc))
        self._labels = new_labels
        return label_dict

    def _backpatch(self):
        'Backpatch immediates of branch _instructions'

        labels = self._fix_labels()
        for i, label in self._backpatch_list:
            if label not in labels:
                raise AssembleError('Undefined label {}'.format(label))

            insn = self._instructions[i]
            assert(isinstance(insn, BranchInsn))
            assert(insn.rel)

            insn.immediate = labels[label] - 8*(i + 4)
        self._backpatch_list = []

    def _add_backpatch_item(self, target):
        self._backpatch_list.append((len(self._instructions), target))

    def _get_code(self):
        'Convert list of _instructions to executable bytes.'

        self._backpatch()
        return b''.join(insn.to_bytes() for insn in self._instructions)

#=================================== Alias ====================================

def alias(f):
    setattr(Assembler, f.__name__, f)

for name, code in _ADD_INSN.items():
    setattr(Assembler, name, _partialmethod(Assembler._emit_add, code))

for name, code in _MUL_INSN.items():
    if name not in _ADD_INSN:
        setattr(Assembler, name, _partialmethod(Assembler._emit_mul, code))
    setattr(MulEmitter, name, _partialmethod(MulEmitter._emit, code))

for name, code in _BRANCH_INSN.items():
    setattr(Assembler, name, _partialmethod(Assembler._emit_branch, code))

Assembler.ldi = Assembler._emit_load

@alias
def rotate(asm, dst, src, shift, **kwargs):
    return asm.v8min(dst, src, src, rotate=shift, **kwargs)

def mul_rotate(self, dst, src, shift, **kwargs):
    return self.v8min(dst, src, src, rotate=shift, **kwargs)

MulEmitter.rotate = mul_rotate

@alias
def mov(asm, dst, src, **kwargs):
    set_flags = kwargs.pop('set_flags', False)
    return asm.bor(dst, src, src, set_flags=set_flags, **kwargs)

def mul_mov(self, dst, src, **kwargs):
    return self.v8min(dst, src, src, **kwargs)

MulEmitter.mov = mul_mov

@alias
def mutex_acquire(asm):
    return asm.mov(REGISTERS['null'], REGISTERS['mutex'])

@alias
def mutex_release(asm):
    return asm.mov(REGISTERS['mutex'], REGISTERS['null'])

@alias
def setup_vpm_read(asm, nrows, mode='32bit horizontal', Y=0, stride=1,
                   **kwargs):
    modes = mode.split()
    size = {'8bit': 0, '16bit': 1, '32bit': 2}[modes.pop(0)]
    laned = {'packed': 0, 'laned': 1}[modes.pop(0)] if size != 2 else 0
    horizontal = {'vertical': 0, 'horizontal': 1}[modes.pop(0)]
    if horizontal:
        addr = (
            Y << 2 | kwargs.get('B', 0) if size == 0 else
            Y << 1 | kwargs.get('H', 0) if size == 1 else
            Y
            )
    else:
        X = kwargs['X']
        addr = (
            (Y & 0x30) << 2 | X << 2 | kwargs.get('B', 0) if size == 0 else
            (Y & 0x30) << 1 | X << 1 | kwargs.get('H', 0) if size == 1 else
            (Y & 0x30) | X
            )

    asm.ldi(REGISTERS['vpmvcd_rd_setup'],
            (nrows&0xf)<<20|stride<<12|horizontal<<11|laned<<10|size<<8|addr)

@alias
def setup_vpm_write(asm, mode='32bit horizontal', stride=1, Y=0, **kwargs):
    modes = mode.split()
    size = {'8bit': 0, '16bit': 1, '32bit': 2}[modes.pop(0)]
    laned = {'packed': 0, 'laned': 1}[modes.pop(0)] if size != 2 else 0
    horizontal = {'vertical': 0, 'horizontal': 1}[modes.pop(0)]
    if horizontal:
        addr = (
            Y << 2 | kwargs.get('B', 0) if size == 0 else
            Y << 1 | kwargs.get('H', 0) if size == 1 else
            Y
            )
    else:
        X = kwargs.get('X', 0)
        addr = (
            (Y & 0x30) << 2 | X << 2 | kwargs.get('B', 0) if size == 0 else
            (Y & 0x30) << 1 | X << 1 | kwargs.get('H', 0) if size == 1 else
            (Y & 0x30) | X
            )

    asm.ldi(REGISTERS['vpmvcd_wr_setup'],
            stride<<12|horizontal<<11|laned<<10|size<<8|addr)

@alias
def setup_dma_load_stride(asm, val, tmp_reg=REGISTERS['r0']):
    if not isinstance(val, int) and val.name == tmp_reg.name:
        raise AssembleError('setup_dma_store_stride uses \'{}\' internally'
                            .format(tmp_reg.name))
    if isinstance(val, int):
        return asm.ldi(REGISTERS['vpmvcd_rd_setup'], (9<<28)|val)
    else:
        asm.ldi(tmp_reg, 9<<28)
        return asm.bor(REGISTERS['vpmvcd_rd_setup'], tmp_reg, val)

@alias
def setup_dma_load(asm, nrows=1, ncols=16, mode='32bit horizontal', Y=0, X=0,
                   offset=0, vpitch=1, mpitch=3):
    modes = mode.split()
    modew = (
        0x4 | offset if modes[0] == '8bit' else
        0x2 | offset if modes[0] == '16bit' else
        0
        )
    vertical = { 'horizontal': 0, 'vertical': 1 }[modes[1]]
    asm.ldi(REGISTERS['vpmvcd_rd_setup'],
            0x80000000|modew<<28|mpitch<<24|(ncols&0xf)<<20|(nrows&0xf)<<16|
            vpitch<<12|vertical<<11|Y<<4|X)

@alias
def start_dma_load(asm, reg, rot=0):
    if rot == 0:
        return asm.mov(REGISTERS['vpm_ld_addr'], reg)
    else:
        return asm.rotate(REGISTERS['vpm_ld_addr'], reg, rot)

@alias
def wait_dma_load(asm):
    return asm.mov(REGISTERS['null'], REGISTERS['vpm_ld_wait'])

@alias
def setup_dma_store(asm, nrows=1, ncols=16, mode='32bit horizontal', Y=0, X=0,
                    offset=0):
    modes = mode.split()
    modew = (
        0x4 | offset if modes[0] == '8bit' else
        0x2 | offset if modes[0] == '16bit' else
        0
        )
    horizontal = { 'horizontal': 1, 'vertical': 0 }[modes[1]]
    asm.ldi(REGISTERS['vpmvcd_wr_setup'],
            0x80000000|(nrows&0x7f)<<23|(ncols&0x7f)<<16|horizontal<<14|Y<<7|
            X<<3|modew)

@alias
def setup_dma_store_stride(asm, val, blockmode=False, tmp_reg=REGISTERS['r0']):
    if not isinstance(val, int) and val.name == tmp_reg.name:
        raise AssembleError('setup_dma_store_stride uses \'{}\' internally'
                            .format(tmp_reg.name))
    if isinstance(val, int):
        asm.ldi(REGISTERS['vpmvcd_wr_setup'], (3<<30)|(blockmode<<16)|val)
    else:
        asm.ldi(tmp_reg, (3<<30)|(blockmode<<16))
        asm.bor(REGISTERS['vpmvcd_wr_setup'], tmp_reg, val)

@alias
def start_dma_store(asm, reg, rot=0):
    if rot == 0:
        return asm.mov(REGISTERS['vpm_st_addr'], reg)
    else:
        return asm.rotate(REGISTERS['vpm_st_addr'], reg, rot)

@alias
def wait_dma_store(asm):
    return asm.mov(REGISTERS['null'], REGISTERS['vpm_st_wait'])

@alias
def interrupt(asm):
    return asm.mov(REGISTERS['host_interrupt'], 1)

@alias
def exit(asm, interrupt=True):
    if interrupt:
        asm.interrupt()

    # The instruction signaling program end must not attempt to write to either
    # of the physical A or B register files. (Reference guide Page 22)
    asm.mov(
        REGISTERS['r0'], REGISTERS['r0'], sig='thread end'
        ).mov(
        REGISTERS['r0'], REGISTERS['r0']
        )

    asm.nop()
    asm.nop()

@alias
def sema_up(asm, sema_id):
    asm._emit_sema(0, sema_id)

@alias
def sema_down(asm, sema_id):
    asm._emit_sema(1, sema_id)

REGISTER_ALIASES = '\n'.join(
    '{0} = asm._REGISTERS[\'{0}\']'.format(reg)
    for reg in Assembler._REGISTERS
    )

INSTRUCTION_ALIASES = '\n'.join(
    '{0} = asm.{0}'.format(f)
    for f in dir(Assembler)
    if f[0] != '_'
    )

SETUP_ASM_ALIASES = ast.parse("""
# Alias of registers.
{register_aliases}

# Alias of instructions.
{instruction_aliases}

# Label
L = asm.L
""".format(
        register_aliases=REGISTER_ALIASES,
        instruction_aliases=INSTRUCTION_ALIASES
        ))

def qpu(f):
    """Decorator for writing QPU assembly language.

    To write a QPU assembly program, decorate a function which has a parameter
    ``asm`` as the first argument with @qpu like this::

        @qpu
        def code(asm):
            mov(r0, uniform)
            iadd(r0, r0, 1)

            ...

            exit()

    This code is equivalent to::

        def code(asm):
            asm.mov(asm.r0, asm.uniform)
            asm.iadd(asm.r0, asm.r0, 1)

            ...

            asm.exit()
    """
    args, _, _, _ = inspect.getargspec(f)

    if 'asm' not in args:
        raise AssembleError('Argument named \'asm\' is necessary')

    tree = ast.parse(inspect.getsource(f))

    fundef = tree.body[0]
    fundef.body = SETUP_ASM_ALIASES.body + fundef.body
    fundef.decorator_list = []

    code = compile(tree, '<qpu>', 'exec')
    scope = {}
    exec(code, f.__globals__, scope)
    return scope[f.__name__]

def assemble(f, *args, **kwargs):
    'Assemble QPU program to byte string.'
    asm = Assembler()
    f(asm, *args, **kwargs)
    return asm._get_code()
