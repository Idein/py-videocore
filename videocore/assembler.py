"""VideoCore IV QPU assembler.

This module implements an assembly language for VideoCore IV QPU and its
assembler. We took an approach that the language is implemented as an internal
DSL of Python language.

The QPU instruction set is described in the section 3 of the following document
`VideoCore(R) IV 3D Architecture Reference Guide
<ttps://docs.broadcom.com/docs/12358545>`__
"""

from __future__ import print_function
import sys
from functools import partial, wraps
from struct import pack, unpack
import inspect
import ast
import numbers
import pickle

import numpy
from videocore.vinstr import AddInstr, MulInstr, LoadImmInstr, BranchInstr, SemaInstr, ComposedInstr
from videocore.checker import check_main
import videocore.encoding as enc
from videocore.encoding import REGISTERS, Register, AssembleError

class _partialmethod(partial):
    'A descriptor for methods behaves like :py:class:`functools.partial.`'
    def __get__(self, obj, type):
        return partial(self.func, obj,
                       *(self.args or ()), **(self.keywords or {}))

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

        if add_dst.spec & enc._REG_AW and mul_dst.spec & enc._REG_BW:
            return add_dst.addr, mul_dst.addr, False, pack_bits
        elif mul_dst.spec & enc._REG_AW and add_dst.spec & enc._REG_BW:
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
            if isinstance(opd, Register) and opd.name in enc.ACCUMULATORS:
                muxes[i] = enc._INPUT_MUXES[opd.name]

        if all(m is not None for m in muxes):
            null_addr = REGISTERS['null'].addr
            return [muxes, null_addr, null_addr, False, unpack_bits, pm_bit]

        # Locate operands which have to be regfile B register.
        for i, opd in enumerate(operands):
            if muxes[i] is not None or not isinstance(opd, Register):
                continue
            if opd.spec & enc._REG_BR and not (opd.spec & enc._REG_AR):
                if raddr_b is None:
                    raddr_b = opd.addr
                    muxes[i] = enc._INPUT_MUXES['B']
                elif raddr_b == opd.addr:
                    muxes[i] = enc._INPUT_MUXES['B']
                else:
                    raise AssembleError('Too many regfile B source operand')

        # Locate small immediates.
        for i, opd in enumerate(operands):
            if muxes[i] is not None or isinstance(opd, Register):
                continue

            imm = enc._SMALL_IMM.get(repr(opd))
            if imm is None:
                raise AssembleError('Unsupported small immediate value: {}'.format(opd))
            if small_imm is None:
                small_imm = imm
                muxes[i] = enc._INPUT_MUXES['B']
            elif small_imm == imm:
                muxes[i] = enc._INPUT_MUXES['B']
            else:
                raise AssembleError('Too many immediates')

        # Check of raddr_b conflict.
        if small_imm is not None and raddr_b is not None:
            raise AssembleError(
                'Conflict of regfile B source operand and immediate value'
                )
        if small_imm is not None:
            raddr_b = small_imm

        # Locate operands which have to be regfile A register.
        for i, opd in enumerate(operands):
            if muxes[i] is not None:
                continue
            if opd.spec & enc._REG_AR and not (opd.spec & enc._REG_BR):
                if raddr_a is None:
                    raddr_a = opd.addr
                    muxes[i] = enc._INPUT_MUXES['A']
                elif raddr_a == opd.addr:
                    muxes[i] = enc._INPUT_MUXES['A']
                else:
                    raise AssembleError('Too many regfile A source operand')

        # Locate remaining operands.
        for i, opd in enumerate(operands):
            if muxes[i] is not None: continue

            if not (opd.spec & (enc._REG_AR | enc._REG_BR)):
                raise AssembleError('{} can not be a read operand'.format(opd))

            if raddr_a is None or raddr_a == opd.addr:
                raddr_a = opd.addr
                muxes[i] = enc._INPUT_MUXES['A']
            elif (small_imm is None and raddr_b is None) or raddr_b == opd.addr:
                raddr_b = opd.addr
                muxes[i] = enc._INPUT_MUXES['B']
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

    def _emit(self, op_add, *args, **kwargs):
        l = enc._ADD_DEFAULT_ARGS.get(op_add, [])
        args = list(args) + l
        return self._emit_with_defaults(op_add, *args, **kwargs)

    def _emit_with_defaults(self, op_add, dst, opd1, opd2, sig='no signal', set_flags=True, **kwargs):

        muxes, raddr_a, raddr_b, use_imm, unpack, read_pm = \
                self._encode_read_operands(opd1, opd2)

        if use_imm:
            if sig != 'no signal' and sig != 'alu small imm':
                raise AssembleError(
                        '\'{}\' can not be used with immediate'.format(sig))
            sig = 'alu small imm'
        sig_bits = enc._SIGNAL[sig]

        if op_add == enc._ADD_INSN['nop']:
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

        cond_add_str = kwargs.get('cond', 'always')
        cond_add = enc._COND[cond_add_str]
        cond_mul = enc._COND['never']

        insn = enc.AluInsn(
                sig=sig_bits, unpack=unpack, pm=pm, pack=pack,
                sf=set_flags, ws=write_swap, cond_add=cond_add,
                cond_mul=cond_mul, op_add=op_add, op_mul=enc._MUL_INSN['nop'],
                waddr_add=waddr_add, waddr_mul=waddr_mul, raddr_a=raddr_a,
                raddr_b=raddr_b, add_a=muxes[0], add_b=muxes[1],
                mul_a=muxes[2], mul_b=muxes[3]
                )

        if self.asm.sanity_check:
            insn.verbose = AddInstr(enc._ADD_INSN_REV[op_add], dst, opd1, opd2, sig, set_flags, cond_add_str)
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
    def __init__(self, asm, op_add=enc._ADD_INSN['nop'], add_dst=REGISTERS['null'],
                 add_opd1=REGISTERS['r0'], add_opd2=REGISTERS['r0'],
                 cond_add=enc._COND['never'], sig='no signal', set_flags=False,
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

    def _emit(self, op_mul, *args, **kwargs):
        l = enc._MUL_DEFAULT_ARGS.get(op_mul, [])
        args = list(args) + l
        return self._emit_with_defaults(op_mul, *args, **kwargs)

    def _emit_with_defaults(self, op_mul, mul_dst, mul_opd1, mul_opd2, rotate=0, pack='nop', **kwargs):

        mul_pack = enc._MUL_PACK[pack]

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
        sig_bits = enc._SIGNAL[sig]

        if rotate:
            if muxes[2] == 5 or muxes[2] == 7 or muxes[3] == 5 or muxes[3] == 7:
                raise AssembleError('Rotate operation is only available when'
                                    ' inputs are taken from r0-r4 or ra')

            if raddr_b != REGISTERS['null'].addr:

                # 'r5 rotate' represents -1.
                # 'n-upward rotate' represents n-16.
                # So these combinations can be used (1 <= n <= 15):
                # +-----------+--------+
                # | small imm | rotate |
                # +-----------+--------+
                # |    -16    |   r5   |
                # |     -n    |   -n   |
                # +-----------+--------+
                # c.f. https://vc4-notes.tumblr.com/post/153467713064/

                if rotate == REGISTERS['r5']:
                    if raddr_b != enc._SMALL_IMM['-16']:
                        if use_imm:
                            raise AssembleError('Conflict immediate value and r5 rotate')
                        else:
                            raise AssembleError('Conflict of regfile B source operand and rotate')
                elif raddr_b != enc._SMALL_IMM[str(rotate%16-16)]:
                    if use_imm:
                        raise AssembleError('Conflict immediate value and n rotate')
                    else:
                        raise AssembleError('Conflict of regfile B source operand and rotate')

            if rotate == REGISTERS['r5']:
                raddr_b = 48
            else:
                raddr_b = 48 + rotate%16

        cond_add = self.cond_add
        cond_mul_str = kwargs.get('cond', 'always')
        cond_mul = enc._COND[cond_mul_str]

        insn = enc.AluInsn(
                sig=sig_bits, unpack=unpack, pm=pm, pack=pack,
                sf=self.set_flags, ws=write_swap, cond_add=cond_add,
                cond_mul=cond_mul, op_add=self.op_add, op_mul=op_mul,
                waddr_add=waddr_add, waddr_mul=waddr_mul, raddr_a=raddr_a,
                raddr_b=raddr_b, add_a=muxes[0], add_b=muxes[1],
                mul_a=muxes[2], mul_b=muxes[3]
                )
        if self.asm.sanity_check:
            insn.verbose = MulInstr(enc._MUL_INSN_REV[op_mul], mul_dst, mul_opd1, mul_opd2, sig, self.set_flags, cond_mul_str, rotate)
        self.asm._emit(insn, increment=self.increment)

class LoadEmitter(Emitter):
    'Emitter of load instructions.'

    def _encode_imm(self, val):
        if isinstance(val, float):
            return unpack('<L', pack('f', val))[0], 0
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

        sig = kwargs.get('sig', 'load')
        if sig != 'load':
            raise AssembleError('Conflict of signals')

        waddr_add, waddr_mul, write_swap, pack = \
                self._encode_write_operands(reg1, reg2)

        imm, unpack = self._encode_imm(imm)

        cond_add = cond_mul = enc._COND[kwargs.get('cond', 'always')]
        set_flags = kwargs.get('set_flags', False)

        insn = enc.LoadInsn(
                sig=0xe, unpack=unpack, pm=0, pack=pack, cond_add=cond_add,
                cond_mul=cond_mul, sf=set_flags, ws=write_swap,
                waddr_add=waddr_add, waddr_mul=waddr_mul, immediate=imm
                )
        if self.asm.sanity_check:
            insn.verbose = LoadImmInstr(reg1, reg2, imm)
        self.asm._emit(insn)

class Label(object):
    def __init__(self, asm, name):
        self.name = name
        self.pinned = True

class LabelEmitter(Emitter):
    'Emitter to provide L.<label name> syntax.'

    def __getitem__(self, name):
        label = Label(self.asm, self.asm._generate_label_name(name))
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
            if (not (reg.spec & enc._REG_AR) or
                reg.name not in enc.GENERAL_PURPOSE_REGISTERS):
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

        insn = enc.BranchInsn(
            sig=0xF, cond_br=cond_br, rel=not absolute, reg=use_reg,
            raddr_a=raddr_a, ws=write_swap, waddr_add=waddr_add,
            waddr_mul=waddr_mul, immediate=imm
            )
        if self.asm.sanity_check:
            insn.verbose = BranchInstr(enc._BRANCH_INSN_REV[cond_br], target, reg, absolute, link)
        self.asm._emit(insn)

class SemaEmitter(Emitter):
    'Emitter of semaphore instructions.'

    def _emit(self, sa, sema_id):
        if not (0 <= sema_id and sema_id <= 15):
            raise AssembleError('Semaphore id must be in range (0..15)')

        null_addr = REGISTERS['null'].addr
        insn = enc.SemaInsn(
            sig=0xE, unpack=4, pm=0, pack=0, cond_add=1, cond_mul=1, sf=0,
            ws=0, waddr_add=null_addr, waddr_mul=null_addr, sa=sa,
            semaphore=sema_id)

        if self.asm.sanity_check:
            insn.verbose = SemaInstr(sa, sema_id)
        self.asm._emit(insn)

class RawEmitter(Emitter):
    'Emitter of raw instructions. (DANGER)'

    def _emit(self, val1, val2):
        insn = enc.RawInsn(raw1 = val1, raw2 = val2)
        self.asm._emit(insn)

class RawProgramEmitter(Emitter):
    'Emitter of raw whole program. (DANGER)'

    def _emit(self, val, size):
        prog = enc.RawProgram(raw = val, size = size)
        self.asm._emit(prog)

class LabelNameSpace(object):
    'Label namespace controller.'

    def __init__(self, asm, name):
        super(LabelNameSpace, self).__init__()
        self.asm = asm
        self.name = name

    def __enter__(self):
        self.asm._label_name_spaces.append(self.name)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.asm._label_name_spaces.pop()

#================================= Assembler ==================================


class Assembler(object):
    'QPU Assembler.'

    _REGISTERS = REGISTERS

    def __init__(self, sanity_check=False):
        self._instructions = []
        self._program_counter = 0
        self._labels = []
        self._label_name_spaces = []
        self._backpatch_list = []    # list of (instruction index, label)

        self._add = AddEmitter(self)
        self._mul = MulEmitter(self)
        self._load = LoadEmitter(self)
        self._branch = BranchEmitter(self)
        self._sema = SemaEmitter(self)
        self._raw = RawEmitter(self)
        self.L = LabelEmitter(self)

        self.namespace = lambda ns: LabelNameSpace(self, ns)

        self.sanity_check = sanity_check

    def _emit(self, insn, increment=True):
        """Emit new instruction ``insn`` if increment is True else replace the
        last instruction with ``insn``.
        """

        if increment:
            self._instructions.append(insn)
            self._program_counter += 8
        else:
            add_op = self._instructions[-1]
            if self.sanity_check:
                insn.verbose = ComposedInstr (add_op.verbose, insn.verbose)
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

    def _emit_raw(self, *args, **kwargs):
        return self._raw._emit(*args, **kwargs)

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
            assert(isinstance(insn, enc.BranchInsn))
            assert(insn.rel)

            insn.immediate = labels[label] - 8*(i + 4)
        self._backpatch_list = []

    def _add_backpatch_item(self, target):
        self._backpatch_list.append((len(self._instructions), target))

    def _get_code(self):
        'Convert list of _instructions to executable bytes.'

        self._backpatch()
        return b''.join(insn.to_bytes() for insn in self._instructions)

    def _generate_label_name(self, name):
        return '.'.join(self._label_name_spaces + [name])

#=================================== Alias ====================================

def alias(f):
    setattr(Assembler, f.__name__, f)

for name, code in enc._ADD_INSN.items():
    setattr(Assembler, name, _partialmethod(Assembler._emit_add, code))

for name, code in enc._MUL_INSN.items():
    if name not in enc._ADD_INSN:
        setattr(Assembler, name, _partialmethod(Assembler._emit_mul, code))
    setattr(MulEmitter, name, _partialmethod(MulEmitter._emit, code))

for name, code in enc._BRANCH_INSN.items():
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

@alias
def raw(asm, val1, val2):
    asm._emit_raw(val1, val2);

@alias
def raw_program(asm, val):
    assert(len(val) % 8 == 0)

    for i in range(len(val) // 8):
        head = i * 8
        raw1 = int.from_bytes(val[head:head + 4], 'little')
        raw2 = int.from_bytes(val[head + 4:head + 8], 'little')
        asm.raw(raw1, raw2)

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
    args = inspect.signature(f).parameters

    if 'asm' not in args:
        raise AssembleError('Argument named \'asm\' is necessary')

    def decorate(f):
        @wraps(f)
        def decorated(asm, *args, **kwargs):
            g = f.__globals__
            for reg in Assembler._REGISTERS:
                g[str(reg)] = asm._REGISTERS[str(reg)]
            g['ra'] = [g['ra{}'.format(i)] for i in range(32)]
            g['rb'] = [g['rb{}'.format(i)] for i in range(32)]
            for i in dir(Assembler):
                if i[0] != '_':
                    g[str(i)] = getattr(asm, str(i))
            g['L'] = asm.L
            g['namespace'] = asm.namespace
            f(asm, *args, **kwargs)
        return decorated

    return decorate(f)

def _assemble(f, *args, **kwargs):
    'Assemble QPU program to byte string.'
    if kwargs.get('sanity_check', None):
        asm = Assembler(sanity_check=True)
        del kwargs['sanity_check']
    else:
        asm = Assembler()
    f(asm, *args, **kwargs)
    if asm.sanity_check:
        check_main(asm._instructions, asm._labels)
    return asm

def assemble(f, *args, **kwargs):
    return _assemble(f, *args, **kwargs)._get_code()

def get_label_positions(f, *args, **kwargs):
    asm = _assemble(f, *args, **kwargs)
    asm._get_code()
    return asm._labels

def sanity_check(f, *args, **kwargs):
    asm = Assembler(sanity_check=True)
    f(asm, *args, **kwargs)
    return check_main(asm._instructions, asm._labels)

def print_qbin(program, file = sys.stdout, *args, **kwargs):
    'Print QPU program as .qbin.'
    if hasattr(program, '__call__'):
        program = assemble(program, *args, **kwargs)
    code = memoryview(program).tobytes()
    code = map(ord, code) if type(code) is str else code
    assert(len(code) % 8 == 0)
    for i in range(len(code) // 8):
        for j in range(7, -1, -1):
            print("%08d" % int(bin(code[i * 8 + j])[2:]), end = ' ' if j != 0 else '', file = file)
        print(file = file)

def print_qhex(program, file = sys.stdout, *args, **kwargs):
    'Print QPU program as .qhex.'
    if hasattr(program, '__call__'):
        program = assemble(program, *args, **kwargs)
    code = memoryview(program).tobytes()
    code = map(ord, code) if type(code) is str else code
    assert(len(code) % 8 == 0)
    for c in zip(*[iter(code)]*8):
        print("0x{3:02X}{2:02X}{1:02X}{0:02X}, 0x{7:02X}{6:02X}{5:02X}{4:02X},".format(*c), file = file)

def save_bin(program, file, *args, **kwargs):
    'Save QPU program as .bin.'
    if hasattr(program, '__call__'):
        program = assemble(program, *args, **kwargs)
    code = memoryview(program).tobytes()
    code = map(ord, code) if type(code) is str else code
    assert(len(code) % 8 == 0)
    with open(file, 'wb') as f:
        f.write(code)

def save_asm(program, file, *args, **kwargs):
    'Save QPU program and label information.'
    program = _assemble(program, *args, **kwargs)
    with open(file, 'wb') as f:
        pickle.dump(program._get_code(), f)
        pickle.dump(program._labels, f)

def restore_asm(file, *args, **kwargs):
    'Restore QPU program and label information.'
    with open(file, 'rb') as f:
        code = pickle.load(f)
        labels = pickle.load(f)
        return (code, labels)
