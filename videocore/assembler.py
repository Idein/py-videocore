# QPU assembler
# Copytirhg (c) 2015 Koichi Nakamura

import numpy
from ctypes import Structure, c_ulong, string_at, byref, sizeof
from struct import pack, unpack
import inspect, ast
from functools import partial

class partialmethod(partial):
    def __get__(self, obj, type):
        return partial(self.func, obj, *(self.args or ()), **(self.keywords or {}))

class AssembleError(Exception):
    'Exception related to QPU assembler'
    pass

REG_READ_A  = 0x8
REG_READ_B  = 0x4
REG_WRITE_A = 0x2
REG_WRITE_B = 0x1

UNPACK_CODES = { pat: code for code, pat in enumerate([
    'no unpack', '16a', '16b', 'rep 8d', '8a', '8b', '8c', '8d'])}
PACK_CODES = { pat: code for code, pat in enumerate([
    'no pack', '16a', '16b', '8888', '8a', '8b', '8c', '8d',
    '32 sat', '16a sat', '16b sat', '8888 sat', '8a sat', '8b sat', '8c sat', '8d sat'
    ])}
MUL_PACK_CODES = { '8888 mul': 3, '8a mul': 4 }

class Register(object):
    """Registers.

    This class implements QPU registers.
    'spec' is bitwise-or of following flags:
    - 0x8: can be used as a read register in bank A
    - 0x4: can be used as a read register in bank B
    - 0x2: can be used as a write register in bank A
    - 0x1: can be used as a write register in bank B
    """

    def __init__(self, name, addr, spec, pack = 0, unpack = 0, pm = False):
        self.name        = name
        self.addr        = addr
        self.spec        = spec
        self.pack_bits   = pack
        self.unpack_bits = unpack
        self.pm_bits     = pm

    def __str__(self):
        return self.name

    def unpack(self, pat):
        if self.name in ['r0', 'r1', 'r2', 'r3']:
            raise AssembleError('Accumulators r0-r3 have no unpack functionality')
        if self.name == 'r4':
            spec = REG_READ_A | REG_READ_B
        else:
            if not (self.spec & REG_READ_A):
                raise AssembleError('Unpacking is not supported for the register {}'.format(self))
            spec = REG_READ_A # packing can be used for regfile A read registers.
        return Register(name = self.name, addr = self.addr, spec = spec,
                unpack = UNPACK_CODES[pat], pm = self.name == 'r4')

    def pack(self, pat):
        if self.name in ['r0', 'r1', 'r2', 'r3']:
            raise AssembleError('Accumulators r0-r3 have no pack functionality')
        if pat in PACK_CODES:
            if not (self.spec & REG_WRITE_A):
                raise AssembleError('Packing is not supported for the register {}'.format(self))
            spec = REG_WRITE_A
            pack = PACK_CODES[pat]
            pm   = False
        else:
            if not (self.spec & (REG_WRITE_A | REG_WRITE_B)):
                raise AssembleError('Packing is not supported for the register {}'.format(self))
            spec = self.spec & (REG_WRITE_A | REG_WRITE_B)
            pack = MUL_PACK_CODES[pat]
            pm   = True

        return Register(name = self.name, addr = self.addr, spec = spec, pack = pack, pm = pm)

REGISTERS = {}

# General purpose registers
REGISTERS.update({
    'r'+file+str(addr): Register('r'+file+str(addr), addr, {'a':0xA,'b':0x5}[file])
    for file in ['a', 'b'] for addr in range(64)
    })

# Register file I/O
REGISTERS.update({
    name: Register(name, addr, spec) for name, addr, spec in [
    ('uniform'           , 32 , 0xC),
    ('r0'                , 32 , 0x3),
    ('r1'                , 33 , 0x3),
    ('r2'                , 34 , 0x3),
    ('r3'                , 35 , 0x3),
    ('r4'                , 36 , 0x0),
    ('r5'                , 37 , 0x3),
    ('varying_read'      , 35 , 0xC),
    ('tmu_noswap'        , 36 , 0x3),
    ('host_interrupt'    , 38 , 0x3),
    ('element_number'    , 38 , 0x8),
    ('qpu_number'        , 38 , 0x4),
    ('null'              , 39 , 0xF),
    ('uniforms_address'  , 40 , 0x3),
    ('x_pixel_coord'     , 41 , 0x8),
    ('y_pixel_coord'     , 41 , 0x4),
    ('quad_x'            , 41 , 0x2),
    ('quad_y'            , 41 , 0x1),
    ('ms_flags'          , 42 , 0xA),
    ('rev_flag'          , 42 , 0x5),
    ('tlb_stencil_setup' , 43 , 0x3),
    ('tlb_z'             , 44 , 0x3),
    ('tlb_color_ms'      , 45 , 0x3),
    ('tlb_color_all'     , 46 , 0x3),
    ('tlb_alpha_mask'    , 47 , 0x3),
    ('vpm'               , 48 , 0xF),
    ('vpm_ld_busy'       , 49 , 0x8),
    ('vpm_st_busy'       , 49 , 0x4),
    ('vpmvcd_rd_setup'   , 49 , 0x2),
    ('vpmvcd_wr_setup'   , 49 , 0x1),
    ('vpm_ld_wait'       , 50 , 0x8),
    ('vpm_st_wait'       , 50 , 0x4),
    ('vpm_ld_addr'       , 50 , 0x2),
    ('vpm_st_addr'       , 50 , 0x1),
    ('mutex_acquire'     , 51 , 0xC),
    ('mutex_release'     , 51 , 0x3),
    ('sfu_recip'         , 52 , 0x3),
    ('sfu_recipsqrt'     , 53 , 0x3),
    ('sfu_exp'           , 54 , 0x3),
    ('sfu_log'           , 55 , 0x3),
    ('tmu0_s'            , 56 , 0x3),
    ('tmu0_t'            , 57 , 0x3),
    ('tmu0_r'            , 58 , 0x3),
    ('tmu0_b'            , 59 , 0x3),
    ('tmu1_s'            , 60 , 0x3),
    ('tmu1_t'            , 61 , 0x3),
    ('tmu1_r'            , 62 , 0x3),
    ('tmu1_b'            , 63 , 0x3)
]})

# Table from immediate values to its code
SMALL_IMMEDIATES = {}
SMALL_IMMEDIATES.update({repr(i): i for i in range(16)})
SMALL_IMMEDIATES.update({repr(i-16): i+16 for i in range(16)})
SMALL_IMMEDIATES.update({repr(2.0**i): i+32 for i in range(8)})
SMALL_IMMEDIATES.update({repr(2.0**(i-8)): i+40 for i in range(8)})

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

    code = SMALL_IMMEDIATES.get(repr(val))
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

class Insn(Structure):
    def to_bytes(self):
        return string_at(byref(self), sizeof(self))

class AluInsn(Insn):
    _fields_ = [
        ('mul_b',     c_ulong, 3), ('mul_a',    c_ulong, 3), ('add_b',     c_ulong, 3),
        ('add_a',     c_ulong, 3), ('raddr_b',  c_ulong, 6), ('raddr_a',   c_ulong, 6),
        ('op_add',    c_ulong, 5), ('op_mul',   c_ulong, 3), ('waddr_mul', c_ulong, 6),
        ('waddr_add', c_ulong, 6), ('ws',       c_ulong, 1), ('sf',        c_ulong, 1),
        ('cond_mul',  c_ulong, 3), ('cond_add', c_ulong, 3), ('pack',      c_ulong, 4),
        ('pm',        c_ulong, 1), ('unpack',   c_ulong, 3), ('sig',       c_ulong, 4)
    ]

class BranchInsn(Insn):
    _fields_ = [
        ('immediate', c_ulong, 32), ('waddr_mul', c_ulong, 6), ('waddr_add', c_ulong, 6),
        ('ws',        c_ulong, 1), ('raddr_a',    c_ulong, 5), ('reg',       c_ulong, 1),
        ('rel',       c_ulong, 1), ('cond_br',    c_ulong, 4), ('dontcare',  c_ulong, 4),
        ('sig',       c_ulong, 4)
    ]

class LoadInsn(Insn):
    _fields_ = [
        ('immediate', c_ulong, 32), ('waddr_mul', c_ulong, 6), ('waddr_add', c_ulong, 6),
        ('ws',        c_ulong, 1), ('sf',         c_ulong, 1), ('cond_mul',  c_ulong, 3),
        ('cond_add',  c_ulong, 3), ('pack',       c_ulong, 4), ('pm',        c_ulong, 1),
        ('unpack',    c_ulong, 3), ('sig',        c_ulong, 4)
    ]

class SemaInsn(Insn):
    _fields_ = [
        ('semaphore', c_ulong, 4), ('sa',        c_ulong, 1), ('dontcare',  c_ulong, 27),
        ('waddr_mul', c_ulong, 6), ('waddr_add', c_ulong, 6), ('ws',        c_ulong, 1),
        ('sf',        c_ulong, 1), ('cond_mul',  c_ulong, 3), ('cond_add',  c_ulong, 3),
        ('pack',      c_ulong, 4), ('pm',        c_ulong, 1), ('unpack',    c_ulong, 3),
        ('sig',       c_ulong, 4)
    ]

SIGNALING_BITS = {
    'breakpoint'                : 0,
    'no signal'                 : 1,
    'thread switch'             : 2,
    'thread end'                : 3,
    'wait scoreboard'           : 4,
    'unlock scoreboard'         : 5,
    'last thread switch'        : 6,
    'load coverage'             : 7,
    'load color'                : 8,
    'load color and thread end' : 9,
    'load tmu0'                 : 10,
    'load tmu1'                 : 11,
    'load alpha'                : 12,
    'alu small imm'             : 13,
    'load'                      : 14,
    'branch'                    : 15
}

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
            pm     = opd.pm_bits

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
        elif (opd.spec & REG_READ_A) and not (opd.spec & REG_READ_B):
            if raddr_a is not None and raddr_a != opd.addr:
                raise AssembleError('Too many regfile A operand {}'.format(opd))
            raddr_a = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_A
        elif not (opd.spec & REG_READ_A) and (opd.spec & REG_READ_B):
            if raddr_b is not None and raddr_b != opd.addr:
                raise AssembleError('Too many regfile B operand {}'.format(opd))
            raddr_b = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_B

    # Locate remaining operands.
    for i, opd in enumerate(operands):
        if mux[i] is not None: continue

        if not (opd.spec & (REG_READ_A | REG_READ_B)):
            raise AssembleError('{} can not be used as a read operand'.format(opd))

        if raddr_a is None and opd.spec & REG_READ_A:
            raddr_a = opd.addr
            mux[i]  = INPUT_MUX_REGFILE_A
        elif raddr_b is None and opd.spec & REG_READ_B:
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

    if add_dst.spec & REG_WRITE_A and mul_dst.spec & REG_WRITE_B:
        return add_dst.addr, mul_dst.addr, False, pack, pm
    elif mul_dst.spec & REG_WRITE_A and add_dst.spec & REG_WRITE_B:
        return add_dst.addr, mul_dst.addr, True, pack, pm
    raise AssembleError('{} and {} are not proper combination of destination registers'.format(add_dst, mul_dst))

INSTRUCTION_ALIASES = []
def syntax_sugar(f):
    INSTRUCTION_ALIASES.append(f.__name__)
    return f

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
            mul_opd1 = REGISTERS['null'],
            mul_opd2 = REGISTERS['null'],
            rotate   = 0
            ):
        add_a, add_b, mul_a, mul_b, raddr_a, raddr_b, immed, unpack, read_pm =\
                locate_read_operands(self.add_opd1, self.add_opd2, mul_opd1, mul_opd2)

        waddr_add, waddr_mul, write_swap, pack, write_pm =\
                locate_write_operands(self.add_dst, mul_dst)

        if read_pm != write_pm:
            raise AssembleError('Invalid combination of packing and unpacking')

        if immed or rotate:
            if self.sig != 'no signal':
                raise AssembleError('Signal {} can not be used with ALU small immediate instruction'.format(sig))
            self.sig = 'alu small imm'
        sig_bits = SIGNALING_BITS[self.sig]

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
            pm        = read_pm,
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
    setattr(MulInsnEmitter, name, partialmethod(MulInsnEmitter.assemble, opcode))

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
        buf   = self.insns[pc/8]
        bytes, = unpack('Q', buf)
        sig = bytes >> 60
        if sig == SIGNALING_BITS['branch']:
            return BranchInsn.from_buffer_copy(buf)
        elif sig == SIGNALING_BITS['load']:
            return LoadInsn.from_buffer_copy(buf)
        else:
            return AluInsn.from_buffer_copy(buf)

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

        if not (reg1.spec & REG_WRITE_A and reg2.spec & REG_WRITE_B):
            reg1, reg2 = reg2, reg1
            if not (reg1.spec & REG_WRITE_A):
                raise AssembleError('{} is not a write register of regfile A'.format(reg1))
            if not (reg2.spec & REG_WRITE_B):
                raise AssembleError('{} is not a write register of regfile B'.format(reg2))

        imm, unpack = pack_imm(imm)
        self.emit(LoadInsn(sig = 0xE, unpack = unpack, pm = 0, pack = 0, cond_add = 1,
            cond_mul = 1, sf = 0, ws = 0, waddr_add = reg1.addr, waddr_mul = reg2.addr,
            immediate = imm,))

    def add_insn(self, name, opcode, 
            dst  = REGISTERS['null'], # destination opdister
            opd1 = REGISTERS['null'], # operand 1
            opd2 = REGISTERS['null'], # operand 2
            sig  = 'no signal',
            set_flags = True          # if True Z,N,C flags will be set
            ):

        add_a, add_b, mul_a, mul_b, raddr_a, raddr_b, immed, unpack, read_pm = \
                locate_read_operands(opd1, opd2)

        if immed:
            if sig != 'no signal':
                raise AssembleError('Signal {} can not be used with ALU small immediate instruction'.format(sig))
            sig = 'alu small imm'
        sig_bits = SIGNALING_BITS[sig]

        if name == 'nop':
            set_flags = False

        waddr_add, waddr_mul, write_swap, pack, write_pm = \
                locate_write_operands(dst, self.REGISTERS['null'])

        if read_pm != write_pm:
            raise AssembleError('Invalid combination of packing and unpacking')
        self.emit(AluInsn(sig = sig_bits, unpack = unpack, pm = read_pm, pack = pack,
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
            if not (reg.spec & REG_READ_A):
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
    def mov(self, dst, src):
        return self.bor(dst, src, src)

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
        return self.read(self.REGISTERS['vpm_st_wait'])

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
        setattr(Assembler, name, partialmethod(Assembler.add_insn, name, opcode))
        INSTRUCTION_ALIASES.append(name)

for name in MUL_INSTRUCTIONS:
    if name not in ADD_INSTRUCTIONS:
        setattr(Assembler, name, partialmethod(Assembler.mul_insn, name))
        INSTRUCTION_ALIASES.append(name)

for cond_br, name in enumerate(BRANCH_INSTRUCTIONS):
    if name:
        setattr(Assembler, name, partialmethod(Assembler.branch_insn, cond_br))
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

def assemble(f,*args, **kwargs):
    asm = Assembler()
    f(asm = asm, *args, **kwargs)
    return asm.getcode()
