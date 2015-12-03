'Test of ALU and load instructions'

import numpy as np
from random import getrandbits

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def boilerplate(asm, f, nin, nout):
    setup_dma_load(nrows=nin)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=nin)
    setup_vpm_write()

    f(asm)

    setup_dma_store(nrows=nout)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def run_code(code, X, output_shape, output_type):
    with Driver() as drv:
        X = drv.copy(X)
        Y = drv.alloc(output_shape, dtype=output_type)
        drv.execute(
                n_threads=1,
                program=drv.program(boilerplate, code, X.shape[0],
                                    output_shape[0]),
                uniforms=[X.address, Y.address]
                )
        return np.copy(Y)

#============================= Integer arithmetic =============================

@qpu
def int_ops(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    for op in ['iadd', 'isub', 'imin', 'imax']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_int_ops():
    X = np.array(
            [getrandbits(32) for i in range(2*16)]
            ).reshape(2, 16).astype('int32')
    Y = run_code(int_ops, X, (4, 16), 'int32')
    assert all(Y[0] == X[0] + X[1])
    assert all(Y[1] == X[0] - X[1])
    assert all(Y[2] == np.minimum(X[0], X[1]))
    assert all(Y[3] == np.maximum(X[0], X[1]))

@qpu
def imul24_op(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    imul24(r2, r0, r1)
    mov(vpm, r2)

def test_imul24_op():
    X = np.random.randint(0, 2**24, (2, 16)).astype('uint32')
    Y = run_code(imul24_op, X, (1, 16), 'uint32')
    assert all((X[0]*X[1]).astype('uint32') == Y[0])

#========================== Bitwise binary operation ==========================

@qpu
def bitwise_bin_ops(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    for op in ['band', 'bor', 'bxor']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_bitwise_bin_ops():
    X = np.array(
            [getrandbits(32) for i in range(2*16)]
            ).reshape(2, 16).astype('uint32')
    Y = run_code(bitwise_bin_ops, X, (3, 16), 'uint32')
    assert all(Y[0] == X[0] & X[1])
    assert all(Y[1] == X[0] | X[1])
    assert all(Y[2] == X[0] ^ X[1])

#========================== Bitwise unary operation ==========================

def count_leading_zeros(n):
    bit = 0x80000000
    count = 0
    while not (n & bit):
        count += 1
        bit >>= 1
    return count

@qpu
def bitwise_un_ops(asm):
    mov(r1, vpm)
    for op in ['bnot', 'clz']:
        getattr(asm, op)(r2, r1)
        mov(vpm, r2)

def test_bitwise_un_ops():
    X = np.array(
            [getrandbits(32) for i in range(16)]
            ).reshape(1, 16).astype('uint32')
    Y = run_code(bitwise_un_ops, X, (2, 16), 'uint32')
    assert all(Y[0] == ~X[0])
    assert all(Y[1] == np.vectorize(count_leading_zeros)(X[0]))

#============================== Shift operation ===============================

def rotate_right(n, s):
    return ((n << (32-s)) | (n >> s)) & 0xffffffff

@qpu
def shift_ops(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    for op in ['shr', 'asr', 'ror', 'shl']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_shift_ops():
    X = np.zeros((2, 16), dtype='uint32')
    X[0] = np.array([getrandbits(32) for i in range(16)])
    X[1] = np.random.randint(0, 32, 16)
    Y = run_code(shift_ops, X, (4, 16), 'uint32')
    assert all(X[0] >> X[1] == Y[0])
    assert all(X[0].astype('int32') >> X[1] == Y[1].astype('int32'))
    assert all(np.vectorize(rotate_right)(X[0], X[1]) == Y[2])
    assert all(X[0] << X[1] == Y[3])

#========================= Floating-point arithmetic ==========================

@qpu
def float_ops(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    for op in ['fadd', 'fsub', 'fmul', 'fmin', 'fmax', 'fminabs', 'fmaxabs']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_float_ops():
    X = np.random.randn(2, 16).astype('float32')
    Y = run_code(float_ops, X, (7, 16), 'float32')

    assert np.allclose(X[0] + X[1], Y[0], rtol=1e-6)
    assert np.allclose(X[0] - X[1], Y[1], rtol=1e-6)
    assert np.allclose(X[0] * X[1], Y[2], rtol=1e-6)
    assert all(np.minimum(X[0], X[1]) == Y[3])
    assert all(np.maximum(X[0], X[1]) == Y[4])
    assert all(np.minimum(np.abs(X[0]), np.abs(X[1])) == Y[5])
    assert all(np.maximum(np.abs(X[0]), np.abs(X[1])) == Y[6])

#============================== Type conversion ===============================

@qpu
def type_conv(asm):
    mov(r0, vpm)

    ftoi(r1, r0)
    itof(r2, r1)

    mov(vpm, r1)
    mov(vpm, r2)

def test_type_conv():
    X = np.random.randn(1, 16).astype('float32') * 1000

    Y = run_code(type_conv, X, (2, 16), 'int32')

    assert all(X[0].astype('int32') == Y[0])
    assert all(X[0].astype('int32') == np.ndarray(16, 'float32', Y[1]))


#======================== 8-bit saturation arithmetic =========================


@qpu
def v8sat_ops(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    for op in ['v8adds', 'v8subs', 'v8muld', 'v8min', 'v8max']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_v8sat_ops():
    X = np.array(
            [getrandbits(8) for i in range(2*4*16)]
            ).reshape(2, 4*16).astype('uint8')
    Y = run_code(v8sat_ops, X, (5, 4*16), 'uint8')

    u_X = X
    s_X = X.astype('int32')
    assert all(np.clip(s_X[0] + s_X[1], 0, 255) == Y[0])
    assert all(np.clip(s_X[0] - s_X[1], 0, 255) == Y[1])
    assert all(np.abs((s_X[0]*s_X[1]/255.0).astype('int32') - Y[2]) <= 1)
    assert all(np.minimum(u_X[0], u_X[1]) == Y[3])
    assert all(np.maximum(u_X[0], u_X[1]) == Y[4])


#============================== Small immediate ===============================

@qpu
def small_imm_int(asm):
    mov(r0, vpm)
    for imm in range(-16, 16):
        iadd(vpm, r0, imm)

def test_small_imm_int():
    X = np.array(
            [getrandbits(32) for i in range(16)]
            ).reshape(1, 16).astype('int32')
    Y = run_code(small_imm_int, X, (32, 16), 'int32')
    for i, imm in enumerate(range(-16, 16)):
        assert all(X[0] + imm == Y[i])

@qpu
def small_imm_float(asm):
    mov(r0, vpm)
    for e in range(-8, 8):
        fmul(vpm, r0, 2.0**e)

def test_small_imm_float():
    X = np.random.randn(1, 16).astype('float32')
    Y = run_code(small_imm_float, X, (16, 16), 'float32')
    for i, e in enumerate(range(-8, 8)):
        assert np.allclose(X[0] * 2.0**e, Y[i], rtol=1e-6)

#=============================== Load operation ===============================

@qpu
def load_two_dest(asm):
    mov(r1, vpm)
    ldi(r2, r3, 0x12345678)
    iadd(vpm, r1, r2)
    iadd(vpm, r1, r3)

def test_load_two_dest():
    X = np.ones((1, 16), dtype='uint32')
    Y = run_code(load_two_dest, X, (2, 16), 'uint32')

    assert np.all(X[0] + 0x12345678 == Y[0])
    assert np.all(X[0] + 0x12345678 == Y[1])

#=========================== Per-element immediate ============================

PER_ELMT_UNSIGNED_VALUES = np.random.randint(0, 4, 16)
PER_ELMT_SIGNED_VALUES = np.random.randint(-2, 2, 16)

@qpu
def per_elmt_imm(asm):
    mov(r0, vpm)
    ldi(r1, PER_ELMT_UNSIGNED_VALUES)
    iadd(vpm, r0, r1)
    ldi(r2, PER_ELMT_SIGNED_VALUES)
    iadd(vpm, r0, r2)

def test_per_elmt_imm():
    X = np.array(
            [getrandbits(32) for i in range(16)]
            ).reshape(1, 16).astype('int32')
    Y = run_code(per_elmt_imm, X, (2, 16), 'int32')
    assert all(X[0] + PER_ELMT_UNSIGNED_VALUES == Y[0])
    assert all(X[0] + PER_ELMT_SIGNED_VALUES == Y[1])

#============================== Vector rotatoin ===============================

@qpu
def vector_rotation(asm):
    mov(r0, vpm)
    ldi(r1, 1.0)
    for shift in range(1, 16):
        rotate(vpm, r0, shift)
    for shift in range(-16, 16):
        ldi(r5_elm0, shift)
        nop()
        rotate(vpm, r0, r5)

def test_vector_rotation():
    X = np.array(
            [getrandbits(32) for i in range(16)]
            ).reshape(1, 16).astype('uint32')
    Y = run_code(vector_rotation, X, (15 + 32, 16), 'uint32')

    for i, shift in enumerate(range(1, 16)):
        assert all(np.roll(X[0], shift) == Y[i])
    for i, shift in enumerate(range(-16, 16)):
        assert all(np.roll(X[0], shift) == Y[i+15])

@qpu
def add_and_rotation(asm):
    mov(r0, vpm)
    mov(r1, vpm)
    rotate(r2, r0, -1)
    fadd(r0, r0, r2).rotate(r3, r1, -1)
    nop()
    fadd(r1, r1, r3).rotate(r2, r0, -2)
    nop()
    fadd(r0, r0, r2).rotate(r3, r1, -2)
    nop()
    fadd(r1, r1, r3).rotate(r2, r0, -4)
    nop()
    fadd(r0, r0, r2).rotate(r3, r1, -4)
    nop()
    fadd(r1, r1, r3).rotate(r2, r0, -8)
    nop()
    fadd(r0, r0, r2).rotate(r3, r1, -8)
    fadd(r1, r1, r3)

    mov(vpm, r0)
    mov(vpm, r1)

def test_add_and_rotation():
    X = np.random.randn(2, 16).astype('float32')
    Y = run_code(add_and_rotation, X, (2, 16), 'float32')
    assert np.allclose(Y[0], X[0].sum(), rtol=1e-4)
    assert np.allclose(Y[1], X[1].sum(), rtol=1e-4)

#=============================== Miscellaneous ================================

@qpu
def shared_small_imm(asm):
    mov(r0, vpm)
    fadd(r1, r0, 2.0).fmul(r2, r0, 2.0)
    mov(vpm, r1)
    mov(vpm, r2)

def test_shared_small_imm():
    X = np.random.randn(1, 16).astype('float32')
    Y = run_code(shared_small_imm, X, (2, 16), 'float32')
    assert np.allclose(X[0] + 2.0, Y[0], rtol=1e-6)
    assert np.allclose(X[0] * 2.0, Y[1], rtol=1e-6)
