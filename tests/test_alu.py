'Test of Add ALU'

import numpy as np
from random import getrandbits

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def boilerplate(asm, f, nout):
    setup_dma_load(nrows=2)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=2)
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
                num_threads=1,
                program=drv.program(boilerplate, code, output_shape[0]),
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
    mul24(r2, r0, r1)
    mov(vpm, r2)

def test_imul24_op():
    X = np.random.randint(0, 2**24-1, (2, 16)).astype('uint32')
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
    mov(r0, vpm)
    for op in ['bnot', 'clz']:
        getattr(asm, op)(r1, r0)
        mov(vpm, r1)

def test_bitwise_un_ops():
    X = np.array([getrandbits(32) for i in range(16)]).astype('uint32')
    Y = run_code(bitwise_un_ops, X, (2, 16), 'uint32')
    assert all(Y[0] == ~X)
    assert all(Y[1] == np.vectorize(count_leading_zeros)(X))

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
    for op in ['fadd', 'fsub', 'fmul', 'fmin', 'fmax']:
        getattr(asm, op)(r2, r0, r1)
        mov(vpm, r2)

def test_float_ops():
    X = np.random.randn(2, 16).astype('float32')
    Y = run_code(float_ops, X, (5, 16), 'float32')

    assert np.allclose(X[0] + X[1], Y[0], rtol=1e-3)
    assert np.allclose(X[0] - X[1], Y[1], rtol=1e-3)
    assert np.allclose(X[0] * X[1], Y[2], rtol=1e-3)
    assert all(np.minimum(X[0], X[1]) == Y[3])
    assert all(np.maximum(X[0], X[1]) == Y[4])

#============================== Type conversion ===============================

@qpu
def type_conv(asm):
    mov(r0, vpm)

    ftoi(r1, r0)
    itof(r2, r1)

    mov(vpm, r1)
    mov(vpm, r2)

def test_type_conv():
    X = np.random.randn(16).astype('float32') * 1000

    Y = run_code(type_conv, X, (2, 16), 'int32')

    assert all(X.astype('int32') == Y[0])
    assert all(X.astype('int32') == np.ndarray(16, 'float32', Y[1]))

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

