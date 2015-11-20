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

def run_code(code, X, nout, dtype):
    with Driver() as drv:
        X = drv.copy(X)
        Y = drv.alloc((nout, 16), dtype=dtype)
        drv.execute(
                num_threads=1,
                program=drv.program(boilerplate, code, nout),
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
    Y = run_code(int_ops, X, 4, 'int32')
    assert all(Y[0] == X[0] + X[1])
    assert all(Y[1] == X[0] - X[1])
    assert all(Y[2] == np.minimum(X[0], X[1]))
    assert all(Y[3] == np.maximum(X[0], X[1]))

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
    Y = run_code(bitwise_bin_ops, X, 3, 'uint32')
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
    Y = run_code(bitwise_un_ops, X, 2, 'uint32')
    print(np.vectorize(hex)(X))
    print(np.vectorize(hex)(Y))
    assert all(Y[0] == ~X)
    assert all(Y[1] == np.vectorize(count_leading_zeros)(X))

#============================== Shift operation ===============================

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
    Y = run_code(shift_ops, X, 4, 'uint32')
    print(np.vectorize(hex)(X))
    print(np.vectorize(hex)(Y))
