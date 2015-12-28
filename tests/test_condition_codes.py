'Test of Condition codes'

import numpy as np
from random import getrandbits

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def boilerplate(asm, f, nout):
    setup_dma_load(nrows=1)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=1)
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
                program=drv.program(boilerplate, code, output_shape[0]),
                uniforms=[X.address, Y.address]
                )
        return np.copy(Y)

#================================== cond_add ==================================

@qpu
def cond_add(asm):
    mov(ra0, vpm)
    nop()

    mov(r1, ra0)
    iadd(r1, r1, 1, cond='never')
    mov(vpm, r1)

    ldi(r2, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    ldi(r3, [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    isub(null, r2, r3)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='zs', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='zc', set_flags = False)
    mov(vpm, r1)

    isub(null, r3, r2)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='ns', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='nc', set_flags = False)
    mov(vpm, r1)

    isub(null, r3, r2)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='cs', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    iadd(r1, r1, 1, cond='cc', set_flags = False)
    mov(vpm, r1)

def test_cond_add():
    X = np.array([getrandbits(32) for i in range(16)]).astype('uint32')
    Y = run_code(cond_add, X, (7, 16), 'uint32')
    assert all(X == Y[0])
    assert all(X[:8] == Y[1,:8])
    assert all(X[8:]+1 == Y[1,8:])
    assert all(X[:8]+1 == Y[2,:8])
    assert all(X[8:] == Y[2,8:])
    assert all(X[:8]+1 == Y[3,:8])
    assert all(X[8:] == Y[3,8:])
    assert all(X[:8] == Y[4,:8])
    assert all(X[8:]+1 == Y[4,8:])
    assert all(X[:8]+1 == Y[5,:8])
    assert all(X[8:] == Y[5,8:])
    assert all(X[:8] == Y[6,:8])
    assert all(X[8:]+1 == Y[6,8:])

#================================== cond_mul ==================================

@qpu
def cond_mul(asm):
    mov(ra0, vpm)
    nop()

    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='never')
    mov(vpm, r1)

    ldi(r2, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    ldi(r3, [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    isub(null, r2, r3)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='zs', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='zc', set_flags = False)
    mov(vpm, r1)

    isub(null, r3, r2)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='ns', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='nc', set_flags = False)
    mov(vpm, r1)

    isub(null, r3, r2)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='cs', set_flags = False)
    mov(vpm, r1)
    mov(r1, ra0)
    fmul(r1, r1, 2.0, cond='cc', set_flags = False)
    mov(vpm, r1)

def test_cond_mul():
    X = np.random.randn(16).astype('float32')
    Y = run_code(cond_mul, X, (7, 16), 'float32')
    assert all(X == Y[0])
    assert all(X[:8] == Y[1,:8])
    assert all(X[8:]*2 == Y[1,8:])
    assert all(X[:8]*2 == Y[2,:8])
    assert all(X[8:] == Y[2,8:])
    assert all(X[:8]*2 == Y[3,:8])
    assert all(X[8:] == Y[3,8:])
    assert all(X[:8] == Y[4,:8])
    assert all(X[8:]*2 == Y[4,8:])
    assert all(X[:8]*2 == Y[5,:8])
    assert all(X[8:] == Y[5,8:])
    assert all(X[:8] == Y[6,:8])
    assert all(X[8:]*2 == Y[6,8:])
