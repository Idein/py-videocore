'Test of special function unit'

import numpy as np

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

def run_code(code, X, nout):
    with Driver() as drv:
        X = drv.copy(X)
        Y = drv.alloc((nout, 16), dtype='float32')
        drv.execute(
                n_threads=1,
                program=drv.program(boilerplate, code, nout),
                uniforms=[X.address(0, 0), Y.address(0, 0)]
                )
        return np.copy(Y)

@qpu
def sfu(asm):
    mov(r0, vpm)
    for unit in [sfu_recip, sfu_recipsqrt, sfu_exp2, sfu_log2]:
        mov(unit, r0)
        nop()
        nop()
        mov(vpm, r4)

def test_sfu():
    X = np.random.uniform(0, 1, 16).astype('float32')
    Y = run_code(sfu, X, 4)
    assert np.allclose(1/X, Y[0], rtol=1e-4)
    assert np.allclose(1/np.sqrt(X), Y[1], rtol=1e-4)
    assert np.allclose(np.exp2(X), Y[2], rtol=1e-4)
    assert np.allclose(np.log2(X), Y[3], rtol=1e-2)

@qpu
def zero_division(asm):
    mov(r0, vpm)
    for unit in [sfu_recip, sfu_recipsqrt]:
        mov(unit, r0)
        nop()
        nop()
        mov(vpm, r4)

def test_zero_division():
    X = np.zeros(16).astype('float32')
    Y = run_code(zero_division, X, 2)
    assert np.all(np.isinf(Y[0]))
    assert np.all(np.isinf(Y[1]))

@qpu
def sqrt_of_negative(asm):
    mov(sfu_recipsqrt, vpm)
    nop()
    nop()
    mov(vpm, r4)

def test_sqrt_of_negative():
    X = np.random.uniform(-1, 0, 16).astype('float32')
    Y = run_code(sqrt_of_negative, X, 1)
    assert np.allclose(1/np.sqrt(np.abs(X)), Y[0], rtol=1e-4)

@qpu
def log_of_negative(asm):
    mov(sfu_log2, vpm)
    nop()
    nop()
    mov(vpm, r4)

def test_log_of_negative():
    X = np.random.uniform(-1, 0, 16).astype('float32')
    Y = run_code(log_of_negative, X, 1)
    assert np.allclose(np.log2(np.abs(X)), Y[0], rtol=1e-2)

