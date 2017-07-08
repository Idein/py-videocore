'Test of rotate operation'

import random
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
        Y = drv.alloc((nout, 16), dtype=np.uint32)
        drv.execute(
                n_threads=1,
                program=drv.program(boilerplate, code, nout),
                uniforms=[X.address, Y.address]
                )
        return np.copy(Y)

@qpu
def rotate_r0123(asm):
    mov(r0, vpm)

    for i in range(1, 16):
        mov(r1, 0)
        nop()
        rotate(r1, r0, i)
        mov(vpm, r1)
        nop()

    for i in range(0, 16):
        mov(r1, 0)
        mov(broadcast, i)
        nop()
        rotate(r1, r0, r5)
        mov(vpm, r1)
        nop()

@qpu
def rotate_r4(asm):
    mov(r0, vpm)
    mov(tmu0_s, r0)
    nop(sig='load tmu0')

    mov(vpm, r4)

    for i in range(1, 16):
        mov(r1, 0)
        nop()
        rotate(r1, r4, i)
        mov(vpm, r1)
        nop()

    for i in range(0, 16):
        mov(r1, 0)
        mov(broadcast, i)
        nop()
        rotate(r1, r4, r5)
        mov(vpm, r1)
        nop()

@qpu
def rotate_ra(asm):
    mov(ra0, vpm)

    for i in range(1, 16):
        mov(r1, 0)
        nop()
        rotate(r1, ra0, i)
        nop()
        mov(vpm, r1)
        nop()

    for i in range(0, 16):
        mov(r1, 0)
        mov(broadcast, i)
        nop()
        rotate(r1, ra0, r5)
        nop()
        mov(vpm, r1)
        nop()

def list_full_rotate(l, n):
    return np.append(l[-n:], l[:-n])

def list_half_rotate(l, n):
    r = np.array([], dtype = l.dtype)
    for i in range(4):
        r = np.append(r, list_full_rotate(l[4*i:4*(i+1)], n%4))
    return r

def test_rotate_r0123():
    X = np.array([random.getrandbits(32) for i in range(16)]).astype(np.uint32)
    Y = run_code(rotate_r0123, X, 15+16)
    for i in range(1, 16):
        Y_ref = list_full_rotate(X, i)
        assert np.alltrue(Y[i-1] == Y_ref)
    for i in range(0, 16):
        Y_ref = list_full_rotate(X, i)
        assert np.alltrue(Y[15+i] == Y_ref)

def test_rotate_r4():
    d = np.array([random.getrandbits(32) for i in range(16)]).astype(np.uint32)
    with Driver() as drv:
        addr = drv.copy(d).address
        X = np.array([addr+4*i for i in range(16)], dtype=np.uint32)
        Y = run_code(rotate_r4, X, 1+15+16)
        assert np.alltrue(Y[0] == d)
        for i in range(1, 16):
            Y_ref = list_half_rotate(d, i)
            assert np.alltrue(Y[i] == Y_ref)
        for i in range(0, 16):
            Y_ref = list_half_rotate(d, i)
            assert np.alltrue(Y[16+i] == Y_ref)

def test_rotate_ra():
    X = np.array([random.getrandbits(32) for i in range(16)]).astype(np.uint32)
    Y = run_code(rotate_ra, X, 15+16)
    for i in range(1, 16):
        Y_ref = list_half_rotate(X, i)
        assert np.alltrue(Y[i-1] == Y_ref)
    for i in range(0, 16):
        Y_ref = list_half_rotate(X, i)
        assert np.alltrue(Y[15+i] == Y_ref)
