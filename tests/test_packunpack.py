'Test of pack/unpack functionality'

import numpy as np
from struct import unpack
from random import getrandbits

from videocore.assembler import qpucode
from videocore.driver import Driver

@qpucode
def boilerplate(asm, f, nrows):
    mov(r0, uniform)

    setup_dma_load(nrows=nrows)
    start_dma_load(r0)
    wait_dma_load()
    setup_vpm_read(nrows=nrows)
    setup_vpm_write()

    f(asm)

    setup_dma_store(nrows=nrows)
    start_dma_store(r0)
    wait_dma_store()
    exit()

def runcode(code, X):
    with Driver() as drv:
        X = drv.copy(X)
        drv.execute(
                num_threads=1,
                program=drv.program(boilerplate, code, X.shape[0]),
                uniforms=[X.address]
                )
        return np.copy(X)

#============================== Regfile-A Unpack ==============================

@qpucode
def unpack_rega_int(asm):
    for op in ['nop', '16a', '16b', 'rep 8d', '8a', '8b', '8c', '8d']:
        mov(ra0, vpm)
        nop()
        mov(vpm, ra0.unpack(op))
        nop()

def test_unpack_rega_int():
    X = np.array(
            [getrandbits(32) for i in range(8*16)]
            ).reshape(8, 16).astype('uint32')

    Y = runcode(unpack_rega_int, X)

    assert all(X[0] == Y[0])
    assert all(((X[1]>> 0)&0xffff).astype('int16') == Y[1].astype('int16'))
    assert all(((X[2]>>16)&0xffff).astype('int16') == Y[2].astype('int16'))
    assert all((((X[3]>>24)&0xff) * 0x01010101) == Y[3])
    assert all(((X[4]>> 0)&0xff).astype('uint8') == Y[4])
    assert all(((X[5]>> 8)&0xff).astype('uint8') == Y[5])
    assert all(((X[6]>>16)&0xff).astype('uint8') == Y[6])
    assert all(((X[7]>>24)&0xff).astype('uint8') == Y[7])

@qpucode
def unpack_rega_float(asm):
    for op in ['nop', '16a', '16b', '8a', '8b', '8c', '8d']:
        mov(ra0, vpm)
        nop()
        fadd(vpm, ra0.unpack(op), 0.0)
        nop()

def test_unpack_rega_float():
    F = np.random.randn(16)

    X = np.zeros((7, 16), dtype='uint32')
    X[0] = unpack('16L', F.astype('float32'))
    X[1] = unpack('16H', F.astype('float16'))
    X[2] = unpack('16H', F.astype('float16'))
    X[2] <<= 16
    X[3:7] = np.array([getrandbits(32) for i in range(4*16)]).reshape(4, 16)

    Y = runcode(unpack_rega_float, X)

    assert all(X[0] == Y[0])
    assert np.allclose(F, unpack('16f', Y[1]), rtol=1e-3)
    assert np.allclose(F, unpack('16f', Y[2]), rtol=1e-3)
    assert np.allclose(((X[3]>> 0)&0xff)/255.0, unpack('16f', Y[3]), rtol=1e-7)
    assert np.allclose(((X[4]>> 8)&0xff)/255.0, unpack('16f', Y[4]), rtol=1e-7)
    assert np.allclose(((X[5]>>16)&0xff)/255.0, unpack('16f', Y[5]), rtol=1e-7)
    assert np.allclose(((X[6]>>24)&0xff)/255.0, unpack('16f', Y[6]), rtol=1e-7)

#================================= R4 Unpack ==================================

@qpucode
def unpack_R4(asm):
    # Use TMU for loading data to R4.
    shl(r0, element_number, 2)
    iadd(r0, uniform, r0)       # r0[i] = tmu0_base + 4*i
    ldi(r1, 64)

    setup_vpm_write()

    for op in ['nop', '16a', '16b', '8a', '8b', '8c', '8d']:
        mov(tmu0_s, r0)
        nop(sig = 'load tmu0')
        fadd(vpm, r4.unpack(op), 0.0)
        iadd(r0, r0, r1) 

    setup_dma_store(nrows=7)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_unpack_R4():
    F = np.random.randn(16)

    X = np.zeros((7, 16), dtype='uint32')
    X[0] = unpack('16L', F.astype('float32'))
    X[1] = unpack('16H', F.astype('float16'))
    X[2] = unpack('16H', F.astype('float16'))
    X[2] <<= 16
    X[3:7] = np.array([getrandbits(32) for i in range(4*16)]).reshape(4, 16)

    with Driver() as drv:
        X = drv.copy(X)
        Y = drv.alloc((7, 16), dtype='uint32')
        drv.execute(
                num_threads=1,
                program=drv.program(unpack_R4),
                uniforms=[X.address, Y.address]
                )
        X = np.copy(X)
        Y = np.copy(Y)

    assert np.allclose(F, unpack('16f', Y[0]), rtol=1e-3)
    assert np.allclose(F, unpack('16f', Y[1]), rtol=1e-3)
    assert np.allclose(F, unpack('16f', Y[2]), rtol=1e-3)
    assert np.allclose(((X[3]>> 0)&0xff)/255.0, unpack('16f', Y[3]), rtol=1e-7)
    assert np.allclose(((X[4]>> 8)&0xff)/255.0, unpack('16f', Y[4]), rtol=1e-7)
    assert np.allclose(((X[5]>>16)&0xff)/255.0, unpack('16f', Y[5]), rtol=1e-7)
    assert np.allclose(((X[6]>>24)&0xff)/255.0, unpack('16f', Y[6]), rtol=1e-7)
