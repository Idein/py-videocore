'Test of pack/unpack functionality'

import numpy as np
from struct import unpack
from random import getrandbits

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def boilerplate(asm, f, nrows):
    #mov(rb0, uniform)

    setup_dma_load(nrows=nrows)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=nrows)
    setup_vpm_write()

    f(asm)

    setup_dma_store(nrows=nrows)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def run_code(code, X):
    with Driver() as drv:
        X = drv.copy(X)
        Y = drv.copy(X)
        drv.execute(
                n_threads=1,
                program=drv.program(boilerplate, code, X.shape[0]),
                uniforms=[X.address, Y.address]
                )
        return np.copy(Y)

#============================== Regfile-A Unpack ==============================

@qpu
def unpack_regA_int(asm):
    for op in ['nop', '16a', '16b', 'rep 8d', '8a', '8b', '8c', '8d']:
        mov(ra0, vpm)
        nop()
        mov(vpm, ra0.unpack(op))

def test_unpack_regA_int():
    X = np.array(
            [getrandbits(32) for i in range(8*16)]
            ).reshape(8, 16).astype('uint32')

    Y = run_code(unpack_regA_int, X)

    assert all(X[0] == Y[0])
    assert all(((X[1]>> 0)&0xffff).astype('int16') == Y[1].astype('int16'))
    assert all(((X[2]>>16)&0xffff).astype('int16') == Y[2].astype('int16'))
    assert all((((X[3]>>24)&0xff)*0x01010101) == Y[3])
    assert all(((X[4]>> 0)&0xff) == Y[4])
    assert all(((X[5]>> 8)&0xff) == Y[5])
    assert all(((X[6]>>16)&0xff) == Y[6])
    assert all(((X[7]>>24)&0xff) == Y[7])

@qpu
def unpack_regA_float(asm):
    for op in ['nop', '16a', '16b', '8a', '8b', '8c', '8d']:
        mov(ra0, vpm)
        nop()
        fmul(vpm, ra0.unpack(op), 1.0)
        nop()

def test_unpack_regA_float():
    F = np.random.randn(16)

    X = np.zeros((7, 16), dtype='uint32')
    X[0] = unpack('16L', F.astype('float32'))
    X[1] = unpack('16H', F.astype('float16'))
    X[2] = unpack('16H', F.astype('float16'))
    X[2] <<= 16
    X[3:7] = np.array([getrandbits(32) for i in range(4*16)]).reshape(4, 16)

    Y = run_code(unpack_regA_float, X)

    assert all(X[0] == Y[0])
    assert np.allclose(F, unpack('16f', Y[1]), rtol=1e-3)
    assert np.allclose(F, unpack('16f', Y[2]), rtol=1e-3)
    assert np.allclose(((X[3]>> 0)&0xff)/255.0, unpack('16f', Y[3]), rtol=1e-7)
    assert np.allclose(((X[4]>> 8)&0xff)/255.0, unpack('16f', Y[4]), rtol=1e-7)
    assert np.allclose(((X[5]>>16)&0xff)/255.0, unpack('16f', Y[5]), rtol=1e-7)
    assert np.allclose(((X[6]>>24)&0xff)/255.0, unpack('16f', Y[6]), rtol=1e-7)

#================================= R4 Unpack ==================================

@qpu
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
                n_threads=1,
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

#=============================== Regfile-A Pack ===============================

# packing without saturation

@qpu
def pack_regA_int_no_sat(asm):
    for op in ['nop', '16a', '16b', 'rep 8', '8a', '8b', '8c', '8d']:
        mov(ra0.pack(op), vpm)
        nop()
        mov(vpm, ra0)

def test_pack_regA_int_no_sat():
    X = np.array(
            [getrandbits(32) for i in range(8*16)]
            ).reshape(8, 16).astype('uint32')

    Y = run_code(pack_regA_int_no_sat, X)

    assert all(X[0] == Y[0])
    assert all(X[1]&0xffff == (Y[1]>> 0)&0xffff)
    assert all(X[2]&0xffff == (Y[2]>>16)&0xffff)
    assert all((X[3]&0xff)*0x01010101 == Y[3])
    assert all(X[4]&0xff == (Y[4]>> 0)&0xff)
    assert all(X[5]&0xff == (Y[5]>> 8)&0xff)
    assert all(X[6]&0xff == (Y[6]>>16)&0xff)
    assert all(X[7]&0xff == (Y[7]>>24)&0xff)

# packing with saturation

@qpu
def pack_regA_int_sat(asm):
    for op in ['32 sat', '16a sat', '16b sat', 'rep 8 sat', '8a sat',
               '8b sat', '8c sat', '8d sat']:
        mov(ra0, vpm)
        nop()
        iadd(ra0.pack(op), ra0, ra0)
        nop()
        mov(vpm, ra0)

def test_pack_regA_int_sat():
    X = np.zeros((8, 16), dtype='int32')
    X[0] = np.array([getrandbits(32) for i in range(16)]).astype('int32')
    X[1:3] = np.array(
            [getrandbits(16) for i in range(2*16)]
            ).reshape(2, 16).astype('int16')
    X[3:] = np.array(
            [getrandbits(8) for i in range(5*16)]
            ).reshape(5, 16).astype('int8')

    Y = run_code(pack_regA_int_sat, X)

    assert all(np.clip(2*X[0].astype('int64'),-2**31,2**31-1) == Y[0])
    assert all(np.clip(2*X[1],-2**15,2**15-1) == (Y[1]&0xffff).astype('int16'))
    assert all(np.clip(2*X[2],-2**15,2**15-1) == (Y[2]>>16).astype('int16'))
    assert all(np.clip(2*X[3].astype('int32'),0,2**8-1)*0x01010101 ==Y[3])
    assert all(np.clip(2*X[4].astype('int32'),0,2**8-1) == ((Y[4]>> 0)&0xff))
    assert all(np.clip(2*X[5].astype('int32'),0,2**8-1) == ((Y[5]>> 8)&0xff))
    assert all(np.clip(2*X[6].astype('int32'),0,2**8-1) == ((Y[6]>>16)&0xff))
    assert all(np.clip(2*X[7].astype('int32'),0,2**8-1) == ((Y[7]>>24)&0xff))

@qpu
def pack_regA_float(asm):
    for op in ['nop', '16a', '16b', '16a sat', '16b sat']:
        mov(ra0, vpm)
        nop()
        fadd(ra0.pack(op), ra0, 0.0)
        nop()
        mov(vpm, ra0)

def test_pack_regA_float():
    def to_float16(vec):
        return np.ndarray(16, 'float16', (vec&0xffff).astype('uint16'))

    F = np.random.randn(5, 16).astype('float32')
    X = np.ndarray((5, 16), 'uint32', F)

    Y = run_code(pack_regA_float, X)

    assert all(F[0] == np.ndarray(16, 'float32', Y[0]))
    assert np.allclose(F[1], to_float16(Y[1]>> 0), rtol=1e-3)
    assert np.allclose(F[2], to_float16(Y[2]>>16), rtol=1e-3)
    assert np.allclose(F[3], to_float16(Y[3]>> 0), rtol=1e-3)
    assert np.allclose(F[4], to_float16(Y[4]>>16), rtol=1e-3)

#================================ MUL ALU Pack ================================

@qpu
def pack_mul(asm):
    for op in ['rep 8', '8a', '8b', '8c', '8d']:
        mov(ra0, vpm)
        nop()
        fmul(ra0, ra0, 1.0, pack=op)
        nop()
        mov(vpm, ra0)

def test_pack_mul():
    def to_byte(vec):
        return np.clip(vec*255,0,255).astype('int32')
    X = np.random.randn(5, 16).astype('float32')*0.5 + 0.5

    Y = run_code(pack_mul, X)
    Y = np.ndarray((5, 16), 'int32', Y)

    assert all(np.abs(to_byte(X[0]) - ((Y[0]>> 0)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[0]) - ((Y[0]>> 8)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[0]) - ((Y[0]>>16)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[0]) - ((Y[0]>>24)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[1]) - ((Y[1]>> 0)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[2]) - ((Y[2]>> 8)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[3]) - ((Y[3]>>16)&0xff).astype('int32')) <= 1)
    assert all(np.abs(to_byte(X[4]) - ((Y[4]>>24)&0xff).astype('int32')) <= 1)
