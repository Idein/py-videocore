'Test of VPM read and write'

import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

#=================================== 32 bit ===================================

@qpu
def horizontal_32bit_read(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_vpm_read(Y=16, nrows=16)
    setup_vpm_write(Y=0)
    for i in range(16):
        mov(r0, vpm)
        mov(vpm, r0)

    setup_dma_store(mode='32bit horizontal', nrows=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_horizontal_32bit_read():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_read),
                uniforms=[X.address, Y.address]
                )

        assert np.all(X[16:32] == Y)

@qpu
def horizontal_32bit_write(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_vpm_read(Y=0, nrows=16)
    setup_vpm_write(Y=16)
    for i in range(16):
        mov(r0, vpm)
        mov(vpm, r0)

    setup_dma_store(Y=16, mode='32bit horizontal', nrows=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_horizontal_32bit_write():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_write),
                uniforms=[X.address, Y.address]
                )


        assert np.all(X[:16] == Y)

@qpu
def vertical_32bit_read(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_vpm_read(nrows=16, Y=16, X=0, mode='32bit vertical')
    setup_vpm_write(Y=0, X=0, mode='32bit vertical')
    for i in range(16):
        mov(r0, vpm)
        mov(vpm, r0)

    setup_dma_store(mode='32bit horizontal', nrows=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_vertical_32bit_read():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(vertical_32bit_read),
                uniforms=[X.address, Y.address]
                )

        assert np.all(X[16:32] == Y)

@qpu
def vertical_32bit_write(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_vpm_read(nrows=16, Y=0, X=0, mode='32bit vertical')
    setup_vpm_write(Y=16, X=0, mode='32bit vertical')
    for i in range(16):
        mov(r0, vpm)
        mov(vpm, r0)

    setup_dma_store(Y=16, mode='32bit horizontal', nrows=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_vertical_32bit_write():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(vertical_32bit_write),
                uniforms=[X.address, Y.address]
                )

        assert np.all(X[:16] == Y)
