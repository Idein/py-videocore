'Test of DMA load and store'

import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

#=================================== 32 bit ===================================

@qpu
def horizontal_32bit_load(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_dma_store(mode='32bit horizontal', nrows=64)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_horizontal_32bit_load():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((64, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_load),
                uniforms=[X.address, Y.address]
                )
        assert np.all(X == Y)

@qpu
def horizontal_32bit_load_calc_and_store(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_vpm_read(mode='32bit horizontal', Y=0, X=0, nrows=1)
    setup_vpm_write(mode='32bit horizontal', Y=0, X=0)
    mov(r0, vpm)
    iadd(vpm, r0, 1)

    setup_dma_store(mode='32bit horizontal', nrows=64)
    start_dma_store(uniform)

    # TODO: remove this nop
    for i in range(16*1024):
        nop()

    wait_dma_store()
    exit()

def test_horizontal_32bit_load_calc_and_store_another_buffer():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((64, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_load_calc_and_store),
                uniforms=[X.address, Y.address]
                )

        X[0] = X[0] + 1
        assert np.all(X == Y)

def test_horizontal_32bit_load_calc_and_store_self_buffer():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((64, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_load_calc_and_store),
                uniforms=[X.address, X.address]
                )

        Y[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y[0] = Y[0] + 1
        assert np.all(X == Y)

@qpu
def vertical_32bit_load(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*4)
    for x in range(16):
        setup_dma_load(Y=0, X=x, mode='32bit vertical', nrows=4, vpitch=0)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    setup_dma_store(mode='32bit horizontal', nrows=64)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_vertical_32bit_load():
    with Driver() as drv:
        X = drv.alloc((16, 64), dtype='uint32')
        X[:] = np.arange(16*64).reshape(16, 64).astype('uint32')
        Y = drv.alloc((64, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(vertical_32bit_load),
                uniforms=[X.address, Y.address]
                )

        assert np.all(X == Y.T)

@qpu
def vertical_32bit_store(asm):
    mov(ra0, uniform)
    ldi(rb0, 4*16*16)
    for i in range(4):
        setup_dma_load(Y=16*i, mode='32bit horizontal', nrows=16)
        start_dma_load(ra0)
        iadd(ra0, ra0, rb0)
    wait_dma_load()

    mov(ra0, uniform)
    ldi(rb0, 4*16)
    for x in range(16):
        for y in range(0, 64, 16):
            setup_dma_store(Y=y, X=x, mode='32bit vertical', nrows=1, ncols=16)
            start_dma_store(ra0)
            iadd(ra0, ra0, rb0)
    wait_dma_store()
    exit()

def test_vertical_32bit_store():
    with Driver() as drv:
        X = drv.alloc((64, 16), dtype='uint32')
        X[:] = np.arange(64*16).reshape(64, 16).astype('uint32')
        Y = drv.alloc((16, 64), dtype='uint32')
        Y[:] = 0

        drv.execute(
                n_threads=1,
                program=drv.program(vertical_32bit_store),
                uniforms=[X.address, Y.address]
                )

        assert np.all(X == Y.T)

@qpu
def horizontal_32bit_partial(asm):
    setup_dma_load(X=4, Y=4, mode='32bit horizontal', nrows=8, ncols=8,
            mpitch=2)
    start_dma_load(uniform)
    wait_dma_load()

    setup_dma_store(mode='32bit horizontal', nrows=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_horizontal_32bit_partial():
    with Driver() as drv:
        X = drv.alloc((8, 8), dtype='uint32')
        X[:] = np.arange(8*8).reshape(8, 8).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_partial),
                uniforms=[X.address, Y.address]
                )
        assert np.all(X == Y[4:12, 4:12])

@qpu
def horizontal_32bit_stride_load(asm):
    setup_dma_load_stride(128)
    setup_dma_load(mode='32bit horizontal', nrows=16, ncols=16, mpitch=0)
    start_dma_load(uniform)
    wait_dma_load()

    setup_dma_store(mode='32bit horizontal', nrows=16, ncols=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()


def test_horizontal_32bit_stride_load():
    with Driver() as drv:
        X = drv.alloc((16, 32), dtype='uint32')
        X[:] = np.arange(16*32).reshape(16, 32).astype('uint32')
        Y = drv.alloc((16, 16), dtype='uint32')
        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_stride_load),
                uniforms=[X.address, Y.address]
                )
        assert np.all(X[:, :16] == Y)

@qpu
def horizontal_32bit_stride_store(asm):
    setup_dma_load(mode='32bit horizontal', nrows=16, ncols=16)
    start_dma_load(uniform)
    wait_dma_load()

    setup_dma_store_stride(64)
    setup_dma_store(mode='32bit horizontal', nrows=16, ncols=16)
    start_dma_store(uniform)
    wait_dma_store()
    exit()


def test_horizontal_32bit_stride_store():
    with Driver() as drv:
        X = drv.alloc((16, 16), dtype='uint32')
        X[:] = np.arange(16*16).reshape(16, 16).astype('uint32')
        Y = drv.alloc((16, 32), dtype='uint32')
        Y[:] = 0

        drv.execute(
                n_threads=1,
                program=drv.program(horizontal_32bit_stride_store),
                uniforms=[X.address, Y.address]
                )
        assert np.all(X == Y[:, :16])
