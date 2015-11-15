from qpu.assembler import qpucode
from qpu.driver import Driver
import numpy as np

@qpucode
def vector_add(asm):
    setup_dma_load(mode = '32bit horizontal', Y = 0, X = 0, nrows = 2, ncols = 16)
    start_dma_load(uniform)
    wait_dma_load()

    setup_vpm_read(mode = '32bit horizontal', Y = 0, nrows = 2)
    setup_vpm_write(mode = '32bit horizontal', Y = 0)

    mov(r0, vpm)
    mov(r1, vpm)
    fadd(vpm, r0, r1)

    setup_dma_store(mode = '32bit horizontal', Y = 0, X = 0, nrows = 2, ncols = 16)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    prog = drv.program(vector_add)
    a   = np.random.random(16)
    b   = np.random.random(16)
    inp = drv.array((2, 16), 'float32')
    c   = drv.array(16, 'float32')

    inp[0] = a
    inp[1] = b

    unif = drv.array(2, 'int32')
    unif[0] = inp.address
    unif[1] = c.address

    prog(1, unif, timeout=1000)
