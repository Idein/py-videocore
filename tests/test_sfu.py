'Test of special function unit'

import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def sfu(asm):
    setup_dma_load(nrows=1)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=1)
    setup_vpm_write()

    mov(r0, vpm)
    for unit in [sfu_recip, sfu_recipsqrt, sfu_exp2, sfu_log2]:
        mov(unit, r0)
        nop()
        nop()
        mov(vpm, r4)

    setup_dma_store(nrows=4)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_sfu():
    with Driver() as drv:
        X = np.random.uniform(0, 1, 16).astype('float32')
        X = drv.copy(X)
        Y = drv.alloc((4, 16), dtype='float32')
        drv.execute(
                num_threads=1,
                program=drv.program(sfu),
                uniforms=[X.address, Y.address]
                )

        assert np.allclose(1/X, Y[0], rtol=1e-3)
        assert np.allclose(1/np.sqrt(X), Y[1], rtol=1e-3)
        assert np.allclose(np.exp2(X), Y[2], rtol=1e-3)
        assert np.allclose(np.log2(X), Y[3], rtol=1e-3)
