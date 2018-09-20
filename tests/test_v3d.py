'Test of accessing V3D registers'

import numpy as np
from random import getrandbits

from videocore.assembler import qpu
from videocore.driver import Driver
import videocore.v3d as v3d

@qpu
def qpu_code(asm):
    setup_dma_load(nrows=2)
    start_dma_load(uniform)
    wait_dma_load()
    setup_vpm_read(nrows=2)
    setup_vpm_write()

    mov(r0, vpm)
    mov(r1, vpm)
    iadd(r2, r0, r1)
    mov(vpm, r2)

    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

'''
# Bypass. Performance Counter value is not predictable
def test_performance_counter():
    with Driver() as drv:
        X = drv.alloc((2,16), dtype=np.int32)
        Y = drv.alloc(16, dtype=np.int32)
        X[:] = np.arange(2*16).reshape(2, 16).astype('int32')
        Y[:] = 0
        with v3d.RegisterMapping(drv) as regmap:
            with v3d.PerformanceCounter(regmap, [16,17,18,19,20,21,22,23,24,25,26,27,28,29]) as pc:
                drv.execute(
                    n_threads=1,
                    program=drv.program(qpu_code),
                    uniforms=[X.address, Y.address]
                )
                print(pc.result())
        assert(np.all(Y == [ 2*(i + 8) for i in range(16) ]))
'''
