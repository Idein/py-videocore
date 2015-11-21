'Test of Branch instruction'

import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def boilerplate(asm, f, nout):
    setup_vpm_write()

    f(asm)

    setup_dma_store(nrows=nout)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def run_code(code, nout):
    with Driver() as drv:
        X = drv.alloc((nout, 16), 'uint32')
        drv.execute(
                num_threads=1,
                program=drv.program(boilerplate, code, nout),
                uniforms=[X.address]
                )
        return np.copy(X)

@qpu
def jmp_always(asm):
    mov(r0, 1)
    jmp(L._1)
    nop()
    nop()
    nop()

    mov(r0, 2)
    L._1
    mov(vpm, r0)

def test_jmp_always():
    X = run_code(jmp_always, 1)
    assert np.all(X == 1)
