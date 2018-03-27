'Test of label scope and label exporter'
import numpy as np

from videocore.assembler import qpu, get_label_positions
from videocore.driver import Driver

@qpu
def given_jmp(asm):

    mov(ra0, uniform)

    mov(r0, 0)

    L.entry
    jmp(reg=ra0)
    nop()
    nop()
    nop()

    iadd(r0, r0, 1)

    L.test
    iadd(r0, r0, 4)
    setup_vpm_write()
    mov(vpm, r0)

    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_given_jump():
    lbls = get_label_positions(given_jmp)
    entry_pc = 0
    test_pc = 0
    for lbl, pc in lbls:
        if lbl.name == 'entry':
            entry_pc = pc
        if lbl.name == 'test':
            test_pc = pc
    with Driver() as drv:
        X = drv.alloc((1, 16), 'int32')
        X[:] = 1234
        drv.execute(
            n_threads=1,
            program=drv.program(given_jmp),
            uniforms=[test_pc-entry_pc-32, X.address]
        )
        assert np.all(X == 4)

@qpu
def with_namespace(asm):

    mov(r0, 0)

    with namespace('ns1'):
        jmp(L.test)
        nop()
        nop()
        nop()
        iadd(r0, r0, 10)
        L.test
        iadd(r0, r0, 1)

        with namespace('nested'):
            jmp(L.test)
            nop()
            nop()
            nop()
            iadd(r0, r0, 10)
            L.test
            iadd(r0, r0, 1)

    with namespace('ns2'):
        jmp(L.test)
        nop()
        nop()
        nop()
        iadd(r0, r0, 10)
        L.test
        iadd(r0, r0, 1)

    jmp(L.test)
    nop()
    nop()
    nop()
    iadd(r0, r0, 10)
    L.test
    iadd(r0, r0, 1)

    setup_vpm_write()
    mov(vpm, r0)

    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()
    exit()

def test_with_namespace():
    with Driver() as drv:
        X = drv.alloc((1, 16), 'int32')
        X[:] = 1234
        drv.execute(
            n_threads=1,
            program=drv.program(with_namespace),
            uniforms=[X.address]
        )
        assert np.all(X == 4)
