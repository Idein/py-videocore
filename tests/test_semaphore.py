'Test of semaphore instruction'

import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def increment_thread(asm, nthreads):
    vpm_address = ra0
    thread_id = ra1
    i = r1

    COUNTER_LOCK = 0
    COMPLETED = 1

    mov(vpm_address, uniform)
    mov(thread_id, uniform, set_flags=True)
    
    jzc(L.skip_load)
    nop(); nop(); nop()

    setup_dma_load(nrows=1)
    start_dma_load(vpm_address)
    wait_dma_load()
    sema_up(COUNTER_LOCK)   # Release lock for counter.

    L.skip_load

    ldi(i, 10000)
    L.loop

    sema_down(COUNTER_LOCK)

    setup_vpm_read(nrows=1)
    setup_vpm_write()
    mov(r2, vpm)
    iadd(vpm, r2, 1)    # Increment the counter.

    sema_up(COUNTER_LOCK)

    isub(i, i, 1)
    jzc(L.loop)
    nop(); nop(); nop() 

    sema_up(COMPLETED)  # Notify completion to the thread 0.

    mov(null, thread_id, set_flags=True)
    jzc(L.skip_store)
    nop(); nop(); nop()

    for j in range(nthreads):
        sema_down(COMPLETED)    # Wait all threads complete. 

    setup_dma_store(nrows=1)
    start_dma_store(vpm_address)
    wait_dma_store()
    interrupt()

    L.skip_store
    exit(interrupt=False)


def test_semaphore():
    with Driver() as drv:
        nthreads = 10
        X = drv.alloc(16, dtype='uint32')
        X[:] = 0
        unifs = np.zeros((nthreads, 2), dtype='uint32')
        unifs[:, 0] = X.address
        unifs[:, 1] = np.arange(nthreads)
        drv.execute(
            num_threads=nthreads,
            program=drv.program(increment_thread, nthreads),
            uniforms=unifs
            )
        assert np.all(X == nthreads*10000)
