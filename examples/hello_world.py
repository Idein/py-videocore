import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def hello_world(asm):
    # Load two vectors of length 16 from the host memory (address=uniforms[0]) to VPM
    setup_dma_load(nrows=2)
    start_dma_load(uniform)
    wait_dma_load()

    # Setup VPM read/write operaitons
    setup_vpm_read(nrows=2)
    setup_vpm_write()

    # Compute a + b
    mov(r0, vpm)
    mov(r1, vpm)
    fadd(vpm, r0, r1)

    # Store the result vector from VPM to the host memory (address=uniforms[1])
    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()

    # Finish the thread
    exit()

with Driver() as drv:
    # Input vectors
    a = np.random.random(16).astype('float32')
    b = np.random.random(16).astype('float32')

    # Copy vectors to shared memory for DMA transfer
    inp = drv.copy(np.r_[a, b])
    out = drv.alloc(16, 'float32')

    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(hello_world),
            uniforms=[inp.address, out.address]
            )

    print ' a '.center(80, '=')
    print(a)
    print ' b '.center(80, '=')
    print(b)
    print ' a+b '.center(80, '=')
    print(out)
    print ' error '.center(80, '=')
    print(np.abs(a+b-out))

