# GPU accelerated single precision matrix multiplication (multi thread)
import numpy as np
import struct
import time

from videocore.assembler import qpu, assemble
from videocore.driver import Driver

def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values

@qpu
def sgemm_gpu_code(asm, n_threads):
    B_CUR_IDX = 0
    K_IDX = 1
    I_IDX = 2
    J_IDX = 3
    P_IDX = 4
    Q_IDX = 5
    R_IDX = 6
    A_CUR_IDX = 7
    C_CUR_IDX = 8
    A_BASE_IDX = 9
    B_BASE_IDX = 10
    C_BASE_IDX = 11
    A_STRIDE_IDX = 12
    B_STRIDE_IDX = 13
    C_STRIDE_IDX = 14
    COEF_ADDR_IDX = 15

    # Semaphore
    COMPLETED = 0

    #==== Load constants ====
    # Load constants to r2.
    mov(r0, uniform)    # uniforms address
    mov(r2, 1)
    ldi(null, mask(P_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # p/16
    ldi(null, mask(Q_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # q
    ldi(null, mask(R_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # r/64
    ldi(null, mask(A_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of A[0,0]
    ldi(null, mask(B_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of B[0,0]
    ldi(null, mask(C_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of C[0,0]
    ldi(null, mask(A_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # A stride
    ldi(null, mask(B_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # B stride
    ldi(null, mask(C_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # C stride
    ldi(null, mask(COEF_ADDR_IDX), set_flags=True)
    ldi(r1, 4*10)
    iadd(r2, r0, r1, cond='zs')     # address of alpha, beta and thread index

    #==== Semaphores ===
    nop()
    rotate(broadcast, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r5)
    nop(); nop()
    mov(null, uniform)
    mov(null, uniform)
    mov(null, uniform, set_flags=True)  # thread index
    
    jzc(L.skip_init)
    nop(); nop(); nop()

    L.skip_init


    #==== Variables ====

    # A_base = address of A[0,0] + 16*p*A_stride
    # B_base = address of B[0,0] + 4*64*r
    # C_base = address of C[0,0] + 16*p*C_stride + 4*64*r

    # A_cur = A_base - 16*i*A_stride
    # B_cur = B_base - 4*64*j
    # C_cur = C_base - 16*i*C_stride - 4*64*j

    rotate(broadcast, r2, -P_IDX)
    shl(r0, r5, 4)                  # r0=16*p
    rotate(broadcast, r2, -R_IDX)
    shl(r1, r5, 8)                  # r1=4*64*r
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r3, r5, r0)              # r3=16*p*A_stride
    ldi(null, mask(A_BASE_IDX), set_flags=True)
    iadd(r2, r2, r3, cond='zs')
    ldi(null, mask(B_BASE_IDX), set_flags=True)
    iadd(r2, r2, r1, cond='zs')
    rotate(broadcast, r2, -C_STRIDE_IDX)
    imul24(r3, r5, r0)              # r3=16*p*C_stride
    ldi(null, mask(C_BASE_IDX), set_flags=True)
    iadd(r2, r2, r3, cond='zs', set_flags=False)
    iadd(r2, r2, r1, cond='zs')

    # Disable swapping of two TMUs.
    mov(tmu_noswap, 1)

    # Initialize column vectors.
    mov(ra0,  0.0).mov(rb0,  0.0)
    mov(ra1,  0.0).mov(rb1,  0.0)
    mov(ra2,  0.0).mov(rb2,  0.0)
    mov(ra3,  0.0).mov(rb3,  0.0)
    mov(ra4,  0.0).mov(rb4,  0.0)
    mov(ra5,  0.0).mov(rb5,  0.0)
    mov(ra6,  0.0).mov(rb6,  0.0)
    mov(ra7,  0.0).mov(rb7,  0.0)
    mov(ra8,  0.0).mov(rb8,  0.0)
    mov(ra9,  0.0).mov(rb9,  0.0)
    mov(ra10, 0.0).mov(rb10, 0.0)
    mov(ra11, 0.0).mov(rb11, 0.0)
    mov(ra12, 0.0).mov(rb12, 0.0)
    mov(ra13, 0.0).mov(rb13, 0.0)
    mov(ra14, 0.0).mov(rb14, 0.0)
    mov(ra15, 0.0).mov(rb15, 0.0)
    mov(ra16, 0.0).mov(rb16, 0.0)
    mov(ra17, 0.0).mov(rb17, 0.0)
    mov(ra18, 0.0).mov(rb18, 0.0)
    mov(ra19, 0.0).mov(rb19, 0.0)
    mov(ra20, 0.0).mov(rb20, 0.0)
    mov(ra21, 0.0).mov(rb21, 0.0)
    mov(ra22, 0.0).mov(rb22, 0.0)
    mov(ra23, 0.0).mov(rb23, 0.0)
    mov(ra24, 0.0).mov(rb24, 0.0)
    mov(ra25, 0.0).mov(rb25, 0.0)
    mov(ra26, 0.0).mov(rb26, 0.0)
    mov(ra27, 0.0).mov(rb27, 0.0)
    mov(ra28, 0.0).mov(rb28, 0.0)
    mov(ra29, 0.0).mov(rb29, 0.0)
    mov(ra30, 0.0).mov(rb30, 0.0)
    mov(ra31, 0.0).mov(rb31, 0.0)

    #==== i-loop ====

    # Initialize i.
    # i=p.
    rotate(broadcast, r2, -P_IDX)
    ldi(null, mask(I_IDX), set_flags=True)
    mov(r2, r5, cond='zs')

    L.i_loop

    #==== j-loop ====
    
    # Initialize j.
    # j=r.
    rotate(broadcast, r2, -R_IDX)
    ldi(null, mask(J_IDX), set_flags=True)
    mov(r2, r5, cond='zs')

    rotate(broadcast, r2, -I_IDX)
    shl(r0, r5, 4)                          # r0=16*i
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r0, r0, r5)                      # r0=16*i*A_stride
    rotate(broadcast, r2, -A_BASE_IDX)
    ldi(null, mask(A_CUR_IDX), set_flags=True)
    isub(r2, r5, r0, cond='zs')

    L.j_loop

    rotate(broadcast, r2, -I_IDX)
    shl(r0, r5, 4)                          # r0=16*i
    rotate(broadcast, r2, -C_STRIDE_IDX)
    imul24(r0, r0, r5)                      # r0=16*i*C_stride
    rotate(broadcast, r2, -J_IDX)
    shl(r1, r5, 8)                          # r1=4*64*j
    rotate(broadcast, r2, -C_BASE_IDX)
    ldi(null, mask(C_CUR_IDX), set_flags=True)
    isub(r2, r5, r0, cond='zs', set_flags=False)
    isub(r2, r2, r1, cond='zs')

    rotate(broadcast, r2, -B_BASE_IDX)
    ldi(null, mask(B_CUR_IDX), set_flags=True)
    isub(r2, r5, r1, cond='zs')

    # r1[e] = A_cur + A_stride*e   (e=element number)
    nop()
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r0, element_number, r5)
    rotate(broadcast, r2, -A_CUR_IDX)
    iadd(r1, r0, r5)

    # Initialize loop delta.
    # r3[0] = B_stride
    # r3[1] = -1
    # r3[other] = 0

    mov(r3, 0)
    rotate(broadcast, r2, -B_STRIDE_IDX)
    ldi(null, mask(B_CUR_IDX), set_flags=True)
    mov(r3, r5, cond='zs')
    ldi(null, mask(K_IDX), set_flags=True)
    mov(r3, -1, cond='zs')

    #==== k-loop ==== 
    # r2[1] = q (k=q)
    nop()
    rotate(broadcast, r2, -Q_IDX)
    ldi(null, mask(K_IDX), set_flags=True)
    mov(r2, r5, cond='zs')

    mov(uniforms_address, r2)
    mov(tmu0_s, r1)
    iadd(r1, r1, 4)
    nop(sig='load tmu0')

    iadd(r2, r2, r3).mov(tmu0_s, r1)
    iadd(r1, r1, 4).fmul(r0, r4, uniform)
    fadd(ra0,  ra0,  r0).fmul(r0, r4, uniform)
    fadd(rb0,  rb0,  r0).fmul(r0, r4, uniform)

    L.k_loop

    fadd(ra1,  ra1,  r0).fmul(r0, r4, uniform)
    fadd(rb1,  rb1,  r0).fmul(r0, r4, uniform)
    fadd(ra2,  ra2,  r0).fmul(r0, r4, uniform)
    fadd(rb2,  rb2,  r0).fmul(r0, r4, uniform)
    fadd(ra3,  ra3,  r0).fmul(r0, r4, uniform)
    fadd(rb3,  rb3,  r0).fmul(r0, r4, uniform)
    fadd(ra4,  ra4,  r0).fmul(r0, r4, uniform)
    fadd(rb4,  rb4,  r0).fmul(r0, r4, uniform)
    fadd(ra5,  ra5,  r0).fmul(r0, r4, uniform)
    fadd(rb5,  rb5,  r0).fmul(r0, r4, uniform)
    fadd(ra6,  ra6,  r0).fmul(r0, r4, uniform)
    fadd(rb6,  rb6,  r0).fmul(r0, r4, uniform)
    fadd(ra7,  ra7,  r0).fmul(r0, r4, uniform)
    fadd(rb7,  rb7,  r0).fmul(r0, r4, uniform)
    fadd(ra8,  ra8,  r0).fmul(r0, r4, uniform)
    fadd(rb8,  rb8,  r0).fmul(r0, r4, uniform)
    fadd(ra9,  ra9,  r0).fmul(r0, r4, uniform)
    fadd(rb9,  rb9,  r0).fmul(r0, r4, uniform)
    fadd(ra10, ra10, r0).fmul(r0, r4, uniform)
    fadd(rb10, rb10, r0).fmul(r0, r4, uniform)
    fadd(ra11, ra11, r0).fmul(r0, r4, uniform)
    fadd(rb11, rb11, r0).fmul(r0, r4, uniform)
    fadd(ra12, ra12, r0).fmul(r0, r4, uniform)
    fadd(rb12, rb12, r0).fmul(r0, r4, uniform)
    fadd(ra13, ra13, r0).fmul(r0, r4, uniform)
    fadd(rb13, rb13, r0).fmul(r0, r4, uniform)
    fadd(ra14, ra14, r0).fmul(r0, r4, uniform)
    fadd(rb14, rb14, r0).fmul(r0, r4, uniform)
    fadd(ra15, ra15, r0).fmul(r0, r4, uniform)
    fadd(rb15, rb15, r0).fmul(r0, r4, uniform)
    fadd(ra16, ra16, r0).fmul(r0, r4, uniform)
    fadd(rb16, rb16, r0).fmul(r0, r4, uniform)
    fadd(ra17, ra17, r0).fmul(r0, r4, uniform)
    fadd(rb17, rb17, r0).fmul(r0, r4, uniform)
    fadd(ra18, ra18, r0).fmul(r0, r4, uniform)
    fadd(rb18, rb18, r0).fmul(r0, r4, uniform)
    fadd(ra19, ra19, r0).fmul(r0, r4, uniform)
    fadd(rb19, rb19, r0).fmul(r0, r4, uniform)
    fadd(ra20, ra20, r0).fmul(r0, r4, uniform)
    fadd(rb20, rb20, r0).fmul(r0, r4, uniform)
    fadd(ra21, ra21, r0).fmul(r0, r4, uniform)
    fadd(rb21, rb21, r0).fmul(r0, r4, uniform)
    fadd(ra22, ra22, r0).fmul(r0, r4, uniform)
    fadd(rb22, rb22, r0).fmul(r0, r4, uniform)
    fadd(ra23, ra23, r0).fmul(r0, r4, uniform)
    fadd(rb23, rb23, r0).fmul(r0, r4, uniform)
    fadd(ra24, ra24, r0).fmul(r0, r4, uniform)
    fadd(rb24, rb24, r0).fmul(r0, r4, uniform)
    fadd(ra25, ra25, r0).fmul(r0, r4, uniform)
    fadd(rb25, rb25, r0).fmul(r0, r4, uniform)
    fadd(ra26, ra26, r0).fmul(r0, r4, uniform)
    fadd(rb26, rb26, r0).fmul(r0, r4, uniform)
    fadd(ra27, ra27, r0).fmul(r0, r4, uniform)
    fadd(rb27, rb27, r0).fmul(r0, r4, uniform)
    fadd(ra28, ra28, r0).fmul(r0, r4, uniform)
    fadd(rb28, rb28, r0).fmul(r0, r4, uniform)
    fadd(ra29, ra29, r0).fmul(r0, r4, uniform)
    fadd(rb29, rb29, r0).fmul(r0, r4, uniform)
    fadd(ra30, ra30, r0).fmul(r0, r4, uniform)
    fadd(rb30, rb30, r0).fmul(r0, r4, uniform)
    fadd(ra31, ra31, r0).fmul(r0, r4, uniform)
    fadd(rb31, rb31, r0, sig='load tmu0').mov(uniforms_address, r2)
    iadd(r2, r2, r3).mov(tmu0_s, r1)
    jzc(L.k_loop)
    iadd(r1, r1, 4).fmul(r0, r4, uniform)      # delay slot
    fadd(ra0,  ra0,  r0).fmul(r0, r4, uniform) # delay slot
    fadd(rb0,  rb0,  r0).fmul(r0, r4, uniform) # delay slot

    #==== end of k-loop ====

    # Emit load tmu0 signal for the last write to tmu0_s
    mov(r1, r4, sig='load tmu0')

    fadd(ra1,  ra1,  r0).fmul(r0, r1, uniform)
    fadd(rb1,  rb1,  r0).fmul(r0, r1, uniform)
    fadd(ra2,  ra2,  r0).fmul(r0, r1, uniform)
    fadd(rb2,  rb2,  r0).fmul(r0, r1, uniform)
    fadd(ra3,  ra3,  r0).fmul(r0, r1, uniform)
    fadd(rb3,  rb3,  r0).fmul(r0, r1, uniform)
    fadd(ra4,  ra4,  r0).fmul(r0, r1, uniform)
    fadd(rb4,  rb4,  r0).fmul(r0, r1, uniform)
    fadd(ra5,  ra5,  r0).fmul(r0, r1, uniform)
    fadd(rb5,  rb5,  r0).fmul(r0, r1, uniform)
    fadd(ra6,  ra6,  r0).fmul(r0, r1, uniform)
    fadd(rb6,  rb6,  r0).fmul(r0, r1, uniform)
    fadd(ra7,  ra7,  r0).fmul(r0, r1, uniform)
    fadd(rb7,  rb7,  r0).fmul(r0, r1, uniform)
    fadd(ra8,  ra8,  r0).fmul(r0, r1, uniform)
    fadd(rb8,  rb8,  r0).fmul(r0, r1, uniform)
    fadd(ra9,  ra9,  r0).fmul(r0, r1, uniform)
    fadd(rb9,  rb9,  r0).fmul(r0, r1, uniform)
    fadd(ra10, ra10, r0).fmul(r0, r1, uniform)
    fadd(rb10, rb10, r0).fmul(r0, r1, uniform)
    fadd(ra11, ra11, r0).fmul(r0, r1, uniform)
    fadd(rb11, rb11, r0).fmul(r0, r1, uniform)
    fadd(ra12, ra12, r0).fmul(r0, r1, uniform)
    fadd(rb12, rb12, r0).fmul(r0, r1, uniform)
    fadd(ra13, ra13, r0).fmul(r0, r1, uniform)
    fadd(rb13, rb13, r0).fmul(r0, r1, uniform)
    fadd(ra14, ra14, r0).fmul(r0, r1, uniform)
    fadd(rb14, rb14, r0).fmul(r0, r1, uniform)
    fadd(ra15, ra15, r0).fmul(r0, r1, uniform)
    fadd(rb15, rb15, r0).fmul(r0, r1, uniform)
    fadd(ra16, ra16, r0).fmul(r0, r1, uniform)
    fadd(rb16, rb16, r0).fmul(r0, r1, uniform)
    fadd(ra17, ra17, r0).fmul(r0, r1, uniform)
    fadd(rb17, rb17, r0).fmul(r0, r1, uniform)
    fadd(ra18, ra18, r0).fmul(r0, r1, uniform)
    fadd(rb18, rb18, r0).fmul(r0, r1, uniform)
    fadd(ra19, ra19, r0).fmul(r0, r1, uniform)
    fadd(rb19, rb19, r0).fmul(r0, r1, uniform)
    fadd(ra20, ra20, r0).fmul(r0, r1, uniform)
    fadd(rb20, rb20, r0).fmul(r0, r1, uniform)
    fadd(ra21, ra21, r0).fmul(r0, r1, uniform)
    fadd(rb21, rb21, r0).fmul(r0, r1, uniform)
    fadd(ra22, ra22, r0).fmul(r0, r1, uniform)
    fadd(rb22, rb22, r0).fmul(r0, r1, uniform)
    fadd(ra23, ra23, r0).fmul(r0, r1, uniform)
    fadd(rb23, rb23, r0).fmul(r0, r1, uniform)
    fadd(ra24, ra24, r0).fmul(r0, r1, uniform)
    fadd(rb24, rb24, r0).fmul(r0, r1, uniform)
    fadd(ra25, ra25, r0).fmul(r0, r1, uniform)
    fadd(rb25, rb25, r0).fmul(r0, r1, uniform)
    fadd(ra26, ra26, r0).fmul(r0, r1, uniform)
    fadd(rb26, rb26, r0).fmul(r0, r1, uniform)
    fadd(ra27, ra27, r0).fmul(r0, r1, uniform)
    fadd(rb27, rb27, r0).fmul(r0, r1, uniform)
    fadd(ra28, ra28, r0).fmul(r0, r1, uniform)
    fadd(rb28, rb28, r0).fmul(r0, r1, uniform)
    fadd(ra29, ra29, r0).fmul(r0, r1, uniform)
    fadd(rb29, rb29, r0).fmul(r0, r1, uniform)
    fadd(ra30, ra30, r0).fmul(r0, r1, uniform)
    fadd(rb30, rb30, r0).fmul(r0, r1, uniform)
    fadd(ra31, ra31, r0).fmul(r0, r1, uniform)
    fadd(rb31, rb31, r0)

    mutex_acquire()

    # Configure stride.
    rotate(broadcast, r2, -C_STRIDE_IDX)
    setup_dma_load_stride(r5, tmp_reg=r3)
    rotate(broadcast, r2, -C_STRIDE_IDX)
    ldi(r3, 4*16)
    isub(broadcast, r5, r3)
    setup_dma_store_stride(r5, tmp_reg=r3)

    # Issue load of block 0
    setup_dma_load(mode='32bit horizontal', Y=0, nrows=16, mpitch=0)
    rotate(broadcast, r2, -C_CUR_IDX)
    start_dma_load(r5)
    mov(r3, r5)

    # Issue load of block 1
    setup_dma_load(mode='32bit horizontal', Y=16, X=0, nrows=16, mpitch=0)
    ldi(broadcast, 4*16)
    iadd(vpm_ld_addr, r3, r5)

    # Issue load of block 2
    setup_dma_load(mode='32bit horizontal', Y=32, X=0, nrows=16, mpitch=0)
    ldi(r0, 4*16*2)
    iadd(vpm_ld_addr, r3, r0)

    # Issue load of block 3
    setup_dma_load(mode='32bit horizontal', Y=48, X=0, nrows=16, mpitch=0)
    ldi(r0, 4*16*3)
    iadd(vpm_ld_addr, r3, r0)

    # Load alpha and beta.
    rotate(r0, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r0)

    # Setup VPM access for block 0
    setup_vpm_read(mode='32bit vertical', Y=0, X=0, nrows=16)
    setup_vpm_write(mode='32bit vertical', Y=0, X=0)

    mov(r1, uniform)        # r1=alpha
    mov(broadcast, uniform) # r5=beta

    fmul(ra0, ra0, r1)
    fmul(r0, vpm, r5)
    fadd(vpm, ra0, r0).fmul(rb0, rb0, r1)
    mov(ra0, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb0, r0).fmul(ra1, ra1, r1)
    mov(rb0, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra1, r0).fmul(rb1, rb1, r1)
    mov(ra1, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb1, r0).fmul(ra2, ra2, r1)
    mov(rb1, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra2, r0).fmul(rb2, rb2, r1)
    mov(ra2, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb2, r0).fmul(ra3, ra3, r1)
    mov(rb2, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra3, r0).fmul(rb3, rb3, r1)
    mov(ra3, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb3, r0).fmul(ra4, ra4, r1)
    mov(rb3, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra4, r0).fmul(rb4, rb4, r1)
    mov(ra4, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb4, r0).fmul(ra5, ra5, r1)
    mov(rb4, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra5, r0).fmul(rb5, rb5, r1)
    mov(ra5, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb5, r0).fmul(ra6, ra6, r1)
    mov(rb5, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra6, r0).fmul(rb6, rb6, r1)
    mov(ra6, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb6, r0).fmul(ra7, ra7, r1)
    mov(rb6, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra7, r0).fmul(rb7, rb7, r1)
    mov(ra7, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb7, r0)
    mov(rb7, 0.0)

    # Issue store of block 0
    setup_dma_store(mode='32bit horizontal', Y=0, nrows=16)
    start_dma_store(r3)

    # Setup VPM access for block 1
    setup_vpm_read(mode='32bit vertical', Y=16, X=0, nrows=16)
    setup_vpm_write(mode='32bit vertical', Y=16, X=0)

    fmul(ra8, ra8, r1)
    fmul(r0, vpm, r5)
    fadd(vpm, ra8, r0).fmul(rb8, rb8, r1)
    mov(ra8, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb8, r0).fmul(ra9, ra9, r1)
    mov(rb8, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra9, r0).fmul(rb9, rb9, r1)
    mov(ra9, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb9, r0).fmul(ra10, ra10, r1)
    mov(rb9, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra10, r0).fmul(rb10, rb10, r1)
    mov(ra10, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb10, r0).fmul(ra11, ra11, r1)
    mov(rb10, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra11, r0).fmul(rb11, rb11, r1)
    mov(ra11, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb11, r0).fmul(ra12, ra12, r1)
    mov(rb11, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra12, r0).fmul(rb12, rb12, r1)
    mov(ra12, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb12, r0).fmul(ra13, ra13, r1)
    mov(rb12, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra13, r0).fmul(rb13, rb13, r1)
    mov(ra13, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb13, r0).fmul(ra14, ra14, r1)
    mov(rb13, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra14, r0).fmul(rb14, rb14, r1)
    mov(ra14, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb14, r0).fmul(ra15, ra15, r1)
    mov(rb14, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra15, r0).fmul(rb15, rb15, r1)
    mov(ra15, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb15, r0)
    mov(rb15, 0.0)

    # Issue store of block 1
    setup_dma_store(mode='32bit horizontal', Y=16, nrows=16)
    ldi(r0, 4*16)
    iadd(vpm_st_addr, r3, r0)

    wait_dma_load() # block 2 and 3

    # setup VPM access for block 2.
    setup_vpm_read(mode='32bit vertical', X=0, Y=32, nrows=16)
    setup_vpm_write(mode='32bit vertical', X=0, Y=32)

    fmul(ra16, ra16, r1)
    fmul(r0, vpm, r5)
    fadd(vpm, ra16, r0).fmul(rb16, rb16, r1)
    mov(ra16, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb16, r0).fmul(ra17, ra17, r1)
    mov(rb16, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra17, r0).fmul(rb17, rb17, r1)
    mov(ra17, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb17, r0).fmul(ra18, ra18, r1)
    mov(rb17, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra18, r0).fmul(rb18, rb18, r1)
    mov(ra18, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb18, r0).fmul(ra19, ra19, r1)
    mov(rb18, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra19, r0).fmul(rb19, rb19, r1)
    mov(ra19, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb19, r0).fmul(ra20, ra20, r1)
    mov(rb19, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra20, r0).fmul(rb20, rb20, r1)
    mov(ra20, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb20, r0).fmul(ra21, ra21, r1)
    mov(rb20, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra21, r0).fmul(rb21, rb21, r1)
    mov(ra21, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb21, r0).fmul(ra22, ra22, r1)
    mov(rb21, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra22, r0).fmul(rb22, rb22, r1)
    mov(ra22, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb22, r0).fmul(ra23, ra23, r1)
    mov(rb22, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra23, r0).fmul(rb23, rb23, r1)
    mov(ra23, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb23, r0)
    mov(rb23, 0.0)

    # Issue store of block 2. 
    setup_dma_store(mode='32bit horizontal', Y=32, nrows=16)
    ldi(r0, 4*16*2)
    iadd(vpm_st_addr, r3, r0)

    # setup VPM access for block 3
    setup_vpm_read(mode='32bit vertical', X=0, Y=48, nrows=16)
    setup_vpm_write(mode='32bit vertical', X=0, Y=48)

    fmul(ra24, ra24, r1)
    fmul(r0, vpm, r5)
    fadd(vpm, ra24, r0).fmul(rb24, rb24, r1)
    mov(ra24, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb24, r0).fmul(ra25, ra25, r1)
    mov(rb24, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra25, r0).fmul(rb25, rb25, r1)
    mov(ra25, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb25, r0).fmul(ra26, ra26, r1)
    mov(rb25, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra26, r0).fmul(rb26, rb26, r1)
    mov(ra26, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb26, r0).fmul(ra27, ra27, r1)
    mov(rb26, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra27, r0).fmul(rb27, rb27, r1)
    mov(ra27, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb27, r0).fmul(ra28, ra28, r1)
    mov(rb27, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra28, r0).fmul(rb28, rb28, r1)
    mov(ra28, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb28, r0).fmul(ra29, ra29, r1)
    mov(rb28, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra29, r0).fmul(rb29, rb29, r1)
    mov(ra29, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb29, r0).fmul(ra30, ra30, r1)
    mov(rb29, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra30, r0).fmul(rb30, rb30, r1)
    mov(ra30, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb30, r0).fmul(ra31, ra31, r1)
    mov(rb30, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra31, r0).fmul(rb31, rb31, r1)
    mov(ra31, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb31, r0)
    mov(rb31, 0.0)

    # Issue store of block 3
    setup_dma_store(mode='32bit horizontal', Y=48, nrows=16)
    ldi(r0, 4*16*3)
    iadd(vpm_st_addr, r3, r0)

    wait_dma_store()
    mutex_release()

    rotate(broadcast, r2, -J_IDX)
    isub(r0, r5, 1)
    jzc(L.j_loop)   # Jump iz Z-flags are clear
    ldi(null, mask(J_IDX), set_flags=True)  # delay slot
    mov(r2, r0, cond='zs')                  # delay slot
    nop()                                   # delay slot

    rotate(broadcast, r2, -I_IDX)
    isub(r0, r5, 1)
    jzc(L.i_loop)
    ldi(null, mask(I_IDX), set_flags=True)  # delay slot
    mov(r2, r0, cond='zs')                  # delay slot
    nop()                                   # delay slot

    sema_up(COMPLETED)  # Notify completion to the thread 0

    rotate(broadcast, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r5)
    nop(); nop()
    mov(null, uniform)
    mov(null, uniform)
    mov(null, uniform, set_flags=True)  # thread index
    
    jzc(L.skip_fin)
    nop(); nop(); nop()

    # Only thread 0 enters here.
    for i in range(n_threads):
        sema_down(COMPLETED)    # Wait completion of all threads.

    interrupt()

    L.skip_fin

    exit(interrupt=False)

if __name__ == '__main__':
    with Driver() as drv:
        p = 96
        q = 363
        r = 3072

        p_div = 2
        r_div = 6
        n_threads = p_div * r_div

        assert(p%16 == 0 and p >= p_div*16)
        assert(q >= 2)
        assert(r%64 == 0 and r >= r_div*64)

        # Allocate matrices.
        C = drv.alloc((p, r), 'float32')
        A = drv.alloc((p, q), 'float32')
        B = drv.alloc((q, r), 'float32')

        # Initialize matrices.
        np.random.seed(0)
        alpha = 1.0
        beta = 1.0
        A[:] = np.random.randn(p, q)
        B[:] = np.random.randn(q, r)
        C[:] = np.random.randn(p, r)

        # Reference
        start = time.time()
        R = alpha*A.dot(B) + beta*C
        elapsed_ref = time.time() - start

        # Allocate uniforms.
        uniforms = drv.alloc((n_threads, 13), 'uint32')
        uniforms[:, 0] = uniforms.addresses()[:, 0]

        th = 0
        h = p/(16*p_div)
        w = r/(64*r_div)
        for i in range(p_div):
            for j in range(r_div):
                uniforms[th, 1] = h if i != p_div-1 else p/16-i*h
                uniforms[th, 2] = q
                uniforms[th, 3] = w if j != r_div-1 else r/64-j*w
                uniforms[th, 4] = A.addresses()[i*16*h, 0]
                uniforms[th, 5] = B.addresses()[0, j*64*w]
                uniforms[th, 6] = C.addresses()[i*16*h, j*64*w]
                th += 1
        uniforms[:, 7] = A.strides[0]
        uniforms[:, 8] = B.strides[0]
        uniforms[:, 9] = C.strides[0]
        uniforms[:, 10] = struct.unpack('L', struct.pack('f', alpha))[0]
        uniforms[:, 11] = struct.unpack('L', struct.pack('f', beta))[0]
        uniforms[:, 12] = np.arange(n_threads)

        # Allocate GPU program.
        code = drv.program(sgemm_gpu_code, n_threads)

        # GPU
        start = time.time()
        drv.execute(
                n_threads=n_threads,
                program=code,
                uniforms=uniforms
                )
        elapsed_gpu = time.time() - start

        def Gflops(sec):
            return (2*p*q*r + 3*p*r)/sec * 1e-9

        print('==== sgemm example ({p}x{q} times {q}x{r}) ===='.format(
                p=p, q=q, r=r))
        print('threads: {}'.format(n_threads))
        print('numpy: {:.4f} sec, {:.4f} Gflops'.format(
                elapsed_ref, Gflops(elapsed_ref)))
        print('GPU: {:.4f} sec, {:.4f} Gflops'.format(
                elapsed_gpu, Gflops(elapsed_gpu)))
        print('maximum absolute error: {:.4e}'.format(
                float(np.max(np.abs(R - C)))))
