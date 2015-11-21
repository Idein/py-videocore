'Test of Branch instruction'

import inspect
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
        X = drv.alloc((nout, 16), 'int32')
        drv.execute(
                num_threads=1,
                program=drv.program(boilerplate, code, nout),
                uniforms=[X.address]
                )
        return np.copy(X)

ASSERT_OK = 1
ASSERT_NG = -1

LABEL_COUNTER = 0
def fresh_label():
    global LABEL_COUNTER
    LABEL_COUNTER += 1
    return '_' + str(LABEL_COUNTER)

@qpu
def assert_jump(asm, do_jump, br_insn):
    label = fresh_label()

    mov(ra0, [ASSERT_NG, ASSERT_OK][do_jump])

    br_insn(L[label])
    nop()   # delay slots
    nop()
    nop()

    mov(ra0, [ASSERT_OK, ASSERT_NG][do_jump])
    nop()

    L[label]

    mov(vpm, ra0)

def jmp_code_1(asm):
    ldi(r0, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    ldi(r1, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    ldi(r2, [1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,1])
    ldi(r3, [1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1])

    assert_jump(asm, 1, jmp)

    isub(null, r0, r1)  # no Z, no N, no C
    assert_jump(asm, 0, jzs)
    assert_jump(asm, 1, jzc)
    assert_jump(asm, 0, jzs_any)
    assert_jump(asm, 1, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 1, jnc)
    assert_jump(asm, 0, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 0, jcs)
    assert_jump(asm, 1, jcc)
    assert_jump(asm, 0, jcs_any)
    assert_jump(asm, 1, jcc_any)

    isub(null, r0, r0)  # all Z, no N, no C
    assert_jump(asm, 1, jzs)
    assert_jump(asm, 0, jzc)
    assert_jump(asm, 1, jzs_any)
    assert_jump(asm, 0, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 1, jnc)
    assert_jump(asm, 0, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 0, jcs)
    assert_jump(asm, 1, jcc)
    assert_jump(asm, 0, jcs_any)
    assert_jump(asm, 1, jcc_any)

    isub(null, r1, r0)  # no Z, all N, all C
    assert_jump(asm, 0, jzs)
    assert_jump(asm, 1, jzc)
    assert_jump(asm, 0, jzs_any)
    assert_jump(asm, 1, jzc_any)
    assert_jump(asm, 1, jns)
    assert_jump(asm, 0, jnc)
    assert_jump(asm, 1, jns_any)
    assert_jump(asm, 0, jnc_any)
    assert_jump(asm, 1, jcs)
    assert_jump(asm, 0, jcc)
    assert_jump(asm, 1, jcs_any)
    assert_jump(asm, 0, jcc_any)

    isub(null, r0, r2)  # some Z, no N, no C
    assert_jump(asm, 0, jzs)
    assert_jump(asm, 0, jzc)
    assert_jump(asm, 1, jzs_any)
    assert_jump(asm, 1, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 1, jnc)
    assert_jump(asm, 0, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 0, jcs)
    assert_jump(asm, 1, jcc)
    assert_jump(asm, 0, jcs_any)
    assert_jump(asm, 1, jcc_any)

    isub(null, r2, r3)  # some Z, some N, some C
    assert_jump(asm, 0, jzs)
    assert_jump(asm, 0, jzc)
    assert_jump(asm, 1, jzs_any)
    assert_jump(asm, 1, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 0, jnc)
    assert_jump(asm, 1, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 0, jcs)
    assert_jump(asm, 0, jcc)
    assert_jump(asm, 1, jcs_any)
    assert_jump(asm, 1, jcc_any)

def jmp_code_2(asm):
    ldi(r1, 0xffffffff)
    iadd(null, r1, 1)       # all Z, no N, all C
    assert_jump(asm, 1, jzs)
    assert_jump(asm, 0, jzc)
    assert_jump(asm, 1, jzs_any)
    assert_jump(asm, 0, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 1, jnc)
    assert_jump(asm, 0, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 1, jcs)
    assert_jump(asm, 0, jcc)
    assert_jump(asm, 1, jcs_any)
    assert_jump(asm, 0, jcc_any)

    ldi(r2, [1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,1])
    iadd(null, r1, r2)      # some Z, some N, some C
    assert_jump(asm, 0, jzs)
    assert_jump(asm, 0, jzc)
    assert_jump(asm, 1, jzs_any)
    assert_jump(asm, 1, jzc_any)
    assert_jump(asm, 0, jns)
    assert_jump(asm, 0, jnc)
    assert_jump(asm, 1, jns_any)
    assert_jump(asm, 1, jnc_any)
    assert_jump(asm, 0, jcs)
    assert_jump(asm, 0, jcc)
    assert_jump(asm, 1, jcs_any)
    assert_jump(asm, 1, jcc_any)

def test_jump():
    nout = inspect.getsource(jmp_code_1).count('assert_jump')
    X = run_code(qpu(jmp_code_1), nout)
    assert np.all(X == ASSERT_OK)

    nout = inspect.getsource(jmp_code_2).count('assert_jump')
    X = run_code(qpu(jmp_code_2), nout)
    for i in range(nout):
        if not all(X[i] == ASSERT_OK):
            print i
        assert all(X[i] == ASSERT_OK)

@qpu
def delay_slot(asm):
    mov(r1, 0)
    jmp(L._1)
    iadd(r1, r1, 1) # executed
    iadd(r1, r1, 1) # executed
    iadd(r1, r1, 1) # executed
    iadd(r1, r1, 1) # not executed

    L._1

    mov(vpm, r1)

def test_delay_slot():
    X = run_code(delay_slot, 1)
    assert np.all(X == 3)
