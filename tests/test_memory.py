'Test of memory allocation'

import numpy as np
from nose.tools import assert_raises
from videocore.assembler import qpu, assemble
from videocore.driver import Driver, DriverError, \
                             DEFAULT_DATA_AREA_SIZE, DEFAULT_CODE_AREA_SIZE

def test_maximum_alloc():
    with Driver() as drv:
        size = DEFAULT_DATA_AREA_SIZE
        a = drv.alloc(shape = size, dtype = np.uint8)
        assert size == a.nbytes

def test_too_large_alloc():
    with Driver() as drv:
        size = DEFAULT_DATA_AREA_SIZE + 1
        assert_raises(DriverError, drv.alloc, shape = size, dtype = np.uint8)

@qpu
def one_nop(asm):
    nop()

def test_maximum_code():
    with Driver() as drv:
        code_one_nop = assemble(one_nop)
        code = code_one_nop * (DEFAULT_CODE_AREA_SIZE // 8)
        assert len(code) == DEFAULT_CODE_AREA_SIZE
        prog = drv.program(code)
        assert prog.size == DEFAULT_CODE_AREA_SIZE

def test_too_large_code():
    with Driver() as drv:
        code_one_nop = assemble(one_nop)
        code = code_one_nop * (DEFAULT_CODE_AREA_SIZE // 8 + 1)
        assert_raises(DriverError, drv.program, code)
