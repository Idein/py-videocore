'Test of Raw instructions'

import io
import numpy as np
from random import getrandbits

from videocore.assembler import qpu, print_qhex
from videocore.driver import Driver

@qpu
def raw_hex(asm):
    raw(0xDEADBEEF, 0xFEEDFACE)

def test_raw_hex():
    f = io.StringIO()
    print_qhex(raw_hex, file = f)
    assert f.getvalue().rstrip() == '0xDEADBEEF, 0xFEEDFACE,'
