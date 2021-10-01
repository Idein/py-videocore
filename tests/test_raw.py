'Test of Raw instructions'

import io
import numpy as np
from random import getrandbits
from ctypes import create_string_buffer

from videocore.assembler import qpu, print_qhex
from videocore.driver import Driver

@qpu
def raw_hex(asm):
    raw(0xDEADBEEF, 0xFEEDFACE)

@qpu
def raw_program_hex(asm):
    data = b'\xEF\xBE\xAD\xDE\xCE\xFA\xED\xFE\xBE\xBE\xFE\xCA\x0D\xF0\xDD\xBA'
    raw_program(data)

def test_raw_hex():
    f = io.StringIO()
    print_qhex(raw_hex, file = f)
    assert f.getvalue().rstrip() == '0xDEADBEEF, 0xFEEDFACE,'

def test_raw_program():
    f = io.StringIO()
    print_qhex(raw_program_hex, file = f)
    assert f.getvalue().rstrip() == '0xDEADBEEF, 0xFEEDFACE,\n0xCAFEBEBE, 0xBADDF00D,'
