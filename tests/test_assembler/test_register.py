'Test of videocore.Register'

from nose.tools import raises

from videocore.assembler import Register, AssembleError, REGISTERS

def test_register_names():
    for name in REGISTERS:
        assert name == REGISTERS[name].name
        assert name == str(REGISTERS[name])

@raises(AssembleError)
def test_pack_of_accumulator():
    REGISTERS['r0'].pack('nop')

@raises(AssembleError)
def test_pack_of_regfileB():
    REGISTERS['rb0'].pack('nop')

@raises(AssembleError)
def test_unpack_of_regfileB():
    REGISTERS['rb0'].unpack('nop')
