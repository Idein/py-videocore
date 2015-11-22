'Undocumented strange behavior of storing data to r0'

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def case1(asm)
    ldi(r0, 0)  # halt
    exit()

@qpu
def case2(asm):
    mov(r0, 1)  # does not halt
    exit()

@qpu
def case3(asm):
    mov(r1, 0)  # does not halt
    exit()

@qpu
def case4(asm):
    ldi(r0, [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])  # halt
    exit()

@qpu
def case5(asm):
    ldi(r0, [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])  # does not halt
    exit()

@qpu
def case6(asm):
    ldi(r1, 1)
    ldi(r2, 1)
    isub(r0, r1, r2)    # halt
    exit()

def test(prog):
    print '{} ... '.format(prog.__name__), 
    ok = True
    try:
        drv.execute(1, drv.program(prog), timeout=1)
    except:
        ok = False
    print ['\033[31mfailed.\033[39m', '\033[32mok.\033[39m'][ok]

with Driver() as drv:

    print('QPU halts when r0 of SIMD element 0 become 0')

    test(case1)
    test(case2)
    test(case3)
    test(case4)
    test(case5)
    test(case6)

    print('#### REBOOT Raspberry Pi after you run this program ####')
