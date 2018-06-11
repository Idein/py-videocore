from videocore.assembler import Assembler, qpu
import videocore.checker as checker
import videocore.encoding as enc

@qpu
def prog1(asm):
  mov (r1, r0)

@qpu
def prog2(asm):
  mov (r1, r0).mov(r2, r3)

@qpu
def prog3(asm):
  imul24(r0, r1, r2)

def test_get_outputs1():
  asm = Assembler(sanity_check=True)
  prog1(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_outputs(v)) == 1)
  assert (checker.get_outputs(v)[0].name == 'r1')

def test_get_outputs2():
  asm = Assembler(sanity_check=True)
  prog2(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_outputs(v)) == 2)
  assert (checker.get_outputs(v)[0].name == 'r1')
  assert (checker.get_outputs(v)[1].name == 'r2')

def test_get_outputs3():
  asm = Assembler(sanity_check=True)
  prog3(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_outputs(v)) == 1)
  assert (checker.get_outputs(v)[0].name == 'r0')

def test_get_inputs1():
  asm = Assembler(sanity_check=True)
  prog1(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_inputs(v)) == 2)
  assert (checker.get_inputs(v)[0].name == 'r0')
  assert (checker.get_inputs(v)[0].name == 'r0')

def test_get_inputs2():
  asm = Assembler(sanity_check=True)
  prog2(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_inputs(v)) == 4)
  assert (checker.get_inputs(v)[0].name == 'r0')
  assert (checker.get_inputs(v)[1].name == 'r0')
  assert (checker.get_inputs(v)[2].name == 'r3')
  assert (checker.get_inputs(v)[3].name == 'r3')

def test_get_inputs3():
  asm = Assembler(sanity_check=True)
  prog3(asm)
  v = asm._instructions[0].verbose
  assert (len(checker.get_inputs(v)) == 2)
  assert (checker.get_inputs(v)[0].name == 'r1')
  assert (checker.get_inputs(v)[1].name == 'r2')

def test_nexts1():
  @qpu
  def nexts(asm):
    mov (r0, r1)
    mov (r0, r2)

  asm = Assembler(sanity_check=True)
  nexts(asm)
  xs, labels = checker.prepare(asm._instructions, asm._labels)
  v = checker.get_nexts(xs[0], xs, labels, 1)
  assert (len(v) == 1)
  assert (v[0] == xs[1])

def test_nexts2():
  @qpu
  def nexts(asm):
    mov (r0, r1)
    mov (r0, r2)
    mov (r0, r3)

  asm = Assembler(sanity_check=True)
  nexts(asm)
  xs, labels = checker.prepare(asm._instructions, asm._labels)
  v = checker.get_nexts(xs[0], xs, labels, 2)
  assert (len(v) == 1)
  assert (v[0] == xs[2])

def test_nexts3():
  @qpu
  def nexts(asm):
    L.label1
    jzc(L.label1)
    nop()
    nop()
    mov(r0, r1)

  asm = Assembler(sanity_check=True)
  nexts(asm)
  xs, labels = checker.prepare(asm._instructions, asm._labels)
  v1 = checker.get_nexts(xs[3], xs, labels, 1)
  assert (len(v1) == 1)
  assert (v1[0] == xs[0])
  v2 = checker.get_nexts(xs[3], xs, labels, 2)
  assert (len(v2) == 1)
  assert (v2[0] == xs[1])

def test_nexts4():
  @qpu
  def nexts(asm):
    L.label1
    jzc(L.label1)
    nop()
    nop()
    mov(r0, r1)
    iadd(r0, ra0, r1)

  asm = Assembler(sanity_check=True)
  nexts(asm)
  xs, labels = checker.prepare(asm._instructions, asm._labels)
  v1 = checker.get_nexts(xs[3], xs, labels, 1)
  assert (len(v1) == 2)
  assert (v1[0] == xs[0])
  assert (v1[1] == xs[4])
  v2 = checker.get_nexts(xs[3], xs, labels, 2)
  assert (len(v2) == 1)
  assert (v2[0] == xs[1])
