from videocore.assembler import qpu, sanity_check

@qpu
def regfile_1(asm):
  iadd (ra0, r0, r1)
  iadd (r0, ra0, r1)
  exit()

@qpu
def composed_1(asm):
  iadd (r0, r1, r2).v8min(r0, r1, r2)
  exit()

@qpu
def delay_slot_1(asm):
  L.label1
  jzc(L.label1)
  jzc(L.label1)
  nop()
  jzc(L.label1)
  nop()
  nop()
  jzc(L.label1)
  exit()

@qpu
def regfile_2(asm):
  L.label1
  iadd (r0, ra0, r0)
  jzc(L.label1)
  nop()
  nop()
  iadd (ra0, r0, r0)
  exit()

def test_sanity_check():
  assert (not sanity_check(regfile_1))
  assert (not sanity_check(composed_1))
  assert (not sanity_check(delay_slot_1))
  assert (not sanity_check(regfile_2))

def test_tmu_reg1():
  @qpu
  def tmu_reg1(asm):
    mov (tmu0_s, r0)

  for name, prog in locals().items():
    assert (sanity_check(prog))

def test_tmu_reg2():
  @qpu
  def tmu_reg1(asm):
    mov (tmu0_s, r0, sig='load tmu0')

  @qpu
  def tmu_reg2(asm):
    mov (tmu0_s, r0, sig='load tmu1')

  @qpu
  def tmu_reg3(asm):
    mov (tmu1_s, r0, sig='load tmu0')

  @qpu
  def tmu_reg4(asm):
    mov (tmu1_s, r0, sig='load tmu1')

  @qpu
  def tmu_reg5(asm):
    mov (tmu0_s, r0).imul24(r0, r1, r2, sig='load tmu0')

  for name, prog in locals().items():
    assert (not sanity_check(prog))

def test_sfu1():
  @qpu
  def sfu1(asm):
    iadd(sfu_exp2, r0, r1)
    iadd(r0, r4, r0)

  @qpu
  def sfu2(asm):
    iadd(sfu_exp2, r0, r1)
    nop()
    iadd(r0, r4, r0)

  @qpu
  def sfu3(asm):
    L.label1
    mov (r0, r4)
    jzc(L.label1)
    nop()
    nop()
    iadd(sfu_log2, r0, r1)

  @qpu
  def sfu4(asm):
    L.label1
    nop ()
    mov (r0, r4)
    jzc(L.label1)
    nop()
    nop()
    iadd(sfu_log2, r0, r1)

  @qpu
  def sfu5(asm):
    L.label1
    mov (r0, r4)
    jzc(L.label1)
    nop()
    iadd(sfu_log2, r0, r1)
    nop()

  @qpu
  def sfu6(asm):
    L.label1
    nop()
    jzc(L.label1)
    nop()
    nop()
    iadd(sfu_log2, r0, r1)
    mov (r0, r4)
    nop()

  for name, prog in locals().items():
    assert (not sanity_check(prog))

def test_sfu2():
  @qpu
  def sfu1(asm):
    iadd(sfu_log2, r0, r1)
    iadd(sfu_exp2, r0, r1)
    nop()
    nop()

  @qpu
  def sfu2(asm):
    iadd(sfu_log2, r0, r1)
    nop()
    iadd(sfu_exp2, r0, r1)
    nop()
    nop()

  @qpu
  def sfu3(asm):
    iadd(sfu_log2, r0, r1)
    nop()
    nop(sig='load tmu0')
    nop()
    nop()

  @qpu
  def sfu4(asm):
    iadd(sfu_log2, r0, r1)
    nop(sig='load tmu1')
    nop()
    nop()
    nop()

  for name, prog in locals().items():
    assert (not sanity_check(prog))

def test_sfu3():
  @qpu
  def sfu1(asm):
    iadd(sfu_log2, r0, r1)
    nop()
    nop()
    iadd(sfu_exp2, r0, r1)
  for name, prog in locals().items():
    assert (sanity_check(prog))

test_sfu2()
test_sfu3()
