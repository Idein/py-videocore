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

def test_sanity_check():
  assert (not sanity_check(regfile_1))
  assert (not sanity_check(composed_1))
  assert (not sanity_check(delay_slot_1))
