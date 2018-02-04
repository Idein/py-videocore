from videocore.assembler import qpu, assemble

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
def mutex_1(asm):
  mutex_acquire()
  mutex_release()
  exit()

@qpu
def label_1(asm):
  L.label1
  iadd (r0, r1, r2)
  iadd (r0, r1, r2)
  L.label2
  iadd (r0, r1, r2, set_flags=False)
  iadd (r0, r1, r2, set_flags=True)
  jzc(L.label2)
  nop(); nop(); nop()
  exit()

@qpu
def semaphore_1(asm):
  sema_up(0)
  sema_up(1)
  sema_down(1)
  sema_down(0)
  exit()

assemble(regfile_1, sanity_check=True)
assemble(composed_1, sanity_check=True)
assemble(delay_slot_1, sanity_check=True)
assemble(mutex_1, sanity_check=True)
assemble(semaphore_1, sanity_check=True)
