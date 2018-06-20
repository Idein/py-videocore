import videocore.encoding as enc

#=============================================================================
# Verbose instruction definitions for sanity checks
#=============================================================================

def is_add(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, AddInstr)
def is_mul(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, MulInstr)
def is_loadimm(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, LoadImmInstr)
def is_branch(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, BranchInstr)
def is_sema(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, SemaInstr)
def is_composed(instr):
  if not isinstance (instr, InstrBase):
    assert(False)
  return isinstance(instr, ComposedInstr)

class InstrBase(object):
  def __init__():
    return

  def is_nop(self):
    return False

  def get_dst(self):
    return None

  def get_arg1(self):
    return None

  def get_arg2(self):
    return None

  def get_sig(self):
    return None

class AddInstr(InstrBase):
  def __init__ (self, op, dst, opd1, opd2, sig, set_flag, cond):
    self.op = op
    self.dst = dst
    self.opd1 = opd1
    self.opd2 = opd2
    self.sig = sig
    self.set_flag = set_flag
    self.cond = cond

  def __str__(self):
    if self.get_dst() == enc.REGISTERS['null'] and \
       self.get_arg1() == enc.REGISTERS['mutex'] and \
       self.get_arg2() == enc.REGISTERS['mutex'] and \
       enc._ADD_INSN_REV[self.op] == 'bor':
      return 'mutex_acquire'
    elif self.get_dst() == enc.REGISTERS['mutex'] and \
         self.get_arg1() == enc.REGISTERS['null'] and \
         self.get_arg2() == enc.REGISTERS['null'] and \
         enc._ADD_INSN_REV[self.op] == 'bor':
      return 'mutex_release'

    if self.op == 'nop':
      s = 'nop'
    else :
      s = '{} {}, {}, {}'.format(self.op, self.dst, self.opd1, self.opd2)
    if self.set_flag:
      s += ' set_flag=True'
    if not (self.cond == 'always'):
      s += ' cond={}'.format(self.cond)
    if not (self.sig == 'no signal'):
      s += ' sig={}'.format(self.sig)
    return s

  def is_nop(self):
    return self.op == 'nop'

  def get_dst(self):
    return self.dst

  def get_arg1(self):
    return self.opd1

  def get_arg2(self):
    return self.opd2

  def get_sig(self):
    return self.sig

class MulInstr(InstrBase):
  def __init__ (self, op, dst, opd1, opd2, sig, set_flag, cond, rotate):
    self.op = op
    self.dst = dst
    self.opd1 = opd1
    self.opd2 = opd2
    self.sig = sig
    self.set_flag = set_flag
    self.cond = cond
    self.rotate = rotate

  def __str__(self):
    if self.op == 'nop':
      s = 'nop'
    else:
      s = '{}, {}, {}, {}'.format(self.op, self.dst, self.opd1, self.opd2)
    if self.set_flag:
      s += ' set_flag=True'
    if not (self.cond == 'always'):
      s += ' cond={}'.format(self.cond)
    if not (self.sig == 'no signal'):
      s += ' sig={}'.format(self.sig)
    if self.rotate:
      s += ' rotate={}'.format(self.rotate)
    return s

  def get_rotate(self):
    return self.rotate

  def is_nop(self):
    return self.op == 'nop'

  def get_dst(self):
    return self.dst

  def get_arg1(self):
    return self.opd1

  def get_arg2(self):
    return self.opd2

  def get_sig(self):
    return self.sig

class LoadImmInstr(InstrBase):
  def __init__(self, reg1, reg2, imm):
    self.reg1 = reg1
    self.reg2 = reg2
    self.imm = imm

  def __str__(self):
    s = 'ldi {}, {}'.format(self.reg1, self.imm)
    if (self.reg2 != enc.REGISTERS['null']):
      s += '; ldi {}, {}'.format(self.reg2, self.imm)
    return s

  def get_dst(self):
    return self.reg1

  def get_arg1(self):
    return self.imm

  def get_arg2(self):
    return None


class BranchInstr(InstrBase):
  def __init__(self, cond_br, target, reg, absolute, link):
    self.cond_br = cond_br
    self.target = target
    self.reg = reg
    self.absolute = absolute
    self.link = link

  def __str__(self):
    s = self.cond_br
    if self.target:
      s += '(L.{})'.format(self.target.name)
    else:
      s += link
    return s

class SemaInstr(InstrBase):
  def __init__(self, sa, sema_id):
    self.sa = sa
    self.sema_id = sema_id

  def __str__(self):
    if self.sa == 0:
      s = 'sema_up'
    else:
      s = 'sema_down'
    return '{}({})'.format(s, self.sema_id)

class ComposedInstr(InstrBase):
  def __init__(self, add, mul):
    self.add_instr = add
    self.mul_instr = mul

  def __str__(self):
    return str(self.add_instr) + '; ' + str(self.mul_instr)

  def get_dst(self):
    assert (False)

  def get_arg1(self):
    assert (False)

  def get_arg2(self):
    assert (False)

  def get_sig(self):
    if self.add_instr.sig == 'no signal':
      return self.mul_instr.sig
    else:
      return self.add_instr.sig

class Label(InstrBase):
  def __init__(self, name):
    self.name = name

  def __str__(self):
    return ':' + self.name
