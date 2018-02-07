from videocore.vinstr import *
from videocore.encoding import Register

# helper functions

def instruction_number(instrs, target):
  return instrs.index(target)

def print_with_indent(instr):
  print ('    {}'.format(instr))

def print_with_attension(instr):
  print ('>>> {}'.format(instr))

# TODO: print labels
# labels are not stored in asm._instructions, but asm._labels (tuple label with its position)
def print_instructions(instrs, indexes):
  for index in indexes:
    print_with_indent(instrs[index])

def print_around(instrs, target):
  index = instruction_number(instrs, target)
  print_instructions(instrs, range (max (0, index-2), max (0, index)))
  print_with_attension(target)
  print_instructions(instrs, range (min (len(instrs), index), min (len(instrs), index + 2)))

#================ check functions ======================================

def check_branch_delay_slot(instr, instrs, labels):
  f = True
  if (is_branch (instr)):
    index = instrs.index(instr)
    if len(instrs) < index + 3:
      print ('warning: instructions of delay_slot is short?')
      print_around(instrs, instr)
      f = False
    else:
      delay_slot = instrs[index+1:index+4]

      for item in delay_slot:
        if (is_branch (item)):
          print ('warning: branch is located in the position of delay_slot')
          print_around(instrs, item)
          f = False

  return f

# TODO: check signals
# writing to 'tmu0_s', and sending signal 'load tmu0' make invalid
def check_composed(instr, instrs, labels):
  if (is_composed(instr)):
    v = instr
    if v.add_instr.dst == v.mul_instr.dst and v.add_instr.sig != 'thread end':
      print ('warning: dst is the same register in the following composed-instruction')
      print_around(instrs, instr)
      return False
  return True

def get_outputs(instr):
  outputs = []
  if is_composed (instr):
    outputs.append (instr.add_instr.get_dst())
    outputs.append (instr.mul_instr.get_dst())
  elif instr.get_dst():
    outputs.append (instr.get_dst())
  return filter (lambda x: x != None, outputs)

def get_inputs(instr):
  inputs = []
  if is_composed (instr):
    inputs.append(instr.add_instr.get_arg1())
    inputs.append(instr.add_instr.get_arg2())
    inputs.append(instr.mul_instr.get_arg1())
    inputs.append(instr.mul_instr.get_arg2())
  else:
    inputs.append(instr.get_arg1())
    inputs.append(instr.get_arg2())
  return filter (lambda x: x != None, inputs)

# return instruction if instr is located in the position of last delay-slot
def is_in_last_delayslot (instr, instrs, labels):
  index = instrs.index(instr)
  if index - 3 < 0:
    return None

  prev = instrs[index - 3]
  if is_branch(prev):
    return instrs[labels[prev.target]]
  else:
    return None

def check_regfile(instr, instrs, labels):
  f = True
  index = instrs.index(instr)

  if len(instrs) == index + 1:
    return True

  # prev -> current
  prev = instr
  current = is_in_last_delayslot(instr, instrs, labels)
  show_current = True

  if current:
    pass
  else:
    show_current = False
    current = instrs[index + 1]

  outputs = get_outputs(prev)
  inputs = get_inputs(current)

  for out in outputs:
    for read in inputs:
      if enc.GENERAL_PURPOSE_REGISTERS.get(out.name, None) and  out.name == read.name:
        print ('warning: regfile is read next to writing instruction')
        print_around(instrs, prev)
        if show_current:
          print_around(instrs, current)
        f = False

  return f

single_steps = [check_regfile, check_composed, check_branch_delay_slot]

def single_step(instrs, labels):
  f = True
  for instr in instrs:
    for check in single_steps:
      f = f and check (instr, instrs, labels)
  return f

all_checks = [single_step]

def extract_verbose(instr):
  return instr.verbose

def check_main(instrs, labels):
  instrs = list (map(extract_verbose, instrs))
  f = True
  for check in all_checks:
    f = f and check(instrs, dict (labels))
  return f
