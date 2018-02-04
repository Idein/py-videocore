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

def check_branch_delay_slot(instrs):
  for instr in instrs:
    if (is_branch (instr)):
      index = instrs.index(instr)
      if len(instrs) < index + 3:
        print ('warning: instructions of delay_slot is short?')
        print_around(instrs, instr)
      else:
        delay_slot = instrs[index+1:index+4]

        for item in delay_slot:
          if (is_branch (item)):
            print ('warning: branch is located in the position of delay_slot')
            print_around(instrs, item)

# TODO: check signals
# writing to 'tmu0_s', and sending signal 'load tmu0' make invalid
def check_composed(instrs):
  for instr in instrs:
    if (is_composed(instr)):
      v = instr
      if v.add_instr.dst == v.mul_instr.dst and v.add_instr.sig != 'thread end':
        print ('warning: dst is the same register in the following composed-instruction')
        print_around(instrs, instr)

def check_regfile(instrs):
  if len(instrs) == 1:
    return

  for index in range (0, len(instrs) - 1):
    prev = instrs[index]
    current = instrs[index+1]

    outputs = []
    inputs = []
    def add(list, instr, f):
      if f(instr) and isinstance (f(instr), Register):
        list.append(f (instr))

    if is_composed (prev):
      add (outputs, prev.add_instr, lambda x: x.get_dst())
      add (outputs, prev.mul_instr, lambda x: x.get_dst())
    elif prev.get_dst():
      add (outputs, prev, lambda x: x.get_dst())

    if is_composed (current):
      add (inputs, current.add_instr, lambda x: x.get_arg1())
      add (inputs, current.add_instr, lambda x: x.get_arg2())
      add (inputs, current.mul_instr, lambda x: x.get_arg1())
      add (inputs, current.mul_instr, lambda x: x.get_arg2())
    elif current.get_dst():
      add (inputs, current, lambda x: x.get_arg1())
      add (inputs, current, lambda x: x.get_arg2())

    for out in outputs:
      for read in inputs:
        if enc.GENERAL_PURPOSE_REGISTERS.has_key(out.name) and  out.name == read.name:
          print ('warning: regfile is read next to writing instruction')
          print_around(instrs, current)

all_checks = [check_composed, check_regfile, check_branch_delay_slot]

def extract_verbose(instr):
  return instr.verbose

def check_main(instrs):
  instrs = map(extract_verbose, instrs)
  for check in all_checks:
    check(instrs)
  return
