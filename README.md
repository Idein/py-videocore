PyVideoCore
===========

**Python library for GPGPU programming on Raspberry Pi**

*This is work in progress project*

PyVideoCore is a Python library for GPGPU on Raspberry Pi boards.
The Raspberry Pi SoC integrates *Broadcom VideoCore IV* graphics core.  It has
12 quad processor units (QPU) which is dual-issue 16 way (4 way pipelined and 4
way true) SIMD processor.  Read the following guide thoroughly to understand
its architecture.

- [VideoCore(R) IV 3D Architecture Reference Guide (PDF)][1]

As far as we know, modern GPGPU programming environments such as OpenCL do not
support VideoCore.  Assembly programming is the only option at this moment and
several QPU assemblers written by pioneers ([hermanhermitage][2],
[petewarden][3], [elorimer][4] and so on) are available.

PyVideoCore's QPU assembler is different from theirs in terms of that our
assembly language is implemented as an *Internal DSL* of Python language.
This makes GPGPU programming on Raspberry Pi relatively easier in the sense
that

- You can put host programs and GPU side programs in a single Python script.
- You can execute the program without ahead-of-time compilation.
- You can utilize Python functionality, libraries and tools to organize GPU
  programs.

## Requirements

* Python 2.7
* nose (if you want to run tests)

## Installation

    $ git clone https://github.com/nineties/py-videocore.git
    $ cd py-videocore
    $ sudo python setup.py install

## Getting Started

You have to run GPU programs as a super user.

    $ sudo python examples/hello_world.py

## Documentation

TBD

## License
Code and documentation are released under
[MIT license](https://github.com/nineties/py-videocore/blob/master/LICENSE)

[1]: https://www.broadcom.com/docs/support/videocore/VideoCoreIV-AG100-R.pdf
[2]: https://github.com/hermanhermitage/videocoreiv-qpu/blob/master/qpu-tutorial/qpuasm.md
[3]: https://github.com/jetpacapp/qpu-asm
[4]: https://github.com/elorimer/rpi-playground/tree/master/QPU/assembler

