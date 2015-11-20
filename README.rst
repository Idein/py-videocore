PyVideoCore
===========

*This is work in progress project.*

PyVideoCore is a Python library for GPGPU on Raspberry Pi boards. The
Raspberry Pi SoC integrates **Broadcom VideoCore IV** graphics core. It
has 12 quad processor units (QPU) which is dual-issue 16 way (4 way
pipelined and 4 way true) SIMD processor. Read the following guide
thoroughly to study its architecture.

-  `VideoCore(R) IV 3D Architecture Reference Guide
   (PDF) <https://www.broadcom.com/docs/support/videocore/VideoCoreIV-AG100-R.pdf>`__
   [#errata]_

As far as I know, modern GPGPU programming environments such as OpenCL
does not exist for  VideoCore. Assembly programming is the only option at
this moment and several QPU assemblers written by pioneers
(`hermanhermitage <https://github.com/hermanhermitage/videocoreiv-qpu/blob/master/qpu-tutorial/qpuasm.md>`__,
`petewarden <https://github.com/jetpacapp/qpu-asm>`__,
`elorimer <https://github.com/elorimer/rpi-playground/tree/master/QPU/assembler>`__
and so on) are available.

PyVideoCore's QPU assembler is different from theirs in terms of that
its assembly language is implemented as an **Internal DSL** of Python
language. This makes GPGPU programming on Raspberry Pi relatively easier
in the sense that

-  You can put host programs and GPU side programs in a single Python
   script.
-  You can execute the program without ahead-of-time compilation.
-  You can utilize Python functionality, libraries and tools to organize
   GPU programs.

Requirements
------------

-  Python 2.7
-  NumPy
-  nose (if you want to run tests)

Installation
------------

::

    $ git clone https://github.com/nineties/py-videocore.git
    $ cd py-videocore
    $ sudo python setup.py install

You might need to update firmware.

::

    $ sudo rpi-update

Be Careful
----------

-  Raspberry Pi 2 is not tested.
-  You need to run programs as a super user so that this library can access
   ``/dev/mem``.
-  Accessing wrong location of ``/dev/mem``, due to a bug of this library, may
   make your system unstable or crash it. Reboot in such cases.

Getting Started
---------------

::

    $ sudo python examples/hello_world.py

Running Tests
-------------

::

    sudo nosetests -v

Documentation
-------------

TBD

License
-------

Code and documentation are released under `MIT
license <https://github.com/nineties/py-videocore/blob/master/LICENSE>`__

.. rubric:: Footnotes

.. [#errata] `Errata of reference guide
   <https://github.com/nineties/py-videocore/blob/master/ERRATA.rst>`__
