PyVideoCore
===========

*This is work in progress project. Backward compatibility is not guaranteed.*

**PyVideoCore** is a Python library for GPGPU on Raspberry Pi boards. The
Raspberry Pi SoC integrates **Broadcom VideoCore IV** graphics core. It
has 12 quad processor units (QPU) which is a dual-issue 16 way (4 way
pipelined and 4 way true) SIMD processor. Read the following guide
thoroughly to study its architecture.

-  `VideoCore(R) IV 3D Architecture Reference Guide
   (PDF) <https://docs.broadcom.com/docs/12358545>`__
   [#appendix]_

Several QPU assemblers are written by pioneers
(`hermanhermitage <https://github.com/hermanhermitage/videocoreiv-qpu/blob/master/qpu-tutorial/qpuasm.md>`__,
`petewarden <https://github.com/jetpacapp/qpu-asm>`__,
`elorimer <https://github.com/elorimer/rpi-playground/tree/master/QPU/assembler>`__
and so on). There is also an implementation of OpenCL for QPU: `VC4CL <https://github.com/doe300/VC4CL>`_.

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

-  Raspberry Pi Zero, 1, 2, or 3

   - For Raspberry Pi 4, use
     `py-videocore6 <https://github.com/Idein/py-videocore6>`__ instead.

-  Python 2 (>= 2.6) or Python 3
-  NumPy
-  `rpi-vcsm <https://github.com/Idein/rpi-vcsm>`__ ~= 3.0.0
-  `ioctl-opt <https://github.com/vpelletier/python-ioctl-opt>`__ ~= 1.2
-  nose (if you want to run tests)

Installation
------------

::

    $ git clone https://github.com/nineties/py-videocore.git
    $ cd py-videocore
    $ python setup.py install

Note that PyVideoCore does not work with the CPU-side OpenGL graphics stack,
so configure your pi to use the legacy (original non-GL desktop) driver by the
``sudo raspi-config`` command (it just comments out all the
``dtoverlay=vc4-kms-v3d`` and ``dtoverlay=vc4-fkms-v3d`` lines in
``/boot/config.txt``).

Depending on your running kernel version, PyVideoCore allocates memory through
``/dev/vcsm`` or ``/dev/vcsm-cma``, which are the devices of the VCSM (VideoCore
shared memory service) and the VCSM-CMA (contiguous memory allocator) drivers,
respectively.
To access the devices, you need to belong to the ``video`` group or need to be
the ``root`` user.
If you choose the former, run the following command and re-login.

::

    $ sudo usermod --append --groups video $USER

The plain VCSM driver allocates memory in the GPU-side memory, of which size can
be configured by the ``gpu_mem=XXX`` option in ``/boot/config.txt`` (e.g.
``gpu_mem=128`` for 128 MB).
This can also be done via the ``sudo raspi-config`` command.

On the other hand, the VCSM-CMA driver allocates memory in the CPU-side CMA
memory, of which size can be configured by the ``dtoverlay=cma,cma-XXX`` option
in ``/boot/config.txt`` (e.g. ``dtoverlay=cma,cma-128`` for 128 MB).

Nevertheless, VideoCore IV QPUs can access arbitrary portions of the main
memory, which may make your system unstable and even break your pi, so beware of
bugs in the programs.

Getting Started
---------------

::

    $ python examples/hello_world.py

Running Tests
-------------

::

    $ nosetests -v

- 128MB or more GPU memory is required to pass tests. Failed some tests with 64MB or less.

Documentation
-------------

TBD

Tutorials
---------

In japanese.

- `Raspberry PiでGPGPU <http://qiita.com/9_ties/items/2e85318989170f967e4b>`__
- `Raspberry PiのGPUで行列乗算(その1) <http://qiita.com/9_ties/items/15ab7fa198991a61a3a9>`__
- `Raspberry PiのGPUで行列乗算(その2) <http://qiita.com/9_ties/items/e0fdd165c1c7df6bb8ee>`__

Records
-------

- Achieved 8GFlops with sgemm.
.. image:: https://pbs.twimg.com/media/CWYjkH7U4AAh9VE.jpg

License
-------

Code and documentation are released under `MIT
license <https://github.com/nineties/py-videocore/blob/master/LICENSE>`__


----

.. [#appendix] `Supplementary information and errata list.
             <https://github.com/nineties/py-videocore/blob/master/APPENDIX.rst>`__
