Undocumented behavior of hardware
=================================

List of behaviors which seem to be not documented in
`VideoCore(R) IV 3D Architecture Reference Guide
<https://www.broadcom.com/docs/support/videocore/VideoCoreIV-AG100-R.pdf>`__.
Please contact me when you found following cases.

-  It is documented.
-  It is caused by a bug of a library or a firmware.

Writing 0 to element 0 of r0
----------------------------

-  QPU halts when you write 0 to the element 0 of the accumulator r0.
-  The hardware becomes unstable after this event.
-  See `experiments/write_zero_to_r0.py
   <https://github.com/nineties/py-videocore/blob/master/experiments/write_zero_to_r0.py>`__.

Errata of 'VideoCore IV Reference Guide'
========================================

This is an errata of the reference guide. I can not guarantee correctness of
this list.

+--------+-------------------------------+-----------------------------------+
| Page   |          Incorrect            |             Correct               |
+========+===============================+===================================+
| 32     | Table 8: R4 Pack Encoding     | Table 8: R4 *Unpack* Encoding     |
+--------+---+---------------------------+---+-------------------------------+
| 32     | 4 | 32 -> 8a                  | 4 | 32 -> 8a                      |
|        +---+---------------------------+---+-------------------------------+
|        | 5 | 32 -> 8a                  | 5 | 32 -> *8b*                    |
|        +---+---------------------------+---+-------------------------------+
|        | 6 | 32 -> 8a                  | 6 | 32 -> *8c*                    |
|        +---+---------------------------+---+-------------------------------+
|        | 7 | 32 -> 8a                  | 7 | 32 -> *8d*                    |
+--------+---+---------------------------+---+-------------------------------+
| 36     | in the range [1.0, 0]         | in the range *[0.0, 1.0]*         |
+--------+---+---------------------------+---+-------------------------------+

Author
------

- Koichi Nakamura
