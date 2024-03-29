Supplementary Information
========================================

Special function unit
---------------------

-  LOG and EXP are base 2.
-  RECIP(0) and RECIPSQRT(0) become inf.
-  RECIPSQRT(x) is equal to 1/sqrt(abs(x)).
-  LOG(x) is equal to log(abs(x)) 

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

Author
------

- Koichi Nakamura
