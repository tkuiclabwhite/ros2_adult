馬達控制 (Motion Control)
-------------------------------
.. currentmodule:: API

.. automethod:: API.sendBodySector
.. automethod:: API.sendHeadMotor
.. automethod:: API.sendSingleMotor

body 馬達 ID 範圍：**1~27**。頭部馬達（ID 28 水平、ID 29 垂直）請使用 ``sendHeadMotor``\ 。

.. automethod:: API.SingleAbsolutePosition