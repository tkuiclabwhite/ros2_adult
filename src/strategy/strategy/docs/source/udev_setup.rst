udev 規則
=========

設定 udev 規則可讓裝置在每次連接時都使用固定的設備名稱。

U2D2
----

U2D2 是 Dynamixel 馬達的 USB 轉接器，共 3 顆，需依序號綁定固定名稱。

.. code-block:: bash

   sudo nano /etc/udev/rules.d/99-u2d2.rules

填入以下內容（序號依實際情況填寫）：

.. code-block:: text

   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_1>", SYMLINK+="U2D2_P1", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"
   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_2>", SYMLINK+="U2D2_P2", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"
   SUBSYSTEM=="tty", ATTRS{serial}=="<ADULT_U2D2_SERIAL_3>", SYMLINK+="U2D2_P3", GROUP="dialout", MODE="0660", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"

查詢 U2D2 序號（接上後執行）：

.. code-block:: bash

   udevadm info -a -n /dev/ttyUSB0 | grep serial
   udevadm info -a -n /dev/ttyUSB1 | grep serial
   udevadm info -a -n /dev/ttyUSB2 | grep serial

套用規則：

.. code-block:: bash

   sudo udevadm control --reload-rules && sudo udevadm trigger
