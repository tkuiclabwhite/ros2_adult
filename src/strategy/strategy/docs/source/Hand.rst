靈巧手 (LinkerHand O6)
--------------------------------------------------
.. currentmodule:: API

.. note::
   本章節功能為 **大人型專用**。小人型機器人未配備靈巧手，以下 API 不存在。

兩隻 LinkerHand O6，每隻六個關節，透過 CAN 與 ``handpackage`` 節點溝通。
策略這邊看到的都是包裝好的函式，不用自己發 topic。

數值與座標
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::
   **關節順序**\ （六軸，所有函式都一致）：

   1. 拇指彎曲
   2. 拇指橫擺
   3. 食指
   4. 中指
   5. 無名指
   6. 小指

   **位置 0~255**\ ：0 為收到底（握），255 為張到底（開）。
   填 **-1** 代表「這一軸維持不動」。

   **速度 10~255**\ ：越大越快。填 **0** 代表「這一軸沿用目前的速度」。
   不要填 1~9 —— 底層 SDK 只要收到一個超出 10~255 的值，**會把整包速度丟掉**。

左右手共用同一組數值，不需要做鏡像轉換（``side`` 填 ``'both'`` 時兩手下一樣的值）。

所有函式的 ``side`` 參數都接受 ``'left'`` / ``'right'`` / ``'both'``\ ，
填其他值會丟 ``ValueError``\ 。

.. warning::
   **這一章的函式都不會等動作完成。**

   手抓到東西時實際位置**永遠到不了目標**\ （被物體擋住），所以沒有辦法用
   「位置是否收斂」來判斷完成，也就沒有提供阻塞式的 wait 函式。
   唯一可靠的完成訊號是夾持的 :attr:`API.hand_grip_active`\ ，見下方「夾持」一節。

位置控制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: API.setHand
.. automethod:: API.sendHandMotor

速度與扭矩
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

速度影響「之後的位置指令」，呼叫當下手不會動。
扭矩是硬體層的上限，與夾持的壓力門檻（軟體層）互相獨立 ——
要夾易碎的東西建議兩層一起用。

.. automethod:: API.setHandSpeed
.. automethod:: API.setHandTorque

預設動作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

手勢的位置與速度定義在 ``hurocup_interface/js/hand_presets.js``\ ，
**與網頁共用同一份** —— 在 HandControl 網頁上調好的手勢，策略這邊直接就能用。

目前可用的名稱見 :attr:`API.hand_presets`\ ，有「張開」「握拳」「OK」「點贊」
「一」~「九」「中指」「蜘蛛人」「搖滾」等。

.. note::
   那份定義是在 API 節點**啟動時讀一次**\ 。改完數值要重跑策略才會生效
   （網頁那邊重新載入就好）。

.. automethod:: API.handPreset

夾持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

手會慢慢往閉合方向走，觸覺壓力到門檻就停住。這是手部動作裡
**唯一真正閉迴路**\ 的功能。

壓力門檻的參考值（每根手指 10x4=40 格矩陣的總和，範圍 0~10200，不是公克）。
實測空載為 0，所以讀到數字就代表真的碰到東西了：

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 門檻
     - 大約的力道
   * - ``50`` ~ ``150``
     - 輕觸，剛碰到就停
   * - ``200`` ~ ``400``
     - 輕握（一般抓取建議從這裡開始試）
   * - ``600`` ~ ``1000``
     - 紮實，拿得穩但會把軟的東西壓變形
   * - ``2000`` 以上
     - 夾死

不確定要填多少時，開 HandTouch 網頁用手捏那根手指到想要的力道，看讀數是多少。

**要等夾持結束就輪詢** :attr:`API.hand_grip_active`\ ：

.. code-block:: python

   api.handGrip('left', threshold=250, timeout=8)
   while api.hand_grip_active:
       time.sleep(0.05)
   print(api.hand_grip_reason)   # 壓力到門檻（thumb, index）/ 閉合到目標位置 / 逾時

.. automethod:: API.handGrip

觸覺壓力
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: API.handTouch

動作序列
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

手部的 sector 與 MotionList 用 HandMotion 網頁編輯，
**編號空間與全身的動作完全獨立**\ （手的 5 號跟身體的 5 號沒有關係）。

左右手是各自獨立的時間軸，可以同時播不同的序列。

.. automethod:: API.sendHandSector
.. automethod:: API.handPlay
.. automethod:: API.handPlayStop

停止
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: API.handStop

狀態
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下屬性由 ``/hand/status``\ （20 Hz）自動更新，直接讀即可。

.. autoattribute:: API.hand_online
.. autoattribute:: API.hand_position
.. autoattribute:: API.hand_target
.. autoattribute:: API.hand_temperature
.. autoattribute:: API.hand_mass
.. autoattribute:: API.hand_touch_hold
.. autoattribute:: API.hand_grip_active
.. autoattribute:: API.hand_grip_reason
.. autoattribute:: API.hand_play_active
.. autoattribute:: API.hand_mode
.. autoattribute:: API.hand_strategy
.. autoattribute:: API.hand_presets
