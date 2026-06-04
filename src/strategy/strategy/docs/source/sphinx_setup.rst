Sphinx 文件建置
===============

安裝 Sphinx
-----------

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-tabs
   pip install sphinx-autobuild

   # 選配：從 Python 型別標注自動生成說明
   pip install sphinx-autodoc-typehints

初始化專案（僅首次設定）
-------------------------

.. note::

   本專案文件已初始化完成，此步驟僅供**全新建立**文件時參考。

.. code-block:: bash

   cd ~/ros2_adult/src/strategy/strategy/docs/
   sphinx-quickstart

編譯 HTML
---------

.. code-block:: bash

   cd ~/ros2_adult/src/strategy/strategy/docs/

   # 清除舊的編譯檔案（換新電腦或大改後建議先執行）
   make clean

   # 生成 HTML 網頁
   make html

編譯完成後，輸出位於 ``build/html/index.html``\ 。

自動重建（開發用）
------------------

修改 RST 檔案時自動重新編譯並即時預覽，不需每次手動 ``make html``\ 。

啟動：

.. code-block:: bash

   cd ~/ros2_adult/src/strategy/strategy/docs
   sphinx-autobuild source build/html --host 0.0.0.0 --port 8080

瀏覽器開啟 ``http://localhost:8080`` 即可預覽。

關閉 Server：

.. code-block:: bash

   pkill -f sphinx-autobuild
