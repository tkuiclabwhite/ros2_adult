import os
import sys
import shutil
import subprocess

# 讓 Sphinx 找到上一層目錄的 API.py
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'TKU ICLAB ROS2 API'

# 版本號：**改文件時人工更新這一行**。
# 原本是自動抓 git 最後一次 commit 日期，但文件改了卻沒 commit 時版本號不會動，
# 會誤以為站上是舊的。
release = '2026.10.20'
copyright = '2026, Yang,Shu-kai'
author = 'Yang,Shu-kai'
html_short_title = 'ICLAB API'

extensions = [
    'sphinx.ext.autodoc',     # 抓取註解
    'sphinx.ext.napoleon',    # 支援 Google 格式註解
    'sphinx_rtd_theme',       # 使用 Read the Docs 主題
    'sphinx_tabs.tabs',       # tab 指令
    'sphinx_copybutton',      # code block 複製按鈕
]

def _copy_screenshots(app, exception):
    """同步 _static/screenshots 到輸出目錄。

    Sphinx 只會處理被 .. image:: 引用的圖，介面圖是寫在 .. raw:: html 的
    <img src> 裡，它看不到，所以要自己複製。
    """
    if exception or app.builder.format != 'html':
        return
    src_shots = os.path.join(app.confdir, '_static', 'screenshots')
    dst_shots = os.path.join(app.outdir, '_static', 'screenshots')
    if os.path.isdir(src_shots):
        if os.path.exists(dst_shots):
            shutil.rmtree(dst_shots)
        shutil.copytree(src_shots, dst_shots)

def setup(app):
    app.connect('build-finished', _copy_screenshots)

autodoc_mock_imports = [
    "rclpy",
    "numpy",
    "cv_bridge",
    "std_msgs",
    "std_srvs",      # API.sendOdomReset 用到 Trigger
    "nav_msgs",      # ZED 里程計用到 Odometry
    "geometry_msgs",
    "sensor_msgs",
    "tku_msgs",
]

html_theme = 'sphinx_rtd_theme'
language = 'zh_TW'            # 設定為繁體中文

add_module_names = False  # 去掉標頭的模組路徑，只留函式名
napoleon_use_rtype = False # 將回傳型別併入描述，取消獨立的「回傳型別」列
napoleon_use_admonition_for_examples = False # 關閉範例的警告框格式

autoclass_content = 'both' # 同時抓取類別與 __init__ 的註解
