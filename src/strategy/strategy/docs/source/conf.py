import os
import sys
import shutil
import subprocess

# 讓 Sphinx 找到上一層目錄的 API.py
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'TKU ICLAB ROS2 API'

# 自動從 git log 取得最後一次 commit 日期作為版本號
try:
    release = subprocess.check_output(
        ['git', '-C', os.path.abspath('../../'), 'log', '-1', '--format=%cd', '--date=format:%Y.%m.%d'],
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    release = '未知'
copyright = '2026, Yang,Shu-kai'
author = 'Yang,Shu-kai'
html_short_title = 'ICLAB API'

extensions = [
    'sphinx.ext.autodoc',     # 抓取註解
    'sphinx.ext.napoleon',    # 支援 Google 格式註解
    'sphinx_rtd_theme',       # 使用 Read the Docs 主題
    'sphinx_tabs.tabs',       # tab 指令
]

def _copy_interface(app, exception):
    if exception or app.builder.format != 'html':
        return
    src = os.path.abspath(os.path.join(app.confdir, '../../../../../hurocup_interface'))
    dst = os.path.join(app.outdir, 'hurocup_interface')
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        shutil.copytree(src, dst)

def _copy_demo_html(app, exception):
    if exception or app.builder.format != 'html':
        return
    import glob
    src_dir = os.path.join(app.confdir, '_static')
    dst_dir = os.path.join(app.outdir, '_static')
    os.makedirs(dst_dir, exist_ok=True)
    for html_file in glob.glob(os.path.join(src_dir, '*_demo.html')):
        shutil.copy2(html_file, dst_dir)
    src_shots = os.path.join(src_dir, 'screenshots')
    dst_shots = os.path.join(dst_dir, 'screenshots')
    if os.path.isdir(src_shots):
        if os.path.exists(dst_shots):
            shutil.rmtree(dst_shots)
        shutil.copytree(src_shots, dst_shots)

def setup(app):
    app.connect('build-finished', _copy_interface)
    app.connect('build-finished', _copy_demo_html)

autodoc_mock_imports = [
    "rclpy",
    "numpy",
    "cv_bridge",
    "std_msgs",
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
