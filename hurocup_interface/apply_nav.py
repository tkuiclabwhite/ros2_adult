#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把寫死的導覽列換成 js/nav.js 產生的版本（就地改，可以重複執行）。

為什麼要這支：導覽列原本在每一頁各寫一份，加一個頁面就要改 12 個檔案，
結果就是漂移 —— 有的頁面少了某個連結，ImuMonitor 更是只有 WalkingInterface
那一頁有。改成 nav.js 之後選單只有一份。

用在哪：把新機器（例如 172.17.121.10）上的既有頁面轉過去，不用整包覆蓋。
只動兩個地方，其他內容完全不碰：

  1. </head> 前面插一行 <script src="./js/nav.js"></script>
  2. <body> 裡那段寫死的 <ul>…</ul> 導覽列換成 <ul id="mainnav"></ul>

用法：
    python3 apply_nav.py              # 看會改什麼，不寫檔
    python3 apply_nav.py --write      # 真的改
    python3 apply_nav.py --write a.html b.html    # 只改指定的檔案

前提：js/nav.js 要先放進 hurocup_interface/js/。
已經轉過的頁面會被跳過，所以重複執行是安全的。
"""
import io
import os
import re
import sys

# 桌面版 12 頁 + MobileJoystick（手機版選單，<ul> 在 #quickNavBar 裡面）
DESKTOP = [
    'index.html', 'WalkingInterface.html', 'MotionControlInterface.html',
    'ImageProcessInterface.html', 'USB_ImageProcessInterface.html',
    'RecordInterface.html', 'NodeMonitor.html', 'MotorMonitor.html',
    'WaveformMonitor.html', 'miniDRC.html', 'arm_control.html',
    'ImuMonitor.html',
]
MOBILE = ['MobileJoystick.html']

SCRIPT_TAG = '<script src="./js/nav.js"></script>'


def convert(src, mobile=False):
    """回傳 (新內容, 做了什麼)。不需要改就回 (src, [])。"""
    done = []
    nl = '\r\n' if '\r\n' in src else '\n'

    # ---- 1. <script src="./js/nav.js"> ----
    if 'js/nav.js' not in src:
        m = re.search(r'([ \t]*)</head>', src)
        if not m:
            return src, ['找不到 </head>，跳過']
        # 縮排抄 head 裡最後一個 <script>/<link>，不要抄 </head> ——
        # 那一行常常在第 0 欄，插出來會跟上下排不齊（有的頁面用 tab 有的用空白）
        head = src[:m.start()]
        tags = re.findall(r'(?m)^([ \t]*)<(?:script|link)\b', head)
        indent = tags[-1] if tags else m.group(1)
        src = src[:m.start()] + indent + SCRIPT_TAG + nl + src[m.start():]
        done.append('加上 nav.js')

    # ---- 2. 寫死的導覽列 -> <ul id="mainnav"></ul> ----
    if 'id="mainnav"' in src:
        done.append('導覽列已經轉過了')
        return src, done

    body_at = src.find('<body')
    if body_at < 0:
        return src, done + ['找不到 <body>，導覽列沒換']

    # 找 body 裡第一個「看起來是導覽列」的 <ul>：裡面要有 navbarright 或 class="logo"。
    # 用數標籤的方式配對結尾，不能用正則 —— 導覽列裡還有巢狀的 <ul class="dropdown-menu">
    for m in re.finditer(r'[ \t]*<ul\b[^>]*>', src[body_at:]):
        start = body_at + m.start()
        depth, i = 0, body_at + m.start()
        end = None
        for t in re.finditer(r'<(/?)ul\b[^>]*>', src[i:]):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                end = i + t.end()
                break
        if end is None:
            continue
        block = src[start:end]
        if 'navbarright' not in block and 'class="logo"' not in block:
            continue          # 不是導覽列（頁面內容自己的清單），往下找
        indent = re.match(r'[ \t]*', block).group(0)
        attr = ' data-nav="mobile"' if mobile else ''
        src = src[:start] + indent + '<ul id="mainnav"' + attr + '></ul>' + src[end:]
        done.append('導覽列換成 mainnav')
        break
    else:
        done.append('找不到寫死的導覽列（可能本來就沒有）')
    return src, done


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    write = '--write' in sys.argv

    if not os.path.exists(os.path.join('js', 'nav.js')):
        print('✗ js/nav.js 不存在。先把它複製進來再跑這支。')
        return 1

    targets = ([(f, f in MOBILE) for f in args] if args
               else [(f, False) for f in DESKTOP] + [(f, True) for f in MOBILE])

    changed = 0
    for name, mobile in targets:
        if not os.path.exists(name):
            print(f'  -  {name}：這台沒有這個檔案，跳過')
            continue
        # newline='' 保留原本的換行字元 —— 不然整檔 CRLF 會被改成 LF，
        # diff 會變成「整個檔案都改了」
        with io.open(name, encoding='utf-8', newline='') as f:
            src = f.read()
        new, done = convert(src, mobile)
        if new == src:
            print(f'  =  {name}：不用改（{"、".join(done) or "已是最新"}）')
            continue
        changed += 1
        print(f'  *  {name}：{"、".join(done)}')
        if write:
            with io.open(name, 'w', encoding='utf-8', newline='') as f:
                f.write(new)

    print()
    if not write:
        print(f'以上是預覽，{changed} 個檔案會被改。要真的改請加 --write')
    else:
        print(f'已改 {changed} 個檔案。用瀏覽器開一頁確認導覽列出得來。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
