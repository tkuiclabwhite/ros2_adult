#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""靜態檢查 hurocup_interface 的頁面。

這台機器（和機器人）都沒有 node/d8 之類的 JS 引擎可用，Firefox headless 也產不出
截圖，所以沒辦法真的把頁面跑起來。這支做的是結構層的檢查，能攔掉最常見的那幾類
錯誤：

  * JS 括號 / 字串不平衡（認得正規表示式字面值，/"/g 這種不會被誤判成字串）
  * HTML 的 onclick/onchange 指到不存在的函式
  * 程式裡寫死的 element id 在 HTML 裡找不到
  * 用到的 class 沒有對應的 CSS 規則
  * HTML 標籤開關數不相等

用法：
    python3 hurocup_interface/check_pages.py                    # 檢查三個手部頁面
    python3 hurocup_interface/check_pages.py 某頁.html hc-      # 檢查指定頁面

它攔不到的是執行期的邏輯錯誤 —— 那種只能把頁面真的打開才會知道。
"""

import io, re, sys

REGEX_PREV = set('(,=:[!&|?{};+-*%~^<>') | {'\n'}


def balance(js):
    """括號與字串平衡。會跳過註解，並用前一個有意義字元判斷 / 是除法還是 regex。"""
    stack, errs = [], []
    i, n, line = 0, len(js), 1
    pairs = {')': '(', ']': '[', '}': '{'}
    prev = '\n'                      # 前一個非空白、非註解的字元
    while i < n:
        c = js[i]
        if c == '\n':
            line += 1; i += 1; continue
        if c in ' \t':
            i += 1; continue
        if js.startswith('//', i):
            j = js.find('\n', i); i = n if j < 0 else j; continue
        if js.startswith('/*', i):
            j = js.find('*/', i + 2)
            if j < 0:
                errs.append(f'未關閉區塊註解 @{line}'); break
            line += js.count('\n', i, j); i = j + 2; continue
        # 正規表示式字面值：/ 出現在運算子 / 開括號 / return 之後
        if c == '/' and (prev in REGEX_PREV or re.search(r'\breturn\s*$', js[:i])):
            j, inclass = i + 1, False
            while j < n:
                if js[j] == '\\': j += 2; continue
                if js[j] == '[': inclass = True
                elif js[j] == ']': inclass = False
                elif js[j] == '/' and not inclass: break
                elif js[j] == '\n': break
                j += 1
            if j < n and js[j] == '/':
                i = j + 1
                while i < n and js[i].isalpha():   # 旗標 g i m s u y
                    i += 1
                prev = '/'
                continue
        if c in '"\'`':
            q, j = c, i + 1
            while j < n:
                if js[j] == '\\': j += 2; continue
                if js[j] == q: break
                if js[j] == '\n' and q != '`':
                    errs.append(f'字串跨行 @{line}'); break
                j += 1
            line += js.count('\n', i, j); i = j + 1; prev = q
            continue
        if c in '([{':
            stack.append((c, line))
        elif c in ')]}':
            if not stack or stack[-1][0] != pairs[c]:
                errs.append(f'括號不配對 {c!r} @{line}'); break
            stack.pop()
        prev = c
        i += 1
    if stack:
        errs.append(f'未關閉 {stack[-1][0]!r} @{stack[-1][1]}')
    return errs


def strip_js(js):
    """把註解、字串、正規表示式字面值換成空白，只留下真正的程式碼。

    不先剝掉的話，註解裡提到的 finger_move()、字串裡的 url(...) / rgb(...)
    都會被當成函式呼叫，整個檢查就廢了。
    """
    out = []
    i, n, prev = 0, len(js), '\n'
    while i < n:
        c = js[i]
        if js.startswith('//', i):
            j = js.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j; continue
        if js.startswith('/*', i):
            j = js.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(ch if ch == '\n' else ' ' for ch in js[i:j]))
            i = j; continue
        if c == '/' and (prev in REGEX_PREV or re.search(r'\breturn\s*$', js[:i])):
            j, incls = i + 1, False
            while j < n:
                if js[j] == '\\': j += 2; continue
                if js[j] == '[': incls = True
                elif js[j] == ']': incls = False
                elif js[j] == '/' and not incls: break
                elif js[j] == '\n': break
                j += 1
            if j < n and js[j] == '/':
                j += 1
                while j < n and js[j].isalpha(): j += 1
                out.append(' ' * (j - i)); i = j; prev = '/'; continue
        if c in '"\'`':
            q, j = c, i + 1
            while j < n:
                if js[j] == '\\': j += 2; continue
                if js[j] == q: break
                j += 1
            j = min(j + 1, n)
            out.append(''.join(ch if ch == '\n' else ' ' for ch in js[i:j]))
            i = j; prev = q; continue
        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return ''.join(out)


def check(path, prefix):
    src = io.open(path, encoding='utf-8').read()
    # <script> 與 <style> 的屬性寫法各頁不一致（有的帶 type 有的沒有），都要吃
    jm = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', src, re.S)
    js = '\n'.join(jm)
    cm = re.findall(r'<style[^>]*>(.*?)</style>', src, re.S)
    # 連 <link> 進來的 CSS 檔也要算，不然 style.css 裡的 class 會被誤報成沒定義
    for href in re.findall(r'<link[^>]*href="([^"]+\.css)"', src):
        try:
            cm.append(io.open(href.lstrip('./'), encoding='utf-8').read())
        except OSError:
            print(f'  （讀不到 {href}，略過）')
    css = '\n'.join(cm)

    # <script src="..."> 連進來的本地 JS（nav.js / hand_presets.js）定義的函式
    # 也要算已定義，否則會被誤報成「呼叫但沒定義」。
    # 只收函式定義，不丟進 js 一起做括號平衡檢查 —— 那些檔案自己該各別檢查，
    # 混在一起的話出錯時指不出是哪一個檔案的問題。
    ext_fn = set()
    for srcpath in re.findall(r'<script[^>]*\bsrc="([^"]+\.js)"', src):
        if srcpath.startswith(('http://', 'https://', '//')):
            continue          # CDN 的（roslib 之類）抓不到也不該抓
        try:
            ext = io.open(srcpath.lstrip('./'), encoding='utf-8').read()
        except OSError:
            print(f'  （讀不到 {srcpath}，略過）')
            continue
        ext_fn |= set(re.findall(r'function\s+([A-Za-z_$][\w$]*)', ext))
        ext_fn |= set(re.findall(r'(?:window\.)?([A-Za-z_$][\w$]*)\s*=\s*function', ext))

    ok = True
    print(f'===== {path} =====')

    errs = balance(js)
    print('  JS 括號/字串:', '✓' if not errs else '✗ ' + '; '.join(errs)); ok &= not errs

    handlers = set(re.findall(r'on(?:click|change|input)="(\w+)\(', src))
    defined = set(re.findall(r'function\s+(\w+)\s*\(', js))
    missing = sorted(handlers - defined)
    print(f'  事件處理函式 ({len(handlers)}):', '✓' if not missing else '✗ 缺 ' + str(missing))
    ok &= not missing

    ids = set(re.findall(r'\bid="([^"]+)"', src))
    lit = set(re.findall(r"\bel\('([^']+)'\)", js)) | set(
        re.findall(r"getElementById\('([^']+)'\)", js))
    bad = sorted(x for x in lit if x not in ids)
    print('  寫死 element id:', '✓ 全部存在' if not bad else '✗ ' + str(bad)); ok &= not bad

    # class 只從 class="..." 屬性（含 JS 產生的 HTML 字串）與 className 賦值取
    # 只收合法的 CSS class 名稱 —— JS 裡的 class="a' + (x ? ' b' : '') + '" 這種
    # 字串拼接會讓 regex 抓到帶引號的碎片，要濾掉
    valid = re.compile(r'^[A-Za-z_][\w-]*$')
    cls = set()
    for m in (re.findall(r'class="([^"{}<>]+)"', src)
              + re.findall(r"className = '([^']+)'", js)):
        # class="a' + x + '" 這種 JS 字串拼接會被 regex 一路吃到下一個引號，
        # 切出來的碎片不是真的 class，整段跳過
        if "'" in m or '+' in m:
            continue
        cls.update(w for w in m.split() if valid.match(w))
    # prefix 現在只是標示用；實際檢查所有用到的 class
    miss = sorted(c for c in cls if ('.' + c) not in css)
    print(f'  class 都有 CSS ({len(cls)}):', '✓' if not miss else '✗ 沒定義: ' + str(miss))
    ok &= not miss

    # 呼叫了卻沒有定義的函式（logId 這種「改版後殘留的呼叫」會是執行期
    # ReferenceError，前面幾項檢查都抓不到）
    defined_fn = set(re.findall(r'function\s+([A-Za-z_$][\w$]*)', js))
    defined_fn |= set(re.findall(r'(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=', js))
    defined_fn |= set(re.findall(r'([A-Za-z_$][\w$]*)\s*:\s*function', js))
    # 函式參數也算已定義（callback 常常以參數的形式被呼叫，例如 function arr(a, warn)）
    for params in re.findall(r'function[^(]*\(([^)]*)\)', strip_js(js)):
        defined_fn |= {w.strip() for w in params.split(',') if w.strip()}
    KNOWN = {
        'if', 'for', 'while', 'switch', 'catch', 'return', 'typeof', 'function',
        'new', 'else', 'do', 'in', 'of', 'delete', 'void', 'instanceof',
        'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'String', 'Number',
        'Boolean', 'Array', 'Object', 'Math', 'JSON', 'Date', 'RegExp', 'Error',
        'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'alert',
        'console', 'encodeURIComponent', 'decodeURIComponent', 'requestAnimationFrame',
        'ROSLIB', 'WebSocket', 'Image', 'Blob', 'URL',
    }
    code = strip_js(js)
    called = set(re.findall(r'(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(', code))
    undef = sorted(c for c in called
                   if c not in defined_fn and c not in KNOWN and c not in ext_fn)
    print('  呼叫但沒定義的函式:', '✓ 無' if not undef else '✗ ' + str(undef))
    ok &= not undef

    # 反過來：定義了卻從來沒被呼叫的 function。
    # 這會是「功能寫好了但忘記接線」—— 頁面不會報錯，就只是那塊功能沒反應
    # （目前刻度那塊 buildLive/renderLive/lockButtons 就是這樣整塊啞掉）。
    # onclick="" 那種從 HTML 叫的也算，所以 called 要從整份 src 掃。
    html_called = set(re.findall(r'(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(', src))
    own_fn = set(re.findall(r'function\s+([A-Za-z_$][\w$]*)\s*\(', js))
    # 名字被當成參考值傳出去也算用過（requestAnimationFrame(redraw)、
    # addEventListener('x', fn)），所以數的是「名字出現幾次」而不是「被呼叫幾次」
    dead = sorted(f for f in own_fn
                  if len(re.findall(r'(?<![.\w$])' + re.escape(f) + r'\b', src)) < 2)
    print('  定義了卻沒被呼叫:', '✓ 無' if not dead else '✗ ' + str(dead))
    ok &= not dead

    body = src[src.index('<body'):]
    tagbad = []
    for t in ('div', 'span', 'select', 'table', 'label', 'canvas', 'tr', 'td', 'th'):
        o = len(re.findall(r'<%s\b' % t, body)); c = len(re.findall(r'</%s>' % t, body))
        if o != c:
            tagbad.append(f'{t} {o}/{c}')
    print('  HTML 標籤平衡:', '✓' if not tagbad else '✗ ' + ', '.join(tagbad))
    ok &= not tagbad
    print('  行數:', len(src.splitlines()))
    return ok

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT = [('HandControlInterface.html', 'hc-'),
               ('HandTouchInterface.html', 'ht-'),
               ('HandMotionInterface.html', 'hm-')]
    args = sys.argv[1:]
    if len(args) == 1:
        # 只給檔名也要能用（之前會被忽略，默默檢查預設的三個，很容易誤判）
        known = dict(DEFAULT)
        targets = [(args[0], known.get(args[0], ''))]
    elif args:
        targets = [(args[0], args[1])]
    else:
        targets = DEFAULT
    allok = True
    for p, pre in targets:
        allok &= check(p, pre)
        print()
    print('全部通過' if allok else '有問題')
    sys.exit(0 if allok else 1)
