#!/usr/bin/env python3
"""靜態網頁伺服器：一般檔案從本目錄（hurocup_interface）出，
但 /strategy.js 直接即時讀取 src/strategy/strategy/strategy.js 的內容再回傳，
這個檔案的內容會一直變動，複製一份或用 symlink 都會遇到內容過期/傳輸不保留
symlink 的問題，改成每次請求都重新讀取最新內容就沒有這個問題。
"""
import http.server
import os
import sys

HUROCUP_DIR = os.path.dirname(os.path.realpath(__file__))
STRATEGY_JS_PATH = os.path.join(HUROCUP_DIR, '..', 'src', 'strategy', 'strategy', 'strategy.js')


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HUROCUP_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/strategy.js':
            self.serve_strategy_js()
            return
        super().do_GET()

    def serve_strategy_js(self):
        try:
            with open(STRATEGY_JS_PATH, 'rb') as f:
                content = f.read()
        except OSError:
            self.send_error(404, 'strategy.js not found')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/javascript')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)


class QuietThreadingHTTPServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # 手機瀏覽器切換/斷開 MJPEG 串流時常常會中途重置連線，這是預期行為，
        # 不用印一整段 traceback 嚇人，其他非預期的錯誤還是照樣印出來。
        exc_type = sys.exc_info()[0]
        if exc_type in (ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)


if __name__ == '__main__':
    server = QuietThreadingHTTPServer(('0.0.0.0', 9999), Handler)
    server.serve_forever()
