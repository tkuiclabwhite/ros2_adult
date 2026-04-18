#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
import re
from typing import Optional, Tuple

try:
    import serial
except Exception:
    serial = None  # 讓匯入失敗時也不當場崩潰，run() 會提示


class IMUService:
    """
    串口讀取 IMU 的 YPR = yaw, pitch, roll (deg)
    - 以獨立執行緒持續讀取
    - 帶緩衝逐行解析：避免半行丟失
    - 放寬格式：允許 "YPR=...", "YPR : ...", 大小寫、前導 '#'
    - rel_mode=True 時，latest() 會回傳相對 zero 的 (y, p, r)

    針對 Arduino SerialUSB(/dev/imu) 常見問題：
    - 串口開啟會 reset，增加 open 後 sleep 2 秒
    - 行尾可能是 \r\n 或只有 \r，統一把 \r 轉成 \n 再拆行
    """

    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baud: int = 115200,
        rel_mode: bool = True,
        debug_raw: bool = False,
        open_wait_sec: float = 2.0,
    ):
        self.port = port
        self.baud = baud
        self.rel_mode = rel_mode
        self.debug_raw = debug_raw
        self.open_wait_sec = open_wait_sec

        self._ser = None
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest_abs: Optional[Tuple[float, float, float]] = None
        self._has_data = False
        self.zero = [0.0, 0.0, 0.0]

    # ---------------- public API ----------------
    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self.run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.5)
        self._close_serial()

    def latest(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            if not self._has_data or self._latest_abs is None:
                return None
            # 這裡不做額外減法計算，因為 Arduino 已經幫你算好相對角度了
            return self._latest_abs

    def zero_here(self):
        """
        將目前絕對角度設定為零點（只影響 rel_mode 輸出）
        """
        with self._lock:
            if self._ser and self._ser.is_open:
                try:
                    # 根據你的 Arduino 程式碼，發送空白字元觸發歸零
                    self._ser.write(b' ')
                    self._ser.flush()
                    print("[IMU] 已發送空白鍵執行 Arduino 硬體歸零")
                except Exception as e:
                    print(f"[IMU] 發送歸零指令失敗: {e}")
            else:
                print("[IMU] 串口未開啟，無法傳送歸零指令")
    # ---------------- internal helpers ----------------
    def _open_serial(self):
        if serial is None:
            print("[IMU] pyserial 未安裝：pip install pyserial")
            return

        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
            print(f"[IMU] Serial opened at {self.port} @ {self.baud}")

            # Arduino SerialUSB 常見：一開 port 會 reset，需要等它 boot + DMP ready
            if self.open_wait_sec and self.open_wait_sec > 0:
                time.sleep(self.open_wait_sec)

            # 清緩衝
            try:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
            except Exception:
                pass

            # ⚠️ 不送任何喚醒指令：你的 Arduino 程式是單向輸出，不需要 write
            # （寫入反而可能在 reset 期間造成干擾）
        except Exception as e:
            self._ser = None
            print(f"[IMU] open serial failed on {self.port}: {e}")

    def _close_serial(self):
        try:
            if self._ser and getattr(self._ser, "is_open", False):
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    # ---------------- main loop ----------------
    def run(self):
        """
        讀取回圈（帶緩衝、逐行解析）
        接受格式（大小寫皆可）：
          YPR= y, p, r
          YPR : y, p, r
          #YPR=...
        """
        self._open_serial()
        last_hint = time.time()
        buf = ""  # 持續性緩衝：避免半行被丟掉

        # 放寬匹配：ypr[:=] num, num, num
        ypr_regex = re.compile(
            r'^\s*#?\s*ypr\s*[:=]\s*'         # "ypr" 後面冒號或等號
            r'([-+]?\d+(?:\.\d+)?)\s*,\s*'    # yaw
            r'([-+]?\d+(?:\.\d+)?)\s*,\s*'    # pitch
            r'([-+]?\d+(?:\.\d+)?)\s*$',      # roll
            re.IGNORECASE
        )

        while not self._stop.is_set():
            try:
                # 嘗試重連
                if self._ser is None or not getattr(self._ser, "is_open", False):
                    self._open_serial()
                    time.sleep(0.2)
                    continue

                n = self._ser.in_waiting if hasattr(self._ser, "in_waiting") else 0
                if n:
                    raw = self._ser.read(n)
                    try:
                        buf += raw.decode(errors="ignore")
                    except Exception:
                        pass

                    # ✅ 關鍵：把 CR 統一成 LF，避免只送 '\r' 時永遠拆不到行
                    buf = buf.replace('\r', '\n')

                    # 逐行處理（保留最後一段殘行）
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        s = line.strip()
                        if not s:
                            continue

                        m = ypr_regex.match(s)
                        if not m:
                            if self.debug_raw:
                                print("[IMU] raw:", repr(s))
                            continue

                        y = float(m.group(1))
                        p = float(m.group(2))
                        r = float(m.group(3))

                        with self._lock:
                            self._latest_abs = (y, p, r)
                            self._has_data = True

                # 每 2 秒提示一次沒有資料
                if (time.time() - last_hint) > 2.0 and not self._has_data:
                    print("[IMU] no YPR in 2s — 請檢查 baud/port/資料格式（期待類似 '#YPR=1,2,3'）")
                    last_hint = time.time()

                time.sleep(0.001)

            except Exception as e:
                print(f"[IMU] read err: {e}")
                time.sleep(0.1)