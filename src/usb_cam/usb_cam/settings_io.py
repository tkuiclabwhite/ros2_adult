"""相機設定檔 I/O：location 解析 + CameraSet.ini 讀寫。

提供三個對外函式：
  - load_strategy_location() : 讀 strategy/strategy/strategy.ini 取得 location
  - load_camera_set()         : 從 CameraSet.ini 讀回 b/c/s/wb/awb/ae 設定值
  - save_camera_set()         : 把目前 CameraSet 寫回 INI

路徑搜尋規則：
  - strategy_root 解析優先序：tku_STRATEGY_ROOT 環境變數 →
    ~/ros2_adult/src/strategy/strategy → ~/workspace/ros2_adult/src/strategy/strategy
    → 由 cwd 往上找
  - 寫檔時若 primary (strategy_root/<location>/CameraSet.ini) 失敗，
    依序 fallback 到 save_dir/CameraSet.ini 與 ~/.ros/usb_cam/config/CameraSet.ini
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# CameraSet 資料模型：對應 INI 內六個欄位 + 寫檔用的旗標
# ---------------------------------------------------------------------------
@dataclass
class CameraSet:
    """``CameraSet.ini`` 對應的設定值集合。"""
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    white_balance: int = 0
    auto_white_balance: bool = False
    auto_exposure: bool = False
    zoomin: float = 1.0


# ---------------------------------------------------------------------------
# 路徑解析工具
# ---------------------------------------------------------------------------
def _expand_user(p: str) -> Path:
    """將開頭 ``~/`` 展開成 $HOME 對應的絕對路徑。"""
    if p.startswith("~/"):
        home = os.environ.get("HOME")
        if home:
            return Path(home) / p[2:]
    return Path(p)


def _normalize_location(loc: str) -> str:
    """trim 空白並去除前後 '/'，使 ``/bb/Parameter`` 變成 ``bb/Parameter``。"""
    s = (loc or "").strip()
    # 去除所有開頭的 '/'，避免 Path() 把它視為絕對路徑
    while s.startswith("/"):
        s = s[1:]
    while s.endswith("/"):
        s = s[:-1]
    return s


def find_strategy_root() -> Optional[Path]:
    """找 ``strategy/strategy`` 根目錄，回傳第一個存在的候選。"""
    # 1) 環境變數優先
    env = os.environ.get("tku_STRATEGY_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) 常見 workspace 位置
    home = os.environ.get("HOME")
    if home:
        candidates = [
            Path(home) / "ros2_adult" / "src" / "strategy" / "strategy",
            Path(home) / "workspace" / "ros2_adult" / "src" / "strategy" / "strategy",
        ]
        for c in candidates:
            if c.exists():
                return c

    # 3) 由 cwd 往上找
    cur = Path.cwd()
    while True:
        cand = cur / "ros2_adult" / "src" / "strategy" / "strategy"
        if cand.exists():
            return cand
        if cur.parent == cur:
            return None
        cur = cur.parent


# ---------------------------------------------------------------------------
# location 載入：讀 strategy/strategy/strategy.ini 的單行內容
# ---------------------------------------------------------------------------
def load_strategy_location(logger: Optional[Callable[[str], None]] = None) -> str:
    """讀取 ``strategy/strategy/strategy.ini`` 的內容（單行字串）。

    檔案內容範例：``/bb/Parameter``。回傳 trim 後的字串；找不到檔案時回傳空字串。
    """
    log = logger or (lambda m: print(m, flush=True))
    root = find_strategy_root()
    if root is None:
        log("[settings_io] strategy root not found")
        return ""
    ini_path = root / "strategy.ini"
    if not ini_path.exists():
        log(f"[settings_io] cannot open file: {ini_path}")
        return ""
    try:
        text = ini_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"[settings_io] read error {ini_path}: {e}")
        return ""
    # C++ 版只讀第一行
    # 只取第一行（檔案內僅一行字串）
    line = text.splitlines()[0] if text else ""
    line = line.strip()
    log(f"[settings_io] location_: {line}")
    return line


# ---------------------------------------------------------------------------
# CameraSet.ini 路徑組合
# ---------------------------------------------------------------------------
def _primary_path(location: str) -> Optional[Path]:
    """主路徑：``<strategy_root>/<location>/CameraSet.ini``。"""
    loc_clean = _normalize_location(location)
    if not loc_clean:
        return None
    root = find_strategy_root()
    if root is None:
        return None
    return root / loc_clean / "CameraSet.ini"


def _fallback_path(save_dir: str) -> Optional[Path]:
    """save_dir 對應的 fallback：``<save_dir>/CameraSet.ini``。"""
    if not save_dir:
        return None
    return _expand_user(save_dir) / "CameraSet.ini"


def _last_resort_path() -> Path:
    """最後一道防線：``~/.ros/usb_cam/config/CameraSet.ini``。"""
    home = os.environ.get("HOME")
    base = Path(home) if home else Path(".")
    return base / ".ros" / "usb_cam" / "config" / "CameraSet.ini"


# ---------------------------------------------------------------------------
# 載入 CameraSet：依序嘗試 primary → fallback
# ---------------------------------------------------------------------------
def load_camera_set(
    location: str,
    save_dir: str,
    current: CameraSet,
    logger: Optional[Callable[[str], None]] = None,
) -> CameraSet:
    """讀 ``CameraSet.ini`` 並回傳更新後的 CameraSet。

    讀檔時若某 key 解析失敗只警告、保留原值。primary 開不到時自動 fallback
    到 save_dir。
    """
    log = logger or (lambda m: print(m, flush=True))
    if not _normalize_location(location):
        log("[settings_io] location is empty; skip loading.")
        return current

    primary = _primary_path(location)
    fallback = _fallback_path(save_dir)

    # 依序嘗試開檔
    final_path: Optional[Path] = None
    if primary is not None and primary.exists():
        final_path = primary
        log(f"[settings_io] Using strategy path: {final_path}")
    elif fallback is not None and fallback.exists():
        final_path = fallback
        log(
            "[settings_io] Primary missing/unreadable. "
            f"Fallback to save_dir: {final_path}"
        )
    else:
        log(
            "[settings_io] Cannot open CameraSet.ini. Tried: "
            f"{primary or '(none)'}"
            + (f" and {fallback}" if fallback else "")
        )
        return current

    # 用 configparser 讀整個 INI
    parser = configparser.ConfigParser()
    try:
        parser.read(final_path, encoding="utf-8")
    except (OSError, configparser.Error) as e:
        log(f"[settings_io] Read error: {e}")
        return current

    # configparser 會把 key 歸到 sections，這裡掃所有 section 的 key=value
    out = CameraSet(
        brightness=current.brightness,
        contrast=current.contrast,
        saturation=current.saturation,
        white_balance=current.white_balance,
        auto_white_balance=current.auto_white_balance,
        auto_exposure=current.auto_exposure,
        zoomin=current.zoomin,
    )

    def _parse_int(key: str, value: str):
        try:
            return int(value)
        except ValueError as e:
            log(f"[settings_io] Bad value for '{key}': {e}")
            return None

    for section in parser.sections():
        for key, value in parser.items(section):
            value = value.strip()
            if key == "brightness":
                v = _parse_int(key, value)
                if v is not None:
                    out.brightness = v
            elif key == "contrast":
                v = _parse_int(key, value)
                if v is not None:
                    out.contrast = v
            elif key == "saturation":
                v = _parse_int(key, value)
                if v is not None:
                    out.saturation = v
            elif key == "white_balance":
                v = _parse_int(key, value)
                if v is not None:
                    out.white_balance = v
            elif key == "auto_white_balance":
                v = _parse_int(key, value)
                if v is not None:
                    out.auto_white_balance = bool(v)
            elif key == "auto_exposure":
                v = _parse_int(key, value)
                if v is not None:
                    out.auto_exposure = bool(v)
            elif key == "zoomin":
                try:
                    out.zoomin = float(value)
                except ValueError as e:
                    log(f"[settings_io] Bad value for 'zoomin': {e}")
            elif key == "auto_backlight_compensation":
                # 寫檔端不一定產生此欄位、讀端也不使用，靜默忽略避免警告刷屏
                pass
            else:
                log(f"[settings_io] Unknown key: {key}")

    log("Load_Set")
    return out


# ---------------------------------------------------------------------------
# 寫檔：依序嘗試 primary → fallback(save_dir) → last_resort
# ---------------------------------------------------------------------------
def save_camera_set(
    cs: CameraSet,
    location: str,
    save_dir: str,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """把 CameraSet 寫入 INI 檔案。

    寫入順序：primary (strategy_root/<location>/CameraSet.ini) → save_dir/
    CameraSet.ini → ~/.ros/usb_cam/config/CameraSet.ini。回傳實際成功寫入
    的路徑；全部失敗回傳 None。
    """
    log = logger or (lambda m: print(m, flush=True))
    primary = _primary_path(location)
    fallback = _fallback_path(save_dir)
    last_resort = _last_resort_path()

    def _try_write(path: Optional[Path]) -> bool:
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                # INI 格式：含區段標頭，便於後續以 configparser 讀回
                f.write("[Camera Set Parameter]\n")
                f.write(f"brightness = {cs.brightness}\n")
                f.write(f"contrast = {cs.contrast}\n")
                f.write(f"saturation = {cs.saturation}\n")
                f.write(f"white_balance = {cs.white_balance}\n")
                f.write(
                    f"auto_white_balance = {1 if cs.auto_white_balance else 0}\n"
                )
                f.write(
                    f"auto_exposure = {1 if cs.auto_exposure else 0}\n"
                )
                f.write(f"zoomin = {cs.zoomin:.1f}\n")
            log(f"[camera_save] Saved to: {path}")
            return True
        except OSError as e:
            log(f"[camera_save] Write error to {path}: {e}")
            return False

    if _try_write(primary):
        return primary
    log(f"[camera_save] Primary failed: {primary or '(none)'}")

    if _try_write(fallback):
        return fallback
    log(f"[camera_save] Fallback(save_dir) failed: {fallback or '(none)'}")

    if _try_write(last_resort):
        return last_resort
    log(f"[camera_save] All write attempts failed. Last tried: {last_resort}")
    return None
