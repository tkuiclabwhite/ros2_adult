#!/usr/bin/env python3
"""
WiFi 管理工具 (TUI) - 透過 SSH 連線也能操作的網路設定介面
基於 nmcli (NetworkManager) + textual

使用方式:
    python3 wifi_manager.py

快捷鍵:
    Enter   連線到選中的網路
    d       斷開選中網路的連線 (若該網路正在使用中)
    f       忘記 (刪除) 選中網路的已儲存設定
    i       顯示選中網路的詳細資訊
    r       重新掃描 WiFi
    s       切換排序方式 (Signal / SSID / Security / Saved)
    w       開關 WiFi 無線電
    /       搜尋 SSID (Esc 清除)
    q       離開

架構說明 (方便之後維護):
    1. nmcli 操作層：所有跟系統互動的指令集中在檔案前半段的函式中
    2. UI 元件層：ModalScreen 子類別，負責各種彈出視窗
    3. WifiManagerApp：主程式，負責畫面組裝跟事件調度，實際工作都丟給
       asyncio.to_thread() 在背景執行，避免卡住介面
"""

import asyncio
import subprocess
from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    Button,
    LoadingIndicator,
)
from textual.binding import Binding


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------

@dataclass
class WifiNetwork:
    ssid: str               # 隱藏網路時為空字串
    bssid: str
    channel: str
    freq_mhz: int
    rate: str
    signal: int
    security: str
    in_use: bool
    saved: bool = False

    @property
    def display_name(self) -> str:
        return self.ssid if self.ssid else "<Hidden Network>"

    @property
    def is_hidden(self) -> bool:
        return self.ssid == ""

    @property
    def is_open(self) -> bool:
        return self.security in ("", "--")


SORT_MODES = ["Signal", "SSID", "Security", "Saved"]
SIGNAL_BARS = "▁▂▃▄▅▆▇█"
SPINNER_LABEL = "處理中"


# ---------------------------------------------------------------------------
# nmcli 操作層：所有跟系統互動的指令都集中在這裡，方便之後維護/替換
# ---------------------------------------------------------------------------

def run_nmcli(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    """同步執行 nmcli 指令 (在背景 thread 呼叫，避免卡住 UI)；失敗時回傳一個假的
    CompletedProcess，讓呼叫端不用額外處理例外，只需檢查 returncode。

    前面加 sudo 是因為部分系統的 polkit rules.d 沒有生效（精簡編譯版本可能拿掉了
    JS rules 引擎支援），改用 /etc/sudoers.d/ 設定的 NOPASSWD 規則讓 nmcli 免密碼執行。
    """
    try:
        return subprocess.run(
            ["sudo", "nmcli"] + args,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr=str(exc))


def _unescape_nmcli_field(field_text: str) -> str:
    """nmcli -t 輸出中，冒號會被轉成 \\: ，這裡還原回正常字串"""
    return field_text.replace("\\:", ":")


def _split_nmcli_line(line: str) -> list[str]:
    """依照 nmcli -t 的規則，用未被反斜線跳脫的冒號切割欄位"""
    fields = []
    current = ""
    escape = False
    for char in line:
        if escape:
            current += char
            escape = False
        elif char == "\\":
            escape = True
            current += char
        elif char == ":":
            fields.append(_unescape_nmcli_field(current))
            current = ""
        else:
            current += char
    fields.append(_unescape_nmcli_field(current))
    return fields


def freq_to_band(freq_mhz: int) -> str:
    """把頻率 (MHz) 轉成方便閱讀的頻段標籤"""
    if freq_mhz <= 0:
        return "--"
    if freq_mhz < 3000:
        return "2.4 GHz"
    if freq_mhz < 5925:
        return "5 GHz"
    return "6 GHz"


def signal_bar(percent: int) -> str:
    """把訊號百分比轉成 8 級柱狀圖示"""
    if percent <= 0:
        return SIGNAL_BARS[0]
    idx = min(percent // 13, len(SIGNAL_BARS) - 1)
    return SIGNAL_BARS[idx]


def get_saved_wifi_names() -> set[str]:
    """取得所有已儲存的 WiFi connection profile 的 SSID（不是 profile 名稱）
    因為 profile 名稱跟 SSID 不一定相同，用 SSID 做比對才準確"""
    result = run_nmcli(["-t", "-f", "NAME,TYPE", "connection", "show"])
    saved_ssids: set[str] = set()
    if result.returncode != 0:
        return saved_ssids

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        fields = _split_nmcli_line(line)
        if len(fields) >= 2 and fields[1] == "802-11-wireless":
            profile_name = fields[0]
            # 取得這個 profile 實際儲存的 SSID
            detail = run_nmcli(["connection", "show", profile_name])
            if detail.returncode != 0:
                continue
            for detail_line in detail.stdout.splitlines():
                if "802-11-wireless.ssid:" in detail_line:
                    ssid = detail_line.split("802-11-wireless.ssid:")[-1].strip()
                    if ssid and ssid != "--":
                        saved_ssids.add(ssid)
                    break

    return saved_ssids


def get_profile_name_for_ssid(ssid: str) -> str | None:
    """根據 SSID 找到對應的 connection profile 名稱
    用於 connect_saved_wifi 跟 delete_connection（這兩個需要 profile 名稱，不是 SSID）"""
    result = run_nmcli(["-t", "-f", "NAME,TYPE", "connection", "show"])
    if result.returncode != 0:
        return None

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        fields = _split_nmcli_line(line)
        if len(fields) >= 2 and fields[1] == "802-11-wireless":
            profile_name = fields[0]
            detail = run_nmcli(["connection", "show", profile_name])
            if detail.returncode != 0:
                continue
            for detail_line in detail.stdout.splitlines():
                if "802-11-wireless.ssid:" in detail_line:
                    stored_ssid = detail_line.split("802-11-wireless.ssid:")[-1].strip()
                    if stored_ssid == ssid:
                        return profile_name
                    break
    return None


def delete_connection(name: str) -> tuple[bool, str]:
    """刪除已儲存的 connection profile (Forget Network)
    name 可以是 SSID 或 profile 名稱，先嘗試用 SSID 找 profile 名稱"""
    profile_name = get_profile_name_for_ssid(name) or name
    result = run_nmcli(["connection", "delete", profile_name])
    if result.returncode == 0:
        return True, f"已忘記網路：{name}"
    return False, result.stderr.strip() or "刪除失敗"


def get_wifi_radio_status() -> bool:
    """回傳 WiFi 無線電是否開啟"""
    result = run_nmcli(["radio", "wifi"])
    return result.stdout.strip() == "enabled"


def set_wifi_radio(enable: bool) -> tuple[bool, str]:
    result = run_nmcli(["radio", "wifi", "on" if enable else "off"])
    if result.returncode == 0:
        return True, f"WiFi 已{'開啟' if enable else '關閉'}"
    return False, result.stderr.strip() or "操作失敗"


def get_active_wifi_device() -> str | None:
    """取得目前連線中的 WiFi 裝置名稱 (例如 wlan0)，沒有連線則回傳 None"""
    result = run_nmcli(["-t", "-f", "DEVICE,TYPE,STATE", "device", "status"])
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        fields = _split_nmcli_line(line)
        if len(fields) >= 3 and fields[1] == "wifi" and fields[2].startswith("connected"):
            return fields[0]
    return None


def get_ip_info() -> dict:
    """取得目前 WiFi 連線的 IP 資訊；沒有連線則回傳空字典"""
    device = get_active_wifi_device()
    if not device:
        return {}

    result = run_nmcli(["-t", "device", "show", device])
    if result.returncode != 0:
        return {}

    raw: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        raw[key] = value

    ipv4 = raw.get("IP4.ADDRESS[1]", "").split("/")[0]
    gateway = raw.get("IP4.GATEWAY", "")
    dns_list = [v for k, v in raw.items() if k.startswith("IP4.DNS") and v]
    ipv6 = raw.get("IP6.ADDRESS[1]", "").split("/")[0]

    return {
        "ipv4": ipv4,
        "gateway": gateway,
        "dns": ", ".join(dns_list),
        "ipv6": ipv6,
    }


def scan_wifi(saved_names: set[str] | None = None) -> list[WifiNetwork]:
    """掃描並回傳目前可見的 WiFi 清單"""
    saved_names = saved_names or set()

    run_nmcli(["device", "wifi", "rescan"])
    result = run_nmcli(
        ["-t", "-f", "SSID,BSSID,CHAN,FREQ,RATE,SIGNAL,SECURITY,IN-USE", "device", "wifi", "list"]
    )
    if result.returncode != 0:
        return []

    networks: dict[str, WifiNetwork] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        fields = _split_nmcli_line(line)
        if len(fields) < 8:
            continue
        ssid, bssid, chan, freq_text, rate, signal, security, in_use = fields[:8]

        try:
            signal_val = int(signal)
        except ValueError:
            signal_val = 0
        try:
            freq_val = int(freq_text.split()[0]) if freq_text else 0
        except (ValueError, IndexError):
            freq_val = 0

        # 隱藏網路 (SSID 空白) 每個 BSSID 各自獨立顯示；
        # 一般網路用 SSID 當 key，多個基地台時只保留訊號最強的一筆
        key = ssid if ssid else f"__hidden__{bssid}"

        net = WifiNetwork(
            ssid=ssid,
            bssid=bssid,
            channel=chan,
            freq_mhz=freq_val,
            rate=rate or "--",
            signal=signal_val,
            security=security or "--",
            in_use=(in_use.strip() == "*"),
            saved=(ssid in saved_names) if ssid else False,
        )

        existing = networks.get(key)
        if existing is None or net.signal > existing.signal:
            networks[key] = net

    return list(networks.values())


def connect_wifi(ssid: str, password: str | None, hidden: bool = False) -> tuple[bool, str]:
    """連線到指定 WiFi，回傳 (是否成功, 訊息)"""
    args = ["device", "wifi", "connect", ssid]
    if password:
        args += ["password", password]
    if hidden:
        args += ["hidden", "yes"]
    result = run_nmcli(args)
    if result.returncode == 0:
        return True, f"已連線到 {ssid}"
    return False, result.stderr.strip() or result.stdout.strip() or "連線失敗"


def connect_saved_wifi(ssid: str) -> tuple[bool, str]:
    """直接用已儲存的 connection profile 連線，不需要重新輸入密碼
    先找到對應的 profile 名稱，再用 connection up 啟動"""
    profile_name = get_profile_name_for_ssid(ssid)
    if profile_name is None:
        # 找不到 profile，fallback 到直接用 SSID 嘗試
        profile_name = ssid
    result = run_nmcli(["connection", "up", profile_name])
    if result.returncode == 0:
        return True, f"已連線到 {ssid}"
    return False, result.stderr.strip() or result.stdout.strip() or "連線失敗"


def disconnect_wifi(ssid: str) -> tuple[bool, str]:
    """斷開指定 SSID 的連線"""
    result = run_nmcli(["connection", "down", ssid])
    if result.returncode == 0:
        return True, f"已斷開 {ssid}"
    return False, result.stderr.strip() or "斷開失敗"


def looks_like_auth_error(message: str) -> bool:
    """粗略判斷連線失敗訊息是不是密碼/驗證問題，用於決定要不要跳回密碼輸入框"""
    keywords = ("secrets", "802-1x", "key-mgmt", "auth", "psk", "password")
    lowered = message.lower()
    return any(word in lowered for word in keywords)


# ---------------------------------------------------------------------------
# UI 元件：彈出視窗
# ---------------------------------------------------------------------------

class PasswordModal(ModalScreen[str | None]):
    """彈出視窗：輸入 WiFi 密碼。error_message 用於密碼錯誤重試時顯示提示。"""

    DEFAULT_CSS = """
    PasswordModal { align: center middle; }
    PasswordModal > Container {
        width: 50; height: auto;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    PasswordModal Static.error { color: $error; margin-bottom: 1; }
    PasswordModal #buttons { width: 100%; align-horizontal: right; margin-top: 1; }
    PasswordModal Button { margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, ssid: str, error_message: str | None = None) -> None:
        super().__init__()
        self.ssid = ssid
        self.error_message = error_message

    def compose(self) -> ComposeResult:
        with Container():
            if self.error_message:
                yield Static(f"❌ {self.error_message}", classes="error")
            yield Static(f"🔒 輸入「{self.ssid}」的密碼")
            yield Input(placeholder="password", password=True, id="pw_input")
            with Horizontal(id="buttons"):
                yield Button("取消", id="cancel_btn")
                yield Button("連線", id="connect_btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#pw_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect_btn":
            self.dismiss(self.query_one("#pw_input", Input).value)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HiddenSSIDModal(ModalScreen[str | None]):
    """彈出視窗：輸入隱藏網路的 SSID"""

    DEFAULT_CSS = """
    HiddenSSIDModal { align: center middle; }
    HiddenSSIDModal > Container {
        width: 50; height: auto;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    HiddenSSIDModal #buttons { width: 100%; align-horizontal: right; margin-top: 1; }
    HiddenSSIDModal Button { margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("👻 輸入隱藏網路的 SSID")
            yield Input(placeholder="SSID", id="ssid_input")
            with Horizontal(id="buttons"):
                yield Button("取消", id="cancel_btn")
                yield Button("下一步", id="next_btn", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#ssid_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next_btn":
            self.dismiss(self.query_one("#ssid_input", Input).value or None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """通用確認視窗 (目前用於 Forget Network)"""

    DEFAULT_CSS = """
    ConfirmModal { align: center middle; }
    ConfirmModal > Container {
        width: 50; height: auto;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    ConfirmModal #buttons { width: 100%; align-horizontal: right; margin-top: 1; }
    ConfirmModal Button { margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, title: str, target: str) -> None:
        super().__init__()
        self.title_text = title
        self.target = target

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(self.title_text)
            yield Static(f"[b]{self.target}[/b]")
            with Horizontal(id="buttons"):
                yield Button("取消", id="cancel_btn")
                yield Button("刪除", id="confirm_btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm_btn")

    def action_cancel(self) -> None:
        self.dismiss(False)


class InfoModal(ModalScreen[None]):
    """WiFi 詳細資訊視窗"""

    DEFAULT_CSS = """
    InfoModal { align: center middle; }
    InfoModal > Container {
        width: 56; height: auto;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    InfoModal #close_row { width: 100%; align-horizontal: right; margin-top: 1; }
    """

    BINDINGS = [Binding("escape", "close", "關閉"), Binding("enter", "close", "關閉")]

    def __init__(self, net: WifiNetwork) -> None:
        super().__init__()
        self.net = net

    def compose(self) -> ComposeResult:
        n = self.net
        lines = [
            f"SSID       : {n.display_name}",
            f"BSSID      : {n.bssid or '--'}",
            f"Signal     : {n.signal}%",
            f"Security   : {n.security}",
            f"Frequency  : {n.freq_mhz} MHz ({freq_to_band(n.freq_mhz)})",
            f"Channel    : {n.channel or '--'}",
            f"Rate       : {n.rate}",
            f"已儲存設定  : {'是' if n.saved else '否'}",
            f"目前使用中  : {'是' if n.in_use else '否'}",
        ]
        with Container():
            yield Static("📶 WiFi 詳細資訊", classes="title")
            yield Static("\n".join(lines))
            with Horizontal(id="close_row"):
                yield Button("關閉", id="close_btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

class WifiManagerApp(App):
    """主應用程式"""

    TITLE = "WiFi Manager"

    CSS = """
    #top_status {
        height: 4;
        background: $primary-darken-1;
        color: $text;
        padding: 0 2;
    }
    #top_status.connected { background: $success-darken-2; }
    #loading_row {
        height: 1;
        padding: 0 2;
    }
    #loading_row.hidden { display: none; }
    #search_row { height: 3; padding: 0 1; }
    #search_row.hidden { display: none; }
    #search_row Static { width: auto; height: 3; content-align: left middle; padding-left: 1; }
    #search_row Input { width: 1fr; }
    #toolbar {
        height: 3;
        padding: 0 1;
        background: $panel;
    }
    #toolbar Button { margin-right: 1; min-width: 10; }
    DataTable { height: 1fr; }
    .title { text-style: bold; margin-bottom: 1; }
    """

    BINDINGS = [
        Binding("r", "rescan", "重新掃描"),
        Binding("d", "disconnect", "斷開"),
        Binding("f", "forget", "忘記"),
        Binding("i", "info", "詳細資訊"),
        Binding("s", "cycle_sort", "排序"),
        Binding("w", "toggle_radio", "WiFi 開關"),
        Binding("slash", "search", "搜尋", key_display="/"),
        Binding("escape", "clear_search", "清除搜尋", show=False),
        Binding("q", "quit", "離開"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.networks: list[WifiNetwork] = []
        self.displayed: list[WifiNetwork] = []
        self.sort_index = 0
        self.search_query = ""
        self.wifi_radio_on = True
        self.busy = False
        self._search_debounce_timer = None

    # --- 畫面組裝 -----------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="top_status")
        with Horizontal(id="loading_row", classes="hidden"):
            yield LoadingIndicator()
            yield Static(SPINNER_LABEL, id="loading_label")
        with Horizontal(id="search_row", classes="hidden"):
            yield Static("Search: ", id="search_label")
            yield Input(id="search_input")
        with Horizontal(id="toolbar"):
            yield Button("連線", id="btn_connect", variant="primary")
            yield Button("斷開", id="btn_disconnect")
            yield Button("忘記", id="btn_forget")
            yield Button("詳細資訊", id="btn_info")
            yield Button("重新掃描", id="btn_rescan")
            yield Button("排序", id="btn_sort")
            yield Button("WiFi 開關", id="btn_radio")
            yield Button("搜尋", id="btn_search")
            yield Button("離開", id="btn_quit", variant="error")
        yield DataTable(id="wifi_table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#wifi_table", DataTable)
        table.add_columns("★", "●", "SSID", "Band", "Signal", "Security")
        self.run_worker(self._initial_load(), exclusive=True)
        self.set_interval(10, self._auto_refresh_tick)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # 按鈕列直接重用既有的 action_* 方法，跟鍵盤快捷鍵共用同一套邏輯。
        # 注意：Textual 內建的 action_quit 是 async 方法，所以這裡要判斷並 await，
        # 不然只是建立了一個沒被執行的 coroutine，App 不會真的關閉。
        dispatch = {
            "btn_connect": self.action_connect_selected,
            "btn_disconnect": self.action_disconnect,
            "btn_forget": self.action_forget,
            "btn_info": self.action_info,
            "btn_rescan": self.action_rescan,
            "btn_sort": self.action_cycle_sort,
            "btn_radio": self.action_toggle_radio,
            "btn_search": self.action_search,
            "btn_quit": self.action_quit,
        }
        handler = dispatch.get(event.button.id)
        if handler is None:
            return
        result = handler()
        if asyncio.iscoroutine(result):
            await result

    # --- 共用小工具 -----------------------------------------------------

    def _set_busy(self, message: str | None) -> None:
        self.busy = message is not None
        row = self.query_one("#loading_row")
        row.set_class(not self.busy, "hidden")
        if message:
            self.query_one("#loading_label", Static).update(message)

    def _modal_open(self) -> bool:
        return len(self.screen_stack) > 1

    async def _auto_refresh_tick(self) -> None:
        if self.busy or self._modal_open():
            return  # 使用者正在操作彈出視窗，暫停自動更新避免打斷
        await self._do_scan(silent=True)

    # --- 初始載入 / 掃描 -----------------------------------------------------

    async def _initial_load(self) -> None:
        self._set_busy("讀取 WiFi 狀態...")
        self.wifi_radio_on = await asyncio.to_thread(get_wifi_radio_status)
        await self._do_scan()
        self._set_busy(None)

    def action_rescan(self) -> None:
        if self.busy:
            return
        self.run_worker(self._scan_with_busy(), exclusive=True)

    async def _scan_with_busy(self) -> None:
        self._set_busy("掃描中...")
        await self._do_scan()
        self._set_busy(None)

    async def _do_scan(self, silent: bool = False) -> None:
        if not silent:
            self._set_busy("掃描中...")
        saved_names = await asyncio.to_thread(get_saved_wifi_names)
        self.networks = await asyncio.to_thread(scan_wifi, saved_names)
        ip_info = await asyncio.to_thread(get_ip_info)
        self._refresh_table()
        self._update_status(ip_info)
        if not silent:
            self._set_busy(None)

    # --- 畫面更新 -----------------------------------------------------

    def _refresh_table(self) -> None:
        self.displayed = self._filtered_sorted_networks()
        table = self.query_one("#wifi_table", DataTable)
        table.clear()
        for net in self.displayed:
            star = "★" if net.saved else ""
            dot = "🟢" if net.in_use else ""
            band = freq_to_band(net.freq_mhz)
            bar = signal_bar(net.signal)
            table.add_row(star, dot, net.display_name, band, f"{bar} {net.signal}%", net.security)

    def _filtered_sorted_networks(self) -> list[WifiNetwork]:
        items = self.networks
        query = self.search_query.strip().lower()
        if query:
            items = [n for n in items if query in n.display_name.lower()]

        mode = SORT_MODES[self.sort_index]
        if mode == "Signal":
            items = sorted(items, key=lambda n: -n.signal)
        elif mode == "SSID":
            items = sorted(items, key=lambda n: (n.is_hidden, n.display_name.lower()))
        elif mode == "Security":
            items = sorted(items, key=lambda n: n.security.lower())
        elif mode == "Saved":
            items = sorted(items, key=lambda n: (not n.saved, -n.signal))
        return items

    def _update_status(self, ip_info: dict) -> None:
        current = next((n for n in self.networks if n.in_use), None)
        connected = current is not None

        radio_text = f"WiFi: {'ON' if self.wifi_radio_on else 'OFF'}"
        line1 = f"{'🟢' if connected else '⚪'} {current.display_name if current else '未連線'}    {radio_text}"

        if connected and ip_info.get("ipv4"):
            line2 = f"IPv4: {ip_info['ipv4']}"
            if ip_info.get("gateway"):
                line2 += f"   Gateway: {ip_info['gateway']}"
            if ip_info.get("dns"):
                line2 += f"   DNS: {ip_info['dns']}"
        else:
            line2 = ""

        sort_text = f"Sort: {SORT_MODES[self.sort_index]}"
        search_text = f"Search: {self.search_query}" if self.search_query else ""
        line3 = f"{sort_text}    {search_text}"

        bar = self.query_one("#top_status", Static)
        bar.update("\n".join(filter(None, [line1, line2, line3])))
        bar.set_class(connected, "connected")

    # --- 連線 / 斷線 -----------------------------------------------------

    def _current_network(self) -> WifiNetwork | None:
        """取得目前游標停在哪一列的網路，按鈕跟快捷鍵共用這個來決定操作對象"""
        if not self.displayed:
            return None
        table = self.query_one("#wifi_table", DataTable)
        if table.cursor_row is None:
            return None
        return self.displayed[table.cursor_row]

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.cursor_row is None or not self.displayed:
            return
        self._connect_to(self.displayed[event.cursor_row])

    def action_connect_selected(self) -> None:
        """給「連線」按鈕用：對目前游標所在的網路執行連線"""
        net = self._current_network()
        if net is not None:
            self._connect_to(net)

    def _connect_to(self, net: WifiNetwork) -> None:
        if net.in_use:
            self.notify(f"已經連線在 {net.display_name}", severity="information")
            return

        if net.saved:
            # 已經有 connection profile，密碼已存在 NetworkManager 裡，
            # 不用再問一次，直接喚醒舊設定即可
            self.run_worker(self._do_connect_saved(net.ssid), exclusive=True)
        elif net.is_hidden:
            self.push_screen(HiddenSSIDModal(), self._on_hidden_ssid_entered)
        elif net.is_open:
            self.run_worker(self._do_connect(net.ssid, None), exclusive=True)
        else:
            self.push_screen(PasswordModal(net.ssid), self._make_password_callback(net.ssid))

    def _on_hidden_ssid_entered(self, ssid: str | None) -> None:
        if not ssid:
            return
        self.push_screen(PasswordModal(ssid), self._make_password_callback(ssid, hidden=True))

    def _make_password_callback(self, ssid: str, hidden: bool = False):
        def callback(password: str | None) -> None:
            if password is None:
                return
            self.run_worker(self._do_connect(ssid, password, hidden), exclusive=True)
        return callback

    async def _do_connect_saved(self, ssid: str) -> None:
        """已儲存網路：直接用舊設定連線，不問密碼。
        只有在失敗時 (例如密碼在路由器端被改過、profile 失效) 才退回密碼輸入框。"""
        self._set_busy(f"連線中：{ssid} ...")
        ok, msg = await asyncio.to_thread(connect_saved_wifi, ssid)
        self._set_busy(None)

        if ok:
            self.notify(msg, severity="information")
            await self._do_scan()
            return

        self.notify(f"{msg}，需要重新輸入密碼", severity="warning")
        self.push_screen(
            PasswordModal(ssid, error_message="已儲存的設定失效，請重新輸入密碼"),
            self._make_password_callback(ssid),
        )

    async def _do_connect(self, ssid: str, password: str | None, hidden: bool = False) -> None:
        self._set_busy(f"連線中：{ssid} ...")
        ok, msg = await asyncio.to_thread(connect_wifi, ssid, password, hidden)
        self._set_busy(None)

        if ok:
            self.notify(msg, severity="information")
            await self._do_scan()
            return

        if password and looks_like_auth_error(msg):
            # 很可能是密碼錯誤，直接跳回密碼輸入框，不用使用者自己再按一次 Enter
            self.push_screen(
                PasswordModal(ssid, error_message="密碼錯誤，請重新輸入"),
                self._make_password_callback(ssid, hidden),
            )
        else:
            self.notify(msg, severity="error")
            await self._do_scan()

    def action_disconnect(self) -> None:
        net = self._current_network()
        if net is None:
            return
        if not net.in_use:
            self.notify("這個網路目前不是使用中的連線", severity="warning")
            return
        self.run_worker(self._do_disconnect(net.ssid), exclusive=True)

    async def _do_disconnect(self, ssid: str) -> None:
        self._set_busy(f"斷開中：{ssid} ...")
        ok, msg = await asyncio.to_thread(disconnect_wifi, ssid)
        self._set_busy(None)
        self.notify(msg, severity="information" if ok else "error")
        await self._do_scan()

    # --- Forget Network -----------------------------------------------------

    def action_forget(self) -> None:
        net = self._current_network()
        if net is None:
            return
        if not net.saved:
            self.notify("沒有已儲存的設定", severity="warning")
            return

        def handle_confirm(confirmed: bool) -> None:
            if confirmed:
                self.run_worker(self._do_forget(net.ssid), exclusive=True)

        self.push_screen(ConfirmModal("確定忘記此網路？", net.ssid), handle_confirm)

    async def _do_forget(self, ssid: str) -> None:
        self._set_busy(f"刪除設定：{ssid} ...")
        ok, msg = await asyncio.to_thread(delete_connection, ssid)
        self._set_busy(None)
        self.notify(msg, severity="information" if ok else "error")
        await self._do_scan()

    # --- 詳細資訊 -----------------------------------------------------

    def action_info(self) -> None:
        net = self._current_network()
        if net is not None:
            self.push_screen(InfoModal(net))

    # --- 排序 -----------------------------------------------------

    def action_cycle_sort(self) -> None:
        self.sort_index = (self.sort_index + 1) % len(SORT_MODES)
        self._refresh_table()
        self._update_status_only_sort_line()

    def _update_status_only_sort_line(self) -> None:
        # 排序切換不需要重新打 nmcli，直接用目前資料重畫狀態列即可
        self.run_worker(self._refresh_status_only(), exclusive=False)

    async def _refresh_status_only(self) -> None:
        ip_info = await asyncio.to_thread(get_ip_info)
        self._update_status(ip_info)

    # --- WiFi 無線電開關 -----------------------------------------------------

    def action_toggle_radio(self) -> None:
        self.run_worker(self._do_toggle_radio(), exclusive=True)

    async def _do_toggle_radio(self) -> None:
        target_state = not self.wifi_radio_on
        self._set_busy(f"{'開啟' if target_state else '關閉'} WiFi ...")
        ok, msg = await asyncio.to_thread(set_wifi_radio, target_state)
        if ok:
            self.wifi_radio_on = target_state
        self._set_busy(None)
        self.notify(msg, severity="information" if ok else "error")
        if self.wifi_radio_on:
            await self._do_scan()
        else:
            self.networks = []
            self._refresh_table()
            self._update_status({})

    # --- 搜尋 -----------------------------------------------------

    def action_search(self) -> None:
        row = self.query_one("#search_row")
        row.remove_class("hidden")
        self.query_one("#search_input", Input).focus()

    def action_clear_search(self) -> None:
        if not self.search_query and self.query_one("#search_row").has_class("hidden"):
            return  # 沒在搜尋狀態，escape 不做任何事，避免干擾其他操作
        if self._search_debounce_timer is not None:
            self._search_debounce_timer.stop()
        self.search_query = ""
        self.query_one("#search_input", Input).value = ""
        self.query_one("#search_row").add_class("hidden")
        self._refresh_table()
        self.set_focus(self.query_one("#wifi_table", DataTable))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search_input":
            return
        self.search_query = event.value
        # debounce：每次按鍵不立刻重畫整個表格，停頓 150ms 後才重畫，
        # 避免透過 SSH/tmux 時每個字都要來回一次造成的卡頓感
        if self._search_debounce_timer is not None:
            self._search_debounce_timer.stop()
        self._search_debounce_timer = self.set_timer(0.15, self._refresh_table)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search_input":
            self.set_focus(self.query_one("#wifi_table", DataTable))


if __name__ == "__main__":
    WifiManagerApp().run()