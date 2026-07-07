#!/usr/bin/env python3
"""熱點裝置管理 API：讀取 dnsmasq 租約、透過 iptables 踢除/解封裝置。"""
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # 頁面 (port 9999) 與 API (port 5050) 是不同 origin，需要 CORS 標頭
    # 否則瀏覽器會擋掉回應，前端一律顯示成「無法連線」
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


HOTSPOT_SSID = 'TKU_TOWEN'
HOTSPOT_IP = '10.10.10.10'
HOTSPOT_SUBNET_PREFIX = '10.10.10.'
WIFI_INTERFACE = 'wlP1p1s0'
LEASE_PATHS = [
    '/var/lib/NetworkManager/dnsmasq-wlP1p1s0.leases',
    '/var/lib/misc/dnsmasq.leases',
]

MAC_RE = re.compile(r'^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$')

# 這台機器的核心 (5.15.148-tegra) 沒編譯 xt_mac，iptables 的 `-m mac` 用不了，
# 改成用 IP 擋。踢除時記住當下的 mac -> ip 對應，解封時才知道要放行哪個 IP。
BLOCKED_STATE_FILE = Path(__file__).with_name('blocked_devices.json')

# 自訂暱稱：只覆蓋顯示用的名稱，不動裝置本身回報的 hostname。
NICKNAMES_FILE = Path(__file__).with_name('nicknames.json')

# 固定 IP：寫進 NetworkManager 的 dnsmasq-shared 設定，裝置下次重新連線熱點才會生效
# （dnsmasq 不會主動改變一個已經在租用中的 IP）。
RESERVATIONS_FILE = Path(__file__).with_name('reservations.json')
DNSMASQ_RESERVATION_CONF = Path('/etc/NetworkManager/dnsmasq-shared.d/hotspot_reservations.conf')


def find_lease_file():
    for path in LEASE_PATHS:
        if Path(path).exists():
            return path
    return None


def load_json_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_blocked():
    return load_json_state(BLOCKED_STATE_FILE)


def save_blocked(blocked):
    BLOCKED_STATE_FILE.write_text(json.dumps(blocked))


def load_nicknames():
    return load_json_state(NICKNAMES_FILE)


def save_nicknames(nicknames):
    NICKNAMES_FILE.write_text(json.dumps(nicknames))


def load_reservations():
    return load_json_state(RESERVATIONS_FILE)


def save_reservations(reservations):
    RESERVATIONS_FILE.write_text(json.dumps(reservations))
    DNSMASQ_RESERVATION_CONF.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'dhcp-host={mac},{ip}' for mac, ip in reservations.items()]
    content = '\n'.join(lines) + ('\n' if lines else '')
    DNSMASQ_RESERVATION_CONF.write_text(content)


def get_associated_macs():
    """查詢無線網卡目前實際關聯（associated）中的裝置，跟 DHCP 租約是不同的資料來源。"""
    result = subprocess.run(
        ['iw', 'dev', WIFI_INTERFACE, 'station', 'dump'],
        capture_output=True, text=True,
    )
    macs = set()
    for line in result.stdout.splitlines():
        if line.startswith('Station'):
            parts = line.split()
            if len(parts) >= 2:
                macs.add(parts[1].lower())
    return macs


def read_devices():
    blocked = load_blocked()
    nicknames = load_nicknames()
    reservations = load_reservations()
    associated = get_associated_macs()
    devices = {}

    lease_file = find_lease_file()
    if lease_file is not None:
        with open(lease_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                expiry_epoch, mac, ip, hostname = parts[0], parts[1], parts[2], parts[3]
                # 租約檔第一欄是「租約到期時間」，不是裝置連線當下的時間；
                # 續約會讓這個時間一直往後延，沒辦法從這裡反推出最初的連線時刻。
                lease_expires = subprocess.run(
                    ['date', '-d', f'@{expiry_epoch}', '+%Y-%m-%d %H:%M:%S'],
                    capture_output=True, text=True,
                ).stdout.strip()
                mac = mac.lower()
                devices[mac] = {
                    'ip': ip,
                    'mac': mac,
                    'hostname': hostname if hostname != '*' else '',
                    'nickname': nicknames.get(mac, ''),
                    'lease_expires': lease_expires,
                    'blocked': mac in blocked,
                    'reserved_ip': reservations.get(mac),
                    'connected': mac in associated,
                }

    # 租約檔只記錄「目前還沒過期的租約」，不是永久歷史。把我們自己存過資料
    # （暱稱／封鎖／固定 IP）但目前不在租約檔裡的裝置也一併列出來，標成離線。
    known_macs = set(nicknames) | set(blocked) | set(reservations)
    for mac in known_macs:
        if mac in devices:
            continue
        devices[mac] = {
            'ip': reservations.get(mac, ''),
            'mac': mac,
            'hostname': '',
            'nickname': nicknames.get(mac, ''),
            'lease_expires': '',
            'blocked': mac in blocked,
            'reserved_ip': reservations.get(mac),
            'connected': False,
        }

    return list(devices.values())


def is_valid_mac(mac):
    return bool(MAC_RE.match(mac or ''))


def is_valid_hotspot_ip(ip):
    if not isinstance(ip, str) or not ip.startswith(HOTSPOT_SUBNET_PREFIX):
        return False
    last_octet = ip[len(HOTSPOT_SUBNET_PREFIX):]
    if not last_octet.isdigit():
        return False
    value = int(last_octet)
    # .10 是熱點自己的 IP，不能拿來當保留位址
    return 1 < value < 255 and ip != HOTSPOT_IP


def deauth_client(mac):
    """透過 wpa_supplicant 的 AP control interface 送出 802.11 deauth，逼裝置立即斷線。"""
    result = subprocess.run(
        ['wpa_cli', '-p', '/run/wpa_supplicant', '-i', WIFI_INTERFACE, 'deauthenticate', mac],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and 'OK' in result.stdout


@app.route('/devices', methods=['GET'])
def devices():
    return jsonify(read_devices())


@app.route('/kick', methods=['POST'])
def kick():
    data = request.get_json(silent=True) or {}
    mac = data.get('mac')
    if not is_valid_mac(mac):
        return jsonify({'success': False, 'error': 'invalid mac'}), 400
    mac = mac.lower()

    ip = next((d['ip'] for d in read_devices() if d['mac'] == mac), None)
    if ip is None:
        return jsonify({'success': False, 'error': 'device not currently connected'}), 404

    try:
        subprocess.run(
            ['iptables', '-A', 'FORWARD', '-s', ip, '-j', 'DROP'],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ['iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP'],
            check=True, capture_output=True, text=True,
        )
        blocked = load_blocked()
        blocked[mac] = ip
        save_blocked(blocked)
        deauthenticated = deauth_client(mac)
        return jsonify({'success': True, 'deauthenticated': deauthenticated})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': e.stderr.strip()}), 500


@app.route('/kick/<mac>', methods=['DELETE'])
def unkick(mac):
    if not is_valid_mac(mac):
        return jsonify({'success': False, 'error': 'invalid mac'}), 400
    mac = mac.lower()

    blocked = load_blocked()
    ip = blocked.get(mac)
    if ip is None:
        return jsonify({'success': False, 'error': 'device not currently tracked as blocked'}), 404

    try:
        subprocess.run(
            ['iptables', '-D', 'FORWARD', '-s', ip, '-j', 'DROP'],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ['iptables', '-D', 'INPUT', '-s', ip, '-j', 'DROP'],
            check=True, capture_output=True, text=True,
        )
        del blocked[mac]
        save_blocked(blocked)
        return jsonify({'success': True})
    except subprocess.CalledProcessError as e:
        return jsonify({'success': False, 'error': e.stderr.strip()}), 500


@app.route('/devices/<mac>', methods=['PATCH'])
def rename_device(mac):
    if not is_valid_mac(mac):
        return jsonify({'success': False, 'error': 'invalid mac'}), 400
    mac = mac.lower()

    data = request.get_json(silent=True) or {}
    nickname = (data.get('nickname') or '').strip()

    nicknames = load_nicknames()
    if nickname:
        nicknames[mac] = nickname
    else:
        nicknames.pop(mac, None)
    save_nicknames(nicknames)
    return jsonify({'success': True})


@app.route('/devices/<mac>/reserve', methods=['POST'])
def reserve_device_ip(mac):
    if not is_valid_mac(mac):
        return jsonify({'success': False, 'error': 'invalid mac'}), 400
    mac = mac.lower()

    data = request.get_json(silent=True) or {}
    ip = data.get('ip')
    if not is_valid_hotspot_ip(ip):
        return jsonify({'success': False, 'error': 'invalid ip, must be in 10.10.10.0/24'}), 400

    reservations = load_reservations()
    conflict_mac = next((m for m, i in reservations.items() if i == ip and m != mac), None)
    if conflict_mac:
        return jsonify({'success': False, 'error': f'ip already reserved for {conflict_mac}'}), 409

    reservations[mac] = ip
    save_reservations(reservations)
    return jsonify({'success': True, 'note': '裝置下次重新連線熱點時生效'})


@app.route('/devices/<mac>/reserve', methods=['DELETE'])
def unreserve_device_ip(mac):
    if not is_valid_mac(mac):
        return jsonify({'success': False, 'error': 'invalid mac'}), 400
    mac = mac.lower()

    reservations = load_reservations()
    if mac not in reservations:
        return jsonify({'success': False, 'error': 'no reservation for this device'}), 404

    del reservations[mac]
    save_reservations(reservations)
    return jsonify({'success': True})


@app.route('/hotspot_status', methods=['GET'])
def hotspot_status():
    # 直接看熱點 IP 有沒有實際綁在網卡上，比對連線名稱是否等於 SSID 更準確
    # （nmcli connection 的 profile 名稱不一定跟 SSID 一樣）
    result = subprocess.run(
        ['ip', '-o', '-4', 'addr', 'show'],
        capture_output=True, text=True,
    )
    active = HOTSPOT_IP in result.stdout
    return jsonify({
        'active': active,
        'ssid': HOTSPOT_SSID,
        'ip': HOTSPOT_IP,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
