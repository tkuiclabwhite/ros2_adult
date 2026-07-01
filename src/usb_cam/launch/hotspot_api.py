#!/usr/bin/env python3
"""熱點裝置管理 API：讀取 dnsmasq 租約、透過 iptables 踢除/解封裝置。"""
import json
import re
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # 頁面 (port 9999) 與 API (port 5050) 是不同 origin，需要 CORS 標頭
    # 否則瀏覽器會擋掉回應，前端一律顯示成「無法連線」
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


HOTSPOT_SSID = 'TKU_TOWEN'
HOTSPOT_IP = '10.10.10.10'
WIFI_INTERFACE = 'wlP1p1s0'
LEASE_PATHS = [
    '/var/lib/NetworkManager/dnsmasq-wlP1p1s0.leases',
    '/var/lib/misc/dnsmasq.leases',
]

MAC_RE = re.compile(r'^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$')

# 這台機器的核心 (5.15.148-tegra) 沒編譯 xt_mac，iptables 的 `-m mac` 用不了，
# 改成用 IP 擋。踢除時記住當下的 mac -> ip 對應，解封時才知道要放行哪個 IP。
BLOCKED_STATE_FILE = Path(__file__).with_name('blocked_devices.json')


def find_lease_file():
    for path in LEASE_PATHS:
        if Path(path).exists():
            return path
    return None


def load_blocked():
    if not BLOCKED_STATE_FILE.exists():
        return {}
    try:
        return json.loads(BLOCKED_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_blocked(blocked):
    BLOCKED_STATE_FILE.write_text(json.dumps(blocked))


def read_devices():
    lease_file = find_lease_file()
    if lease_file is None:
        return []

    blocked = load_blocked()
    devices = []
    with open(lease_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            expiry_epoch, mac, ip, hostname = parts[0], parts[1], parts[2], parts[3]
            since = subprocess.run(
                ['date', '-d', f'@{expiry_epoch}', '+%Y-%m-%d %H:%M:%S'],
                capture_output=True, text=True,
            ).stdout.strip()
            mac = mac.lower()
            devices.append({
                'ip': ip,
                'mac': mac,
                'hostname': hostname if hostname != '*' else '',
                'since': since,
                'blocked': mac in blocked,
            })
    return devices


def is_valid_mac(mac):
    return bool(MAC_RE.match(mac or ''))


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
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
