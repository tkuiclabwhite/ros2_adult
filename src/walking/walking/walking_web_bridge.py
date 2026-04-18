#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import yaml
import configparser
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from tku_msgs.msg import Parameter as ParameterMsg
from tku_msgs.srv import WalkingGaitParameter
from tku_msgs.msg import Location as LocationMsg

from . import Parameter as parameter

# 骨盆寬度（全寬, cm）
PELVIS_WIDTH = float(getattr(parameter, "LENGTH_PELVIS", 9.1))
FULL_WIDTH_MARGIN = 0.4   # cm
DEFAULT_SAVE_PATH = os.path.expanduser("~/.ros/walking_params.yaml")

class WalkingWebBridge(Node):
    def __init__(self):
        super().__init__('walking_web_bridge')
        # 強制讓 print 即時輸出 (針對整個 Node 生效)
        sys.stdout.reconfigure(line_buffering=True)
        
        self.get_logger().info("DEBUG: [Init] Node is starting...")

        # 1. 初始化路徑邏輯
        self.location = ""
        try:
            self.strategy_root = self._resolve_strategy_root()
            self.get_logger().info(f"DEBUG: [Init] Strategy root: {self.strategy_root}")
        except Exception as e:
            self.get_logger().error(f"ERROR: [Init] Failed to resolve root: {e}")
            self.strategy_root = Path("/tmp") # Fallback

        # 嘗試讀取 strategy.ini
        raw = self._read_strategy_ini_raw()
        if raw:
            self.location = self._canon_location(raw)
            resolved_path = self._resolve_walking_path()
            self.current_save_path = str(resolved_path)
            self.get_logger().info(f"DEBUG: [Init] Loaded from ini: {self.current_save_path}")
        else:
            self.current_save_path = DEFAULT_SAVE_PATH
            self.get_logger().info(f"DEBUG: [Init] Using default path: {self.current_save_path}")

        # 2. 參數初始化
        self.current_params = {
            "com_y_swing":   float(getattr(parameter, "com_y_swing"     , 0.0)),
            "width_size":    float(getattr(parameter, "width_size"      , 4.5)),
            "period_t":      float(getattr(parameter, "period_t"        , 360.0)),
            "t_dsp":         float(getattr(parameter, "t_dsp"           , 0.0)),
            "lift_height":   float(getattr(parameter, "lift_height"     , 3.0)),
            "stand_height":  float(getattr(parameter, "stand_height"    , 23.5)),
            "com_height":    float(getattr(parameter, "com_height"      , 29.5)),
            "hip_roll":      float(getattr(parameter, "hip_roll"        , 0.0)),
            "ankle_roll":    float(getattr(parameter, "ankle_roll"      , 0.0)),

            "step_length":   float(getattr(parameter, "step_length"     , 0.0)),
            "shift_length":  float(getattr(parameter, "shift_length"    , 0.0)),
            "theta":         float(getattr(parameter, "theta"           , 0.0)),
        }

        # 3. 建立通訊
        self.params_update_pub = self.create_publisher(String, '/walking_params_update', 10)
        self.params_sub = self.create_subscription(String, '/walking_params', self.walking_params_callback, 10)
        
        # 訂閱網頁訊號
        self.location_sub = self.create_subscription(LocationMsg, '/location', self.location_callback, 10)
        self.location_pub = self.create_publisher(LocationMsg, '/locationBack', 10)
        self.param_save_sub = self.create_subscription(ParameterMsg, '/web/parameter_Topic', self.param_save_callback, 10)
        
        self.srv = self.create_service(WalkingGaitParameter, '/web/LoadWalkingGaitParameter', self.handle_load_walking_param)

        self.get_logger().info('WalkingWebBridge Ready & Waiting for messages...')

        # 4. 初始載入
        self.load_and_publish_params(self.current_save_path)

    # -------------------------------------------------------------
    # 核心邏輯
    # -------------------------------------------------------------
    def _resolve_walking_path(self) -> Path:
        if not self.location: return Path(DEFAULT_SAVE_PATH)
        p = Path(self.location)

        # 如果是資料夾，自動加上 WalkingParameter.ini
        if p.is_dir():
            if (p / "WalkingParameter.yaml").exists(): return p / "WalkingParameter.yaml"
            return p / "WalkingParameter.ini"

        if p.suffix: return p
        if p.with_suffix(".ini").exists(): return p.with_suffix(".ini")
        if p.with_suffix(".yaml").exists(): return p.with_suffix(".yaml")
        return p.with_suffix(".ini")

    def _resolve_strategy_root(self) -> Path:
        for up in Path(__file__).resolve().parents:
            if up.name == "src" and (up / "strategy" / "strategy").is_dir():
                return up / "strategy" / "strategy"
        return Path.home() / "tku" / "src" / "strategy" / "strategy"

    def _read_strategy_ini_raw(self) -> str:
        try:
            p = self._resolve_strategy_root() / "strategy.ini"
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                for line in txt.splitlines():
                    if line.strip(): return line.strip()
        except Exception as e:
            self.get_logger().warn(f"DEBUG: Read ini failed: {e}")
        return ""

    def _write_strategy_ini_raw(self, raw: str) -> None:
        try:
            p = self._resolve_strategy_root() / "strategy.ini"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write((raw or "").strip() + "\n")
        except Exception as e:
            self.get_logger().error(f"ERROR: Write ini failed: {e}")

    def _canon_location(self, raw: str) -> str:
        if not raw: return ""
        clean_raw = raw.strip().lstrip("/")
        return str(self.strategy_root / clean_raw)

    def load_and_publish_params(self, path):
        self.get_logger().info(f"DEBUG: [Load] Attempting to load: {path}")
        if not os.path.exists(path):
            self.get_logger().warn(f"DEBUG: [Load] File not found: {path}")
            return
        
        loaded_data = {}
        try:
            # 嘗試 YAML
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded_data = yaml.safe_load(f)
            except: pass

            # 嘗試 INI
            if not isinstance(loaded_data, dict) or not loaded_data:
                loaded_data = {}
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith(('#', ';', '[')): continue
                        if '=' in line:
                            parts = line.split('=', 1)
                            # 簡單的數值轉換嘗試
                            val_str = parts[1].strip()
                            try:
                                loaded_data[parts[0].strip()] = float(val_str)
                            except ValueError:
                                loaded_data[parts[0].strip()] = val_str

            if loaded_data:
                self.current_params.update(loaded_data)
                json_str = json.dumps(loaded_data)
                self.params_update_pub.publish(String(data=json_str))
                
                # ★ 強制輸出 Print 訊息
                print(f"\033[93m{'='*40}", flush=True)
                print(f"📂 [Load] 成功載入參數", flush=True)
                print(f"📄 來源: {os.path.basename(path)}", flush=True)
                print(f"🚀 已發送至 WalkingNode", flush=True)
                print(f"{'='*40}\033[0m", flush=True)
            else:
                self.get_logger().warn("DEBUG: [Load] File empty or format error.")

        except Exception as e:
            self.get_logger().error(f"ERROR: [Load] Exception: {e}")

    # -------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------
    def location_callback(self, msg):
        self.get_logger().info(f"DEBUG: [Callback] /location received: {msg.data}")
        raw = (msg.data or "").strip()
        if not raw: raw = "ar/Parameter"

        # 1. 處理路徑
        rel = Path(raw)
        if rel.is_absolute():
            parts = rel.parts
            if len(parts) >= 2: rel = Path(parts[-2]) / parts[-1]
            else: rel = Path(rel.name)
        loc_str = str(rel)

        # 2. 寫入與更新
        self._write_strategy_ini_raw(loc_str)
        self.location = self._canon_location(loc_str)
        target_file = self._resolve_walking_path()
        self.current_save_path = str(target_file)
        
        self.get_logger().info(f"DEBUG: [Callback] Switched path to: {self.current_save_path}")

        # 3. 讀檔並發送
        self.load_and_publish_params(self.current_save_path)

        # 4. 回傳
        msg.data = loc_str
        self.location_pub.publish(msg)

    def param_save_callback(self, msg: ParameterMsg):
        self.get_logger().info("DEBUG: [Callback] /web/parameter_Topic received SAVE request")
        
        target_path = self._resolve_walking_path()
        self.get_logger().info(f"DEBUG: [Save] Saving to: {target_path}")

        data = {}
        for slot in msg.__slots__:
            key = slot.lstrip("_")
            data[key] = getattr(msg, slot)

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if str(target_path).endswith(".ini"):
                with open(target_path, "w", encoding="utf-8") as f:
                    for k, v in data.items():
                        f.write(f"{k}={v}\n")
            else:
                with open(target_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

            # ★ 強制輸出 Print 訊息
            print(f"\033[92m✅ [Save] Success: {target_path}\033[0m", flush=True)
            
            self.current_save_path = str(target_path)
            self.current_params.update(data)

        except Exception as e:
            self.get_logger().error(f"ERROR: [Save] Failed: {e}")

    def walking_params_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.current_params.update(data)
        except: pass

    def handle_load_walking_param(self, request, response):
        self.get_logger().info(f"DEBUG: [Service] Load params called. Source: {self.current_save_path}")
        p = self.current_params

        # =========================================================================
        # p.get("名稱", 預設值))
        # =========================================================================
        response.com_y_swing        = float (p.get("com_y_swing", 0.0))
        response.width_size         = float (p.get("width_size", 0.0))
        response.period_t           = int   (p.get("period_t"))
        response.t_dsp              = float (p.get("t_dsp", 0.0))
        response.lift_height        = float (p.get("lift_height", 2.5))
        response.stand_height       = float (p.get("stand_height", 23.5))
        response.com_height         = float (p.get("com_height", 29.5))
        response.hip_roll           = float (p.get("hip_roll", 0.0))
        response.ankle_roll         = float (p.get("ankle_roll", 0.0))
        
        # ★ Debug 印出確認
        print(f"DEBUG: [Check] com_y_swing 最終回傳值:  {response.com_y_swing}" , flush=True)
        print(f"DEBUG: [Check] width_size 最終回傳值:   {response.width_size}"  , flush=True)
        print(f"DEBUG: [Check] period_t 最終回傳值:     {response.period_t}"    , flush=True)
        print(f"DEBUG: [Check] t_dsp 最終回傳值:        {response.t_dsp}"       , flush=True)
        print(f"DEBUG: [Check] lift_height 最終回傳值:  {response.lift_height}" , flush=True)
        print(f"DEBUG: [Check] stand_height 最終回傳值: {response.stand_height}", flush=True)
        print(f"DEBUG: [Check] com_height 最終回傳值:   {response.com_height}"  , flush=True)
        print(f"DEBUG: [Check] hip_roll 最終回傳值:     {response.hip_roll}"    , flush=True)
        print(f"DEBUG: [Check] ankle_roll 最終回傳值:   {response.ankle_roll}"  , flush=True)

        return response

def main(args=None):
    rclpy.init(args=args)
    node = WalkingWebBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()