#!/usr/bin/env python3
# coding=utf-8

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int16, Int16MultiArray

from tku_msgs.msg import InterfaceSend2Sector, SaveMotion, SingleMotorData
from tku_msgs.srv import CheckSector, ReadMotion
from rcl_interfaces.msg import SetParametersResult

import configparser
import os
import time
import threading

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_strategy_node")
        qos = QoSProfile(depth=1000)

        self.declare_parameter("location", "ar")
        self.location_folder = self.get_parameter("location").get_parameter_value().string_value
        self.add_on_set_parameters_callback(self.cb_param_update)

        self.get_logger().info(f"📂 Current Strategy Location: .../strategy/{self.location_folder}/Parameter/")

        # Publishers
        self.cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.torque_pub = self.create_publisher(Int16MultiArray, '/set_torque', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.web_sub = self.create_subscription(InterfaceSend2Sector, "/package/InterfaceSend2Sector", self.web_interface_cb, qos)
        self.save_sub = self.create_subscription(SaveMotion, '/package/InterfaceSaveMotion', self.cb_save_motion, qos)
        self.singlemotor_sub = self.create_subscription(SingleMotorData, '/package/SingleMotorData', self.cb_single_motor, 10)

        # Variables
        self.current_joints = {} 
        self.joints_lock = threading.Lock()
        self.web_buffer = [] 
        self.current_sector_name = "0" 
        self.saved_sectors = {}        
        self.id_source_map = {} 
        self.file_save_buffer = {}
        
        self.last_goals = {} 

        # Services
        self.check_sector_srv = self.create_service(CheckSector, '/package/InterfaceCheckSector', self.cb_check_sector)
        self.sector_execute_sub = self.create_subscription(Int16, '/package/Sector', self.cb_sector_execute, qos)
        self.read_motion_srv = self.create_service(ReadMotion, '/package/InterfaceReadSaveMotion', self.cb_read_motion)
        
        self.motion_callback_pub = self.create_publisher(Bool, '/package/motioncallback', qos)
        self.execute_callback_pub = self.create_publisher(Bool, '/package/executecallback', qos)
        self.anchor_reset_pub = self.create_publisher(Bool, '/walking_reset_anchor', 10)
        self.get_logger().info("Motion Strategy Node initialized. Waiting for Driver...")
        
        self.stand_dir = "/home/iclab/ros2_adult/src/strategy/strategy/Parameter"
        self.stand_file = os.path.join(self.stand_dir, "stand.ini")

        self.load_startup_stand_motion()
        self.load_all_strategy_motions()

    # ==========================================================
    # Web Communication
    # ==========================================================
    def web_interface_cb(self, msg):
        self.get_logger().info(f"[DEBUG] 收到網頁數據: {msg.package}")
        if msg.sectorname: self.current_sector_name = msg.sectorname
        
        # 使用 extend 展開數據
        self.web_buffer.extend(msg.package)
        self.parse_and_process_buffer()

    def parse_and_process_buffer(self):
        while len(self.web_buffer) >= 3:
            if self.web_buffer[0] != 83 or self.web_buffer[1] != 84:
                self.web_buffer.pop(0)
                continue
            
            opcode = self.web_buffer[2]
            expected_len = 0
            if opcode == 246: expected_len = 7
            elif opcode == 242: expected_len = 59
            elif opcode == 243: expected_len = 59
            elif opcode == 244: expected_len = 45
            elif opcode == 245: expected_len = 59
            else:
                self.get_logger().warn(f"未知 Opcode: {opcode}")
                self.web_buffer.pop(0)
                continue

            if len(self.web_buffer) < expected_len: return 

            if self.web_buffer[expected_len-2] == 78 and self.web_buffer[expected_len-1] == 69:
                packet = self.web_buffer[:expected_len]
                self.process_packet_data(packet, opcode)
                self.web_buffer = self.web_buffer[expected_len:]
            else:
                self.web_buffer.pop(0)

    def process_packet_data(self, packet, opcode):
        if opcode == 246: # Torque Control
            motor_id = packet[3]
            state = packet[4]
            self.get_logger().info(f"★ 執行扭力指令 -> ID: {motor_id}, State: {state}")
            msg = Int16MultiArray()
            msg.data = [motor_id, state]
            self.torque_pub.publish(msg)

        elif opcode in [242, 243, 244]: # Save Sector Data (RAM) & Backup to Sector Folder
            data_content = list(packet[3:-2])
            sector_id = str(self.current_sector_name)
            
            # 1. 更新記憶體 (RAM)
            self.saved_sectors[sector_id] = {
                'opcode': opcode,
                'data': data_content
            }
            
            # 2. ★★★ [新增] 自動備份到 sector 資料夾 (Disk) ★★★
            self.save_sector_to_disk(sector_id, opcode, data_content)

            # 3. 回報網頁
            msg = Bool(); msg.data = True; self.motion_callback_pub.publish(msg)

    # ==========================================================
    # [新增] 單一 Sector 存檔功能
    # ==========================================================
    def save_sector_to_disk(self, sector_id, opcode, data_list):
        """
        將單一 ID 的動作數據儲存到 Disk
        ★ 修正策略：允許所有類型的存檔，但過濾掉「全 0」的無效數據
        """
        
        # --- [Smart Filter: 內容過濾] ---
        # 網頁端在 Send 時會自動發送其他 ID 的預設值 (通常全為 0)
        # 我們檢查：如果這一包數據的位置參數全都是 0，就視為垃圾丟棄
        
        is_empty_data = False
        
        if opcode in [242, 243]: # Relative or Absolute
            # 資料結構: [Speed, Pos, Speed, Pos...]
            # 我們只檢查 "Position" (奇數索引: 1, 3, 5...) 是否有數值
            # 如果所有 Position 都是 0，代表這是「原地不動」的無效指令
            positions = data_list[1::2]
            if all(p == 0 for p in positions):
                is_empty_data = True
                
        # MotionList (244) 通常不應該是空的，如果全是 0 也過濾掉
        elif opcode == 244:
            if all(v == 0 for v in data_list):
                is_empty_data = True

        if is_empty_data:
            self.get_logger().info(f"[Sector Save] ID {sector_id} (Opcode {opcode}) 數值全為 0，忽略存檔。")
            return
        # ---------------------------------------

        try:
            home_path = os.path.expanduser("~")
            sector_dir = os.path.join(home_path, "ros2_adult/src/strategy/strategy", self.location_folder, "Parameter", "sector")
            
            if not os.path.exists(sector_dir):
                os.makedirs(sector_dir)
                
            # 檔名只用 ID
            file_path = os.path.join(sector_dir, f"{sector_id}.ini")
            
            config = configparser.ConfigParser()
            config['Data'] = {}
            section = config['Data']
            
            section['id'] = str(sector_id)
            
            if opcode == 244: # MotionList
                section['motionstate'] = '0'
                section['motionlist'] = ','.join(map(str, data_list))
            elif opcode == 243: # Relative
                section['motionstate'] = '2'
                section['motordata'] = ','.join(map(str, data_list))
            elif opcode == 242: # Absolute
                section['motionstate'] = '4'
                section['motordata'] = ','.join(map(str, data_list))
                
            with open(file_path, 'w', encoding='utf-8') as f:
                config.write(f)
                
            self.get_logger().info(f"[Sector Save] ID {sector_id} 已備份至: {file_path}")
            
        except Exception as e:
            self.get_logger().error(f"[Sector Save] 存檔失敗: {e}")

    # ==========================================================
    # Initialization & File Loading
    # ==========================================================

    def load_startup_stand_motion(self):
        full_path = self.stand_file
        if not os.path.exists(full_path):
            self.get_logger().warn(f"[Init] 找不到 Stand 檔案: {full_path}")
            return
        self.get_logger().info(f"[Init] 載入 Stand: {full_path}")
        self._internal_load_ini(full_path, is_common=True)
        

    def load_all_strategy_motions(self):
        home_path = os.path.expanduser("~")
        base_strategy_path = os.path.join(home_path, "ros2_adult/src/strategy/strategy")
        
        path_common = os.path.join(base_strategy_path, "common", "Parameter")
        if os.path.exists(path_common):
            self.get_logger().info(f"[Init] 載入 Common: {path_common}")
            self._load_folder(path_common, is_common=True)
        
        path_specific = os.path.join(base_strategy_path, self.location_folder, "Parameter")
        if os.path.exists(path_specific):
            self.get_logger().info(f"[Init] 載入 Specific ({self.location_folder}): {path_specific}")
            self._load_folder(path_specific, is_common=False)

    def _load_folder(self, folder_path, is_common):
        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.ini')]
            for filename in files:
                self._internal_load_ini(os.path.join(folder_path, filename), filename, is_common)
        except Exception as e:
            self.get_logger().error(f"Folder Load Error: {e}")

    def _internal_load_ini(self, full_path, filename="unknown", is_common=False):
        config = configparser.ConfigParser()
        try:
            config.read(full_path)
            temp_load = {}
            for section in config.sections():
                try:
                    m_state = int(config[section]['motionstate'])
                    m_id = int(config[section]['id'])
                    def parse_list(key):
                        if key in config[section] and config[section][key]:
                            return [int(x) for x in config[section][key].split(',')]
                        return []
                    data_array = []
                    if m_state == 0: data_array = parse_list('motionlist')
                    elif m_state in [1, 2, 3, 4]: data_array = parse_list('motordata')
                    
                    if m_id not in temp_load: temp_load[m_id] = {}
                    temp_load[m_id][m_state] = data_array
                except: pass
            
            self.auto_load_sectors(temp_load, filename, is_common)
            
        except Exception as e:
            self.get_logger().error(f"INI Parse Error ({full_path}): {e}")

    def auto_load_sectors(self, temp_load, source_filename="unknown", is_common=False):
        for m_id, data_map in temp_load.items():
            sector_name = str(m_id)
            if sector_name in self.saved_sectors:
                if not is_common: pass 

            self.id_source_map[sector_name] = source_filename

            if 3 in data_map and 4 in data_map:
                pos_list = data_map[3]; spd_list = data_map[4]
                merged = []; [merged.extend([s,p]) for p,s in zip(pos_list, spd_list)]
                self.saved_sectors[sector_name] = {'opcode': 242, 'data': merged}
            elif 1 in data_map and 2 in data_map:
                pos_list = data_map[1]; spd_list = data_map[2]
                merged = []; [merged.extend([s,p]) for p,s in zip(pos_list, spd_list)]
                self.saved_sectors[sector_name] = {'opcode': 243, 'data': merged}
            elif 0 in data_map:
                self.saved_sectors[sector_name] = {'opcode': 244, 'data': data_map[0]}
        
        self.get_logger().info(f"[AutoLoad] 載入 {len(temp_load)} 個 Sector ({source_filename})")

    # ==========================================================
    # Read / Save / Callbacks
    # ==========================================================
    def cb_read_motion(self, request, response):
        strategy_ini_path = "/home/iclab/ros2_adult/src/strategy/strategy/strategy.ini"
        
        try:
            if os.path.exists(strategy_ini_path):
                with open(strategy_ini_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip() # 讀取並去除前後空白/換行
               
                new_loc = ""
                if "Parameter" in content:
                    parts = content.split('Parameter')
                    raw_loc = parts[0] # 拿到 "sp/" 或 "/sp/"
                    new_loc = raw_loc.replace("/", "").strip() # 去除斜線 -> "sp"
                else:
                    new_loc = content.replace("/", "").strip()

                # ★ 執行切換邏輯
                if new_loc and new_loc != self.location_folder:
                    self.get_logger().info(f"🔄 [Auto-Switch] 原始內容 '{content}' -> 解析為 '{new_loc}'")
                    self.get_logger().info(f"🔄 偵測到策略變更: {self.location_folder} -> {new_loc}")
                    
                    self.location_folder = new_loc
                    
                    # 清空舊記憶體
                    with self.joints_lock:
                        self.saved_sectors.clear()
                        self.id_source_map.clear()
                    
                    # 重載
                    self.load_startup_stand_motion()
                    self.load_all_strategy_motions()
                    self.get_logger().info(f"✅ 已切換至 {self.location_folder} 並完成重載")

        except Exception as e:
            self.get_logger().warn(f"[Check Strategy] 讀取 strategy.ini 失敗: {e}")

        filename = request.name
        read_state = request.readstate 
        home_path = os.path.expanduser("~")
        
        if read_state == 1:
            base_path = self.stand_dir
        else:
            # 這裡現在會是正確的 location (例如 sp)
            base_path = os.path.join(home_path, "ros2_adult/src/strategy/strategy", self.location_folder, "Parameter")

        full_path = os.path.join(base_path, f"{filename}.ini")
        self.get_logger().info(f"[Read] 嘗試讀取檔案: {full_path}")

        if not os.path.exists(full_path):
            self.get_logger().error(f"[Read] 找不到檔案: {full_path}")
            response.readcheck = False
            return response

        # 1. 載入到 RAM
        self._internal_load_ini(full_path, f"{filename}.ini", is_common=False)

        # 2. 回填 Response
        config = configparser.ConfigParser()
        try:
            config.read(full_path)
            response.vectorcnt = 0
            response.motionstate = []
            response.id = []
            response.motionlist = []
            response.relativedata = []
            response.absolutedata = []
            response.item_names = []

            for section in config.sections():
                try:
                    m_id = int(config[section]['id'])
                    m_state = int(config[section]['motionstate'])
                    response.vectorcnt += 1
                    response.id.append(m_id)
                    response.motionstate.append(m_state)
                    response.item_names.append(config[section].get('item_name', ''))

                    def parse_list(key):
                        if key in config[section] and config[section][key]:
                            return [int(x) for x in config[section][key].split(',')]
                        return []

                    if m_state == 0: response.motionlist.extend(parse_list('motionlist'))
                    elif m_state in [1, 2]: response.relativedata.extend(parse_list('motordata'))
                    elif m_state in [3, 4]: response.absolutedata.extend(parse_list('motordata'))
                except: pass
            
            response.readcheck = True
            self.get_logger().info(f"[Read] 回傳 {response.vectorcnt} 筆數據給網頁")
        except Exception as e:
            self.get_logger().error(f"[Read] Response 打包錯誤: {e}")
            response.readcheck = False

        return response

    def joint_state_cb(self, msg: JointState):
        with self.joints_lock:
            for i, name in enumerate(msg.name):
                try: self.current_joints[int(name)] = int(msg.position[i])
                except: pass

    def publish_command(self, target_joints_dict):
        if not target_joints_dict: return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = []
        msg.position = []
        msg.velocity = [] # ★ 新增：初始化速度陣列
        
        # ★ 修改：解包 (pos, vel)
        for mid, (pos, vel) in target_joints_dict.items():
            msg.name.append(str(mid))
            msg.position.append(float(pos))
            msg.velocity.append(float(vel)) # ★ 新增：填入速度
            
        self.cmd_pub.publish(msg)

    def cb_single_motor(self, msg: SingleMotorData):
        mid = int(msg.id)
        pos = int(msg.position)  # 這是相對位移量
        spd = int(msg.speed)

        # --- 修改重點：強制讀取馬達當前的真實位置 ---
        # 使用 .get(mid, 2048) 確保如果沒讀到數值時有個預設值
        if mid in self.current_joints:
            base_pos = self.current_joints[mid]
        else:
            # 如果連 current_joints 都沒抓到，才考慮用 last_goals 或預設中立點
            base_pos = self.last_goals.get(mid, 2048)
            self.get_logger().warn(f"[SingleMotor] Motor {mid} no feedback, using last goal/default.")
        # ---------------------------------------

        final_target = base_pos + pos

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [str(mid)]
        js.position = [float(final_target)]
        js.velocity = [float(spd)]
        self.cmd_pub.publish(js)

        # 更新最後目標值，確保其他邏輯同步
        self.last_goals[mid] = final_target

        # 通知 WalkingNode 定錨點已改變
        reset_msg = Bool()
        reset_msg.data = True
        self.anchor_reset_pub.publish(reset_msg)

        self.get_logger().info(f"[SingleMotor] ID={mid} Base={base_pos} Target={final_target} (Rel={pos})")
        
    def cb_param_update(self, params):
        for p in params:
            if p.name == "location" and p.type_ == p.Type.STRING:
                new_location = p.value
                
                # 只有當路徑真的改變時才執行重載，避免無意義的消耗
                if new_location != self.location_folder:
                    self.get_logger().info(f"🔄 收到策略切換指令: {self.location_folder} -> {new_location}")
                    
                    # 1. 更新路徑變數
                    self.location_folder = new_location
                    
                    # 2. 清空現有記憶體 (重要！避免新舊策略混雜)
                    with self.joints_lock: # 建議加上鎖，避免讀取時剛好被清空
                        self.saved_sectors.clear()
                        self.id_source_map.clear()
                    
                    # 3. 重新載入檔案
                    self.get_logger().info("正在重新載入策略檔案...")
                    self.load_startup_stand_motion() # 重新載入站姿 (通常是通用的)
                    self.load_all_strategy_motions() # 載入新策略資料夾
                    
                    self.get_logger().info(f"✅ 策略已切換為: {self.location_folder}")

        return SetParametersResult(successful=True)

    def cb_save_motion(self, msg):
        filename = msg.name if msg.name else "untitled"
        if filename not in self.file_save_buffer: self.file_save_buffer[filename] = []
        if msg.saveflag:
            self.write_to_ini(filename, msg.savestate)
            self.file_save_buffer[filename] = []
        else:
            record = {'savestate': msg.savestate, 'motionstate': msg.motionstate, 'id': msg.id, 
                      'item_name': getattr(msg, 'item_name', ''), 'motionlist': list(msg.motionlist), 'motordata': list(msg.motordata)}
            self.file_save_buffer[filename].append(record)

    def write_to_ini(self, filename, savestate):
        if savestate == 1:
            base_path = self.stand_dir
        else:
            home_path = os.path.expanduser("~")
            base_path = os.path.join(home_path, "ros2_adult/src/strategy/strategy", self.location_folder, "Parameter")

        if not os.path.exists(base_path): os.makedirs(base_path)
        full_path = os.path.join(base_path, f"{filename}.ini")
        
        # =========================================================
        # ★ 核心修改：建立一個全新的空字典 (database_to_save)
        # 不再去讀取 os.path.exists(full_path) 的舊檔案
        # 這樣就能保證檔案內容 = 網頁傳過來的 Buffer 內容
        # =========================================================
        database_to_save = {} 
        
        # 讀取這次 Buffer 裡面的資料
        if filename in self.file_save_buffer:
            for record in self.file_save_buffer[filename]:
                data_entry = {
                    'savestate': str(record['savestate']),
                    'motionstate': str(record['motionstate']),
                    'id': str(record['id']),
                    'item_name': str(record['item_name']),
                    'motionlist': ','.join(map(str, record['motionlist'])),
                    'motordata': ','.join(map(str, record['motordata']))
                }
                
                # 我們依然使用 "ID_State" 作為 Key 來暫存
                # 這可以避免網頁端如果不小心送了兩次重複的資料，這裡會自動過濾剩下一筆
                key = f"{record['id']}_{record['motionstate']}"
                database_to_save[key] = data_entry
            
        # 準備寫入檔案
        new_config = configparser.ConfigParser()
        
        # 將整理好的資料依序填入 Config，Section 名稱流水號化 (0, 1, 2...)
        for idx, key in enumerate(database_to_save): 
            new_config[str(idx)] = database_to_save[key]
            
        try:
            with open(full_path, 'w', encoding='utf-8') as f: 
                new_config.write(f)
            self.get_logger().info(f"Saved (Overwrite Mode): {full_path} | 筆數: {len(database_to_save)}")
        except Exception as e: 
            self.get_logger().error(f"Save Failed: {e}")

    def cb_check_sector(self, request, response):
        response.checkflag = str(request.data) in self.saved_sectors
        return response

    def cb_sector_execute(self, msg):
        sector_id = str(msg.data)

        if sector_id == "29":
            full_path = "/home/iclab/ros2_adult/src/strategy/strategy/Parameter/stand.ini"
            self.get_logger().info(f"[Stand] Execute -> Force reload: {full_path}")

            # 先清掉 RAM 裡舊的 0，避免默默用舊資料
            with self.joints_lock:
                self.saved_sectors.pop("0", None)
                self.id_source_map.pop("0", None)

            # 讀 stand.ini
            self._internal_load_ini(full_path, "stand.ini", is_common=True)

            # ★強制：不管 stand.ini 的 id 是多少，都把它當作 sector 0
            # 找出 stand.ini 這次載入後新增/更新的內容，選一個最像站姿的（通常是 242/243）
            # 你如果確定 stand.ini 裡只有一筆站姿，這樣做最穩。
            stand_candidate = None
            for sid, rec in self.saved_sectors.items():
                if self.id_source_map.get(sid) == "stand.ini" and rec.get("opcode") in (242, 243):
                    stand_candidate = rec
                    break

            if stand_candidate is None:
                self.get_logger().error("[Stand] Reload OK but cannot find stand data from stand.ini")
                return

            with self.joints_lock:
                self.saved_sectors["0"] = stand_candidate
                self.id_source_map["0"] = "stand.ini"

            self.get_logger().info("[Stand] sector 0 overwritten from stand.ini")

        # 下面保持原本執行流程
        if sector_id in self.saved_sectors:
            record = self.saved_sectors[sector_id]
            if record['opcode'] == 242:
                self.execute_pose(record['data'], "ABSOLUTE")
            elif record['opcode'] == 243:
                self.execute_pose(record['data'], "RELATIVE")
            elif record['opcode'] == 244:
                self.execute_motion_list(record['data'])
        else:
            self.get_logger().warn(f"[Execute] Sector ID {sector_id} not found in RAM (saved_sectors).")

    def execute_motion_list(self, data):
        self.get_logger().info(f"[MotionList] 收到清單長度: {len(data)}, 內容: {data}")
        for i in range(0, len(data), 2):
            if i+1 >= len(data): break
            sec_id = str(data[i])
            delay = float(data[i+1])/1000.0
            
            self.get_logger().info(f" -> Step {i//2 + 1}: ID={sec_id}, Delay={data[i+1]}")

            if sec_id == '0':
                self.get_logger().info(" -> 遇到 ID 0，跳過動作")
            elif sec_id in self.saved_sectors:
                rec = self.saved_sectors[sec_id]
                self.execute_pose(rec['data'], "ABSOLUTE" if rec['opcode']==242 else "RELATIVE", False)
            else:
                self.get_logger().warn(f" -> [警告] 找不到 Sector ID: {sec_id}，跳過！")

            time.sleep(max(0.05, delay))
        self.get_logger().info("[MotionList] 執行完畢，發送 Callback")
        msg = Bool(); msg.data = True; self.execute_callback_pub.publish(msg)

    def execute_pose(self, data, mode, trigger_callback=True):
        count = len(data) // 2
        target_joints = {} # 結構：{id: (pos, vel)}
        
        for i in range(count):
            # 提取速度 (偶數索引)
            raw_spd = data[i*2]
            
            # 提取位置 (奇數索引)
            raw_val = data[i*2+1]
            
            # 處理負數轉換 (16-bit signed)
            if raw_val > 1000000: raw_val -= 65536
            mid = i + 1
            
            # --- 修改核心邏輯 ---
            if mode == "RELATIVE":
                # 直接讀取當下真實的物理位置。如果讀不到(例如剛開機)，則預設為 2048
                base_pos = self.last_goals.get(mid, self.current_joints.get(mid, 2048))
                final_target = base_pos + raw_val
            else: 
                # ABSOLUTE 模式：直接使用目標值
                final_target = raw_val
            # ------------------
            
            # 儲存目標位置與速度
            target_joints[mid] = (final_target, raw_spd)
            # 更新最後一次指令目標 (為了之後可能的 Absolute 計算)
            self.last_goals[mid] = final_target 

        # 發送指令
        self.publish_command(target_joints)
        
        # 通知 Walking Node：姿勢已改變，請重設錨點
        reset_msg = Bool()
        reset_msg.data = True
        self.anchor_reset_pub.publish(reset_msg)
        self.get_logger().info(f"[Notify] Mode: {mode}, Anchor Reset Flag sent to Walking Node.")

        if trigger_callback: 
            msg = Bool()
            msg.data = True
            self.execute_callback_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok(): time.sleep(1)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()