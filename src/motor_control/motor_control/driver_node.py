#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int16MultiArray
from tku_msgs.msg import HeadPackage 
import threading
from collections import defaultdict
from dynamixel_sdk import *

# ==== 系列位址定義 ====
# X-Series (XM540, XH540, XM430)
ADDR_X_TORQUE_ENABLE          = 64
ADDR_X_GOAL_POSITION          = 116
ADDR_X_PROFILE_VELOCITY       = 112
ADDR_X_PRESENT_POSITION       = 132
ADDR_X_INDIRECT_WRITE_START   = 578 

# P-Series / Pro-Series (H42-20)
ADDR_P_TORQUE_ENABLE          = 562
ADDR_P_GOAL_POSITION          = 596  
ADDR_P_PROFILE_VELOCITY       = 600  
ADDR_P_PRESENT_POSITION       = 611  
ADDR_P_INDIRECT_WRITE_START   = 49  

# 共通位址 (Indirect Data)
ADDR_INDIRECT_DATA_WRITE      = 634 
LEN_INDIRECT_DATA_WRITE       = 8   

PROTOCOL_VERSION              = 2.0
DEFAULT_BAUDRATE              = 1000000
ALL_TARGET_IDS                = list(range(1, 30))

class DynamixelDriver(Node):
    def __init__(self):
        super().__init__('dynamixel_driver_node')
        self.declare_parameter('baudrate', DEFAULT_BAUDRATE)
        self.declare_parameter('ports', ['/dev/U2D2_P1','/dev/U2D2_P2','/dev/U2D2_P3']) 
        self.baudrate = self.get_parameter('baudrate').value
        self.port_list = self.get_parameter('ports').value

        # 狀態儲存
        self.joint_data = defaultdict(lambda: {'present': 2048, 'goal': 2048, 'velocity': 0})
        self.head_map = {1: 28, 2: 29}
        self.data_lock = threading.Lock()
        self.torque_requests = []
        self.id_port_map = {} # 格式: {mid: (gw, gr, ph, is_p_series)}

        self._init_hardware()

        # ROS 介面
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_commands', self._command_cb, 10)
        self.torque_sub = self.create_subscription(Int16MultiArray, '/set_torque', self._torque_cb, 10)
        self.head_sub = self.create_subscription(HeadPackage, 'Head_Topic', self._head_cb, 10)

        # 建議走路控制提升至 20Hz (0.05s)，診斷時可維持 0.5s
        self.timer = self.create_timer(0.05, self._control_loop)

    def _init_hardware(self):
        pk = PacketHandler(PROTOCOL_VERSION)
        valid_ports = []
        for port_name in self.port_list:
            ph = PortHandler(port_name)
            if ph.openPort() and ph.setBaudRate(self.baudrate):
                self.get_logger().info(f"Port 成功開啟: {port_name}")
                valid_ports.append(ph)

        if not valid_ports:
            self.get_logger().error("無可用 Port，請檢查權限或連線！")
            return

        port_tools = {}
        for ph in valid_ports:
            port_tools[ph] = {
                'gw': GroupSyncWrite(ph, pk, ADDR_INDIRECT_DATA_WRITE, LEN_INDIRECT_DATA_WRITE),
                'gr_x': GroupSyncRead(ph, pk, ADDR_X_PRESENT_POSITION, 4),
                'gr_p': GroupSyncRead(ph, pk, ADDR_P_PRESENT_POSITION, 4)
            }

        for mid in ALL_TARGET_IDS:
            for ph in valid_ports:
                model_num, result, _ = pk.ping(ph, mid)
                if result == COMM_SUCCESS:
                    # --- 自動判定系列 ---
                    # 你的配置: 16-19, 22-25 是 H42 (Pro)，其餘是 X 系列
                    is_p_series = (16 <= mid <= 19 or 22 <= mid <= 25)
                    
                    read_addr = ADDR_P_PRESENT_POSITION if is_p_series else ADDR_X_PRESENT_POSITION
                    tools = port_tools[ph]
                    current_gr = tools['gr_p'] if is_p_series else tools['gr_x']
                    
                    # 讀取初始位置
                    val, result, _ = pk.read4ByteTxRx(ph, mid, read_addr)
                    if result == COMM_SUCCESS:
                        if val > 0x7FFFFFFF: val -= 4294967296
                        with self.data_lock:
                            self.joint_data[mid]['present'] = val
                            self.joint_data[mid]['goal'] = val
                    
                    current_gr.addParam(mid)
                    self.id_port_map[mid] = (tools['gw'], current_gr, ph, is_p_series)
                    self._setup_single_indirect_address(ph, pk, mid)
                    self.get_logger().info(f"ID {mid:02d} 初始化成功 ({'P-Series' if is_p_series else 'X-Series'})")
                    break

    def _setup_single_indirect_address(self, ph, pk, mid):
        """設定間接位址映射：將 Velocity 與 Goal Position 映射到 Indirect Data 區塊"""
        is_p = self.id_port_map[mid][3]
        base_idx = ADDR_P_INDIRECT_WRITE_START if is_p else ADDR_X_INDIRECT_WRITE_START
        src_vel = ADDR_P_PROFILE_VELOCITY if is_p else ADDR_X_PROFILE_VELOCITY
        src_pos = ADDR_P_GOAL_POSITION if is_p else ADDR_X_GOAL_POSITION
        torque_addr = ADDR_P_TORQUE_ENABLE if is_p else ADDR_X_TORQUE_ENABLE

        # 必須先關閉扭力才能寫入 Indirect Address
        pk.write1ByteTxRx(ph, mid, torque_addr, 0)

        # 映射 Profile Velocity (4 Bytes)
        for i in range(4):
            pk.write2ByteTxRx(ph, mid, base_idx + (i * 2), src_vel + i)
        # 映射 Goal Position (4 Bytes)
        for i in range(4):
            pk.write2ByteTxRx(ph, mid, base_idx + 8 + (i * 2), src_pos + i)

    def _command_cb(self, msg: JointState):
        with self.data_lock:
            for i, name in enumerate(msg.name):
                try:
                    mid = int(name)
                    if mid in self.id_port_map:
                        self.joint_data[mid]['goal'] = int(msg.position[i])
                        if len(msg.velocity) > i:
                            self.joint_data[mid]['velocity'] = int(msg.velocity[i])
                except: pass

    def _torque_cb(self, msg: Int16MultiArray):
        if len(msg.data) < 2: return
        target_id, state = msg.data[0], msg.data[1]
        with self.data_lock:
            ids = [target_id] if target_id != 0 else list(self.id_port_map.keys())
            for mid in ids:
                self.torque_requests.append((mid, state))

    def _head_cb(self, msg: HeadPackage):
        real_id = self.head_map.get(msg.id)
        if real_id in self.id_port_map:
            with self.data_lock:
                self.joint_data[real_id]['goal'] = int(msg.position)
                self.joint_data[real_id]['velocity'] = int(msg.speed)

    def _control_loop(self):
        pk = PacketHandler(PROTOCOL_VERSION)
        
        # --- 0. Torque 處理 ---
        current_reqs = []
        with self.data_lock:
            if self.torque_requests:
                current_reqs = self.torque_requests[:]
                self.torque_requests.clear()

        for mid, state in current_reqs:
            if mid in self.id_port_map:
                _, _, ph, is_p = self.id_port_map[mid]
                t_addr = ADDR_P_TORQUE_ENABLE if is_p else ADDR_X_TORQUE_ENABLE
                
                # 執行寫入
                res, error = pk.write1ByteTxRx(ph, mid, t_addr, state)
                
                if res != COMM_SUCCESS:
                    self.get_logger().error(f"ID {mid} Torque 狀態切換失敗: {pk.getTxRxResult(res)}")
                else:
                    self.get_logger().info(f"ID {mid} Torque 狀態已切換至 {state}")
                    
                    # 只有在「開啟」扭力後，才需要檢查 Indirect Address 並對齊位置
                    if state == 1:
                        # 檢查 Indirect Address 是否跑掉 (XM/XH 系列有時會因為掉電重設)
                        base_idx = ADDR_P_INDIRECT_WRITE_START if is_p else ADDR_X_INDIRECT_WRITE_START
                        expected = ADDR_P_PROFILE_VELOCITY if is_p else ADDR_X_PROFILE_VELOCITY
                        val, result, _ = pk.read2ByteTxRx(ph, mid, base_idx)
                        
                        if result != COMM_SUCCESS or val != expected:
                            self._setup_single_indirect_address(ph, pk, mid)
                        
                        # 對齊目標位置到當前位置，防止馬達開扭力後「暴衝」回舊目標
                        with self.data_lock:
                            self.joint_data[mid]['goal'] = self.joint_data[mid]['present']
                        
                        # (選擇性) 如果關閉後又重設了 Indirect，必須重新開扭力
                        pk.write1ByteTxRx(ph, mid, t_addr, 1)

        # --- 1. 並列讀取 (保持不變) ---
        active_ports = list(set(ph for _, _, ph, _ in self.id_port_map.values()))
        def read_task(ph):
            mids_on_port = [mid for mid, info in self.id_port_map.items() if info[2] == ph]
            unique_grs = set(info[1] for mid, info in self.id_port_map.items() if info[2] == ph)
            for gr in unique_grs:
                if gr.txRxPacket() != COMM_SUCCESS: continue
                updates = {}
                for mid in mids_on_port:
                    is_p = self.id_port_map[mid][3]
                    r_addr = ADDR_P_PRESENT_POSITION if is_p else ADDR_X_PRESENT_POSITION
                    if gr.isAvailable(mid, r_addr, 4):
                        val = gr.getData(mid, r_addr, 4)
                        if val > 0x7FFFFFFF: val -= 4294967296
                        updates[mid] = val
                with self.data_lock:
                    for mid, val in updates.items():
                        self.joint_data[mid]['present'] = val

        threads = [threading.Thread(target=read_task, args=(p,)) for p in active_ports]
        for t in threads: t.start()
        for t in threads: t.join()

        # --- 2. 發佈 ROS 狀態 ---
        pub_msg = JointState()
        pub_msg.header.stamp = self.get_clock().now().to_msg()
        with self.data_lock:
            sorted_ids = sorted(self.id_port_map.keys())
            pub_msg.name = [str(i) for i in sorted_ids]
            pub_msg.position = [float(self.joint_data[i]['present']) for i in sorted_ids]
        self.joint_pub.publish(pub_msg)

        # --- 3. 並列寫入 (這部分加入了速度數值監控) ---
        def write_task(ph):
            mids_on_port = [mid for mid, info in self.id_port_map.items() if info[2] == ph]
            if not mids_on_port: return
            gw = self.id_port_map[mids_on_port[0]][0]
            gw.clearParam()
            
            with self.data_lock:
                for mid in mids_on_port:
                    # 1. 雖然接收了 raw_vel，但我們不再使用它來限制速度
                    raw_vel = max(0, self.joint_data[mid]['velocity']) 
                    
                    # 2. 直接將輸出速度設為 0
                    # 在 Dynamixel 協議中，Profile Velocity = 0 代表「不限制速度 (Infinite Velocity)」
                    # if raw_vel == 0:
                    #     final_vel = 60  # 約 13 RPM，這是一個相對平穩且安全的起立速度
                    # else:
                    #     final_vel = raw_vel
                    
                    final_vel = raw_vel
                    
                    # 3. 獲取目標位置
                    goal = self.joint_data[mid]['goal']

                    # 封裝並傳送至馬達
                    param = int(final_vel).to_bytes(4, 'little', signed=True) + \
                            int(goal).to_bytes(4, 'little', signed=True)
                    gw.addParam(mid, param)
            
            gw.txPacket()

        threads = [threading.Thread(target=write_task, args=(p,)) for p in active_ports]
        for t in threads: t.start()
        for t in threads: t.join()

def main(args=None):
    rclpy.init(args=args)
    node = DynamixelDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
