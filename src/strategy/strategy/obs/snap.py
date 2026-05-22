#!/usr/bin/env python3
#coding=utf-8
import rclpy
import time
import math
import numpy as np
import cv2
import heapq
from strategy.API import API
import os
import sys # 🌟 新增：用於結束程式
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# --- 狀態定義 ---
STATE_IDLE     = 0
STATE_SCANNING = 1
STATE_PLANNING = 2
STATE_DONE     = 3 # 🌟 將原本的 WALKING 改為 DONE

class HumanoidNavNode(API):
    def __init__(self):
        super().__init__('humanoid_nav_node')
        self.init_params()
        self.bridge = CvBridge()
        
        # 訂閱影像 (zoom_in)
        self.image_sub = self.create_subscription(Image, 'processed_image', self.image_callback, 10)
        # self.image_sub = self.create_subscription(Image, 'zoom_in', self.image_callback, 10)
        # 週期性主迴圈 (20Hz)
        self.timer = self.create_timer(0.05, self.main_loop)
        
        self.get_logger().info("Route Planner Initialized. Starting scan...")

    def init_params(self):
        self.state = STATE_IDLE
        self.current_scan_idx = 0
        self.scan_angles = [45, 0, -45] # 左, 中, 右
        self.captured_frames = {}
        
        self.scale = 3
        self.master_w = 300 * self.scale
        self.master_h = 200 * self.scale
        self.master_map = np.zeros((self.master_h, self.master_w, 3), dtype=np.uint8)
        
        self.path = []
        self.ready_to_capture = False

        # 🌟 修改 1：指定儲存路徑
        # 使用絕對路徑或根據你的 Linux 家目錄結構設定
        # 假設 ros2_adult 在你的家目錄下
        home = os.path.expanduser("~")
        self.save_folder = os.path.join(home, "ros2_adult/src/strategy/strategy/obs/RoutePlan")
        
        # 自動建立目錄
        os.makedirs(self.save_folder, exist_ok=True)
        self.get_logger().info(f"Target folder: {self.save_folder}")

    def image_callback(self, msg):
        if self.state == STATE_SCANNING and self.ready_to_capture:
            self.ready_to_capture = False
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                angle = self.scan_angles[self.current_scan_idx]
                
                # 儲存原圖
                filename = os.path.join(self.save_folder, f"save_1_scan_{angle}.jpg")
                cv2.imwrite(filename, cv_image)
                
                self.captured_frames[angle] = cv_image
                self.get_logger().info(f"Saved raw image at {angle}°")
                
                self.current_scan_idx += 1
                if self.current_scan_idx >= len(self.scan_angles):
                    self.state = STATE_PLANNING
            except CvBridgeError as e:
                self.get_logger().error(str(e))

    def main_loop(self):
        if not self.is_start:
            self.sendHeadMotor(1, 2030, 40)
            self.sendHeadMotor(2, 2500, 40) 
            return

        if self.state == STATE_IDLE:
            self.sendBodySector(29)
            self.state = STATE_SCANNING
            self.current_scan_idx = 0
            self.captured_frames = {}

        elif self.state == STATE_SCANNING:
            angle = self.scan_angles[self.current_scan_idx]
            self.move_head_to_angle(angle) 

        elif self.state == STATE_PLANNING:
            self.get_logger().info("Generating Path and Saving Files...")
            self.generate_global_map()
            self.run_astar_and_save()
            
            # 🌟 修改 2：執行完畢後直接關閉
            self.get_logger().info("Route Plan Completed. Shutting down node...")
            # 停止主迴圈
            self.timer.cancel()
            # 關閉 ROS 2 通訊並結束程式
            rclpy.shutdown()
            sys.exit(0)

    def move_head_to_angle(self, angle):
        # 根據你的舵機係數轉換角度
        target_pos = 2030 + int(angle / 0.088)
        print(f"target_pos{target_pos}")
        self.sendHeadMotor(1, target_pos, 40)
        self.sendHeadMotor(2, 2500, 40)
        time.sleep(5)
        self.ready_to_capture = True  # 頭已到位，允許拍照

    # --- [影像處理與 A* 邏輯維持原樣，僅更新存檔路徑] ---
    def generate_global_map(self):
        self.master_map.fill(0)
        robot_in_master = (self.master_w // 2, self.master_h - 10)
        
        src_pts = np.float32([[115, 73], [202, 76], [10, 237], [307, 239]])
        real_w, real_h = 100 * self.scale, 150 * self.scale
        dst_pts = np.float32([[0, 0], [real_w, 0], [0, real_h], [real_w, real_h]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 🌟 將量測好的 3 個角度邊界放在這裡
        roi_pts_dict = {
            45: np.array([[274, 15], [320, 0], [320, 240], [16, 132]], np.int32),
            0: np.array([[52, 15], [307, 16], [316, 26], [4, 61]], np.int32),
            -45: np.array([[0, 0], [84, 10], [315, 107], [0, 240]], np.int32),
        }   

        for angle, frame in self.captured_frames.items():
            img_320 = cv2.resize(frame, (320, 240))
            cv2.imwrite(os.path.join(self.save_folder, f"save_3_processed_{angle}.jpg"), img_320)
            
            # 🌟 順序修正：1. 先提取底緣！(此時牆壁底邊是完整的，加上凸包運算，絕對不會掉進破洞)
            flattened = self.extract_bottom_edge(img_320, [128, 0, 128], thickness=3)
            
            # 🌟 順序修正：2. 提取出底線後，再套用黃色邊界過濾，切掉場外雜訊！
            roi_pts = roi_pts_dict.get(angle, np.array([[0, 240], [320, 240], [320, 0], [0, 0]], np.int32))
            roi_mask = np.zeros_like(flattened)
            cv2.fillPoly(roi_mask, [roi_pts], (255, 255, 255))
            flattened = cv2.bitwise_and(flattened, roi_mask)
            
            # 3. 轉為鳥瞰圖
            bev = cv2.warpPerspective(flattened, matrix, (real_w, real_h), flags=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.save_folder, f"save_4_bev_{angle}.jpg"), bev)
            
            # 旋轉與合成
            bev_robot_pos = (50 * self.scale, 150 * self.scale)
            M_rot = cv2.getRotationMatrix2D(bev_robot_pos, -angle, 1.0)
            rotated_bev = cv2.warpAffine(bev, M_rot, (real_w, real_h), flags=cv2.INTER_NEAREST)
            
            offset_x = robot_in_master[0] - bev_robot_pos[0]
            offset_y = robot_in_master[1] - bev_robot_pos[1]
            M_shift = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            shifted_bev = cv2.warpAffine(rotated_bev, M_shift, (self.master_w, self.master_h), flags=cv2.INTER_NEAREST)
            
            self.master_map = cv2.bitwise_or(self.master_map, shifted_bev)

        cv2.imwrite(os.path.join(self.save_folder, "save_2_master_combined.jpg"), self.master_map)

    def run_astar_and_save(self):
        gray = cv2.cvtColor(self.master_map, cv2.COLOR_BGR2GRAY)
        _, obstacle_map = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        inf_px = 1 * self.scale
        inf_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, inf_px*2), max(1, inf_px*2)))
        inflated_map = cv2.dilate(obstacle_map, inf_kernel)
        
        dist_trans = cv2.distanceTransform(cv2.bitwise_not(inflated_map), cv2.DIST_L2, 5)
        cost_map = np.zeros_like(dist_trans, dtype=np.float32)
        influence_px = 25 * self.scale
        mask = (dist_trans < influence_px) & (inflated_map == 0)
        cost_map[mask] = (1.0 - (dist_trans[mask] / influence_px)) * 100

        # 🌟 修改：不再使用固定的 goal_pt，而是設定一條「終點線」
        start_pt = (self.master_w // 2, self.master_h - 15)
        goal_y_line = 20 # 只要走到 Y=20 (畫面上方) 就算抵達
        
        self.path = self.astar_path_planning(inflated_map, cost_map, start_pt, goal_y_line)

        # 儲存 A* 路徑圖
        display_map = cv2.cvtColor(inflated_map, cv2.COLOR_GRAY2BGR)
        if self.path:
            for i in range(len(self.path)-1):
                cv2.line(display_map, self.path[i], self.path[i+1], (255, 0, 0), 2)
            
            # 畫出起點 (綠點)
            cv2.circle(display_map, start_pt, 5, (0, 255, 0), -1)
            # 🌟 畫出動態的終點 (紅點，取路徑的最後一個點)
            cv2.circle(display_map, self.path[-1], 5, (0, 0, 255), -1)
            
            # 儲存 path.txt
            path_file = os.path.join(self.save_folder, "path.txt")
            with open(path_file, "w") as f:
                for pt in self.path:
                    f.write(f"{pt[0]},{pt[1]}\n")
                    
        cv2.imwrite(os.path.join(self.save_folder, "save_5_path_planning.jpg"), display_map)

    # 🌟 修改：參數傳入 goal_y (目標線)，而不是單一目標點
    def astar_path_planning(self, binary_map, gradient_map, start_pt, goal_y):
        h, w = binary_map.shape
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = []
        heapq.heappush(open_set, (0, 0, start_pt))
        came_from = {}
        g_score = {start_pt: 0}
        
        # 🌟 修改：計算距離目標線的「垂直距離」
        def heuristic(a):
            return abs(a[1] - goal_y)

        while open_set:
            _, current_g, current_pt = heapq.heappop(open_set)
            
            # 🌟 修改：終點判定！只要現在的 Y 座標走到或越過目標線，就直接收工回傳！
            if current_pt[1] <= goal_y:
                p = []
                while current_pt in came_from:
                    p.append(current_pt)
                    current_pt = came_from[current_pt]
                p.append(start_pt)
                return p[::-1]
                
            for dx, dy in neighbors:
                n_pt = (current_pt[0] + dx, current_pt[1] + dy)
                if 0 <= n_pt[0] < w and 0 <= n_pt[1] < h:
                    if binary_map[n_pt[1], n_pt[0]] < 128:
                        move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                        tentative_g = current_g + move_cost + gradient_map[n_pt[1], n_pt[0]]
                        if n_pt not in g_score or tentative_g < g_score[n_pt]:
                            came_from[n_pt] = current_pt
                            g_score[n_pt] = tentative_g
                            # f 值 = 目前累積代價 + 垂直距離預估
                            f = tentative_g + heuristic(n_pt)
                            heapq.heappush(open_set, (f, tentative_g, n_pt))
        return []
    
    def extract_bottom_edge(self, image, target_color_bgr, thickness=3):
        # 1. 抓出紫色遮罩並稍微平滑
        mask = cv2.inRange(image, np.array(target_color_bgr)-40, np.array(target_color_bgr)+40)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottom_edge_mask = np.zeros_like(mask)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                # 取得色塊的凸包 (簡化形狀)
                hull = cv2.convexHull(cnt)
                pts = hull[:, 0, :]
                
                # 🌟 數學魔法：尋找真正的「左下角」與「右下角」
                # 左下角的幾何特徵：y 很大 (在畫面下方)，x 很小 (在畫面左方) -> (y - x) 的值會最大
                bl_idx = np.argmax(pts[:, 1] - pts[:, 0])
                
                # 右下角的幾何特徵：y 很大 (在畫面下方)，x 很大 (在畫面右方) -> (y + x) 的值會最大
                br_idx = np.argmax(pts[:, 1] + pts[:, 0])
                
                bl = tuple(pts[bl_idx])
                br = tuple(pts[br_idx])
                
                # 🌟 只畫這兩點之間的連線 (保證是純粹的觸地底邊，無視垂直側邊)
                cv2.line(bottom_edge_mask, bl, br, 255, thickness)

        # 組合輸出
        processed_img = image.copy()
        processed_img[mask > 0] = [0, 0, 0] # 抹除原來的立體色塊
        processed_img[bottom_edge_mask > 0] = target_color_bgr # 畫上純平面的底線
        return processed_img
def main(args=None):
    rclpy.init(args=args)
    node = HumanoidNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()