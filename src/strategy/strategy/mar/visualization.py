# # import rclpy
# # from rclpy.node import Node
# # from sensor_msgs.msg import Image
# # from std_msgs.msg import String
# # from cv_bridge import CvBridge
# # import cv2
# # import numpy as np
# # from ultralytics import YOLO

# # class YOLOSignTracker(Node):
# #     def __init__(self):
# #         super().__init__('yolo_sign_tracker')

# #         # 1. 訂閱影像 (不需要訂閱 line_positions 了)
# #         self.image_sub = self.create_subscription(Image, '/image', self.image_callback, 10)

# #         # 2. 發佈 class_id (供決策使用)
# #         self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)

# #         self.bridge = CvBridge()
# #         # 請確認路徑是否正確，建議用絕對路徑
# #         self.model = YOLO('best(4).pt') 
        
# #         # 定義您關注的類別名稱
# #         self.class_names = {
# #             0: "left", 1: "right", 2: "straight"
# #         }

# #         # 定義顏色 (BGR)
# #         self.class_colors = {
# #             0: (255, 0, 0),    # Blue
# #             1: (0, 0, 255),    # Red
# #             2: (0, 255, 0)     # Green
# #         }

# #     def image_callback(self, msg):
# #         # 將 ROS 影像轉為 OpenCV 格式
# #         image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
# #         self.process_and_display(image)

# #     def process_and_display(self, image):
# #         results = self.model(image, verbose=False)
# #         detections = results[0].boxes
        
# #         found_ids = []
        
# #         # 取得畫面中心 (用於計算偏差值)
# #         img_h, img_w = image.shape[:2]
# #         img_center_x = img_w // 2

# #         for result in detections:
# #             if result.xyxy is None or result.conf is None or result.cls is None:
# #                 continue

# #             # 取得數值
# #             bbox = result.xyxy[0].tolist()
# #             confidence = result.conf[0].item()
# #             class_id = int(result.cls[0].item())

# #             # 只處理我們定義過的類別 (左、右、直)
# #             if class_id not in self.class_names:
# #                 continue

# #             x_min, y_min, x_max, y_max = map(int, bbox)
            
# #             # --- 關鍵修改：計算中心座標 ---
# #             center_x = int((x_min + x_max) / 2)
# #             center_y = int((y_min + y_max) / 2)
            
# #             # 計算面積 (可以用來判斷距離：面積越大代表越近)
# #             area = (x_max - x_min) * (y_max - y_min)

# #             # --- 1. 終端機顯示 (Terminal) ---
# #             # 格式：類別 | 座標(x,y) | 面積(距離參考) | 信心度
# #             log_msg = (f"偵測到: {self.class_names[class_id]} | "
# #                        f"座標: ({center_x}, {center_y}) | "
# #                        f"面積: {area} | Conf: {confidence:.2f}")
# #             self.get_logger().info(log_msg)

# #             # 收集 ID 用於發佈
# #             found_ids.append(str(class_id))
            
# #             # --- 2. 畫面繪製 (Visual) ---
# #             color = self.class_colors.get(class_id, (255, 255, 255))
            
# #             # 畫框
# #             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
# #             # 畫中心點 (黃色實心圓點)
# #             cv2.circle(image, (center_x, center_y), 5, (0, 255, 255), -1)
            
# #             # 顯示座標文字在中心點旁邊
# #             coord_text = f"({center_x}, {center_y})"
# #             cv2.putText(image, coord_text, (center_x + 10, center_y), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# #             # 顯示類別名稱在框頂
# #             label = f"{self.class_names[class_id]}"
# #             cv2.putText(image, label, (x_min, y_min - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
# #             # (選用) 畫一條從畫面中心連到物件中心的線，幫助除錯「對準」
# #             cv2.line(image, (img_center_x, img_h), (center_x, center_y), (255, 255, 255), 1)

# #         # 顯示畫面
# #         cv2.imshow("Sign Tracking", image)
# #         cv2.waitKey(1)

# #         # 發佈 ID
# #         if found_ids:
# #             msg = String()
# #             msg.data = ",".join(found_ids)
# #             self.class_id_pub.publish(msg)

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = YOLOSignTracker()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()
# #         cv2.destroyAllWindows()

# # if __name__ == '__main__':
# #     main()
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from ultralytics import YOLO

# class YOLOSignTracker(Node):
#     def __init__(self):
#         super().__init__('yolo_sign_tracker')

#         self.image_sub = self.create_subscription(Image, '/camera1/image_raw', self.image_callback, 10)
#         self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)

#         self.bridge = CvBridge()
#         # self.model = YOLO('best(4).pt') #
#         self.model = YOLO('strategy/strategy/mar/best(4).pt') 

#         # --- 設定區 ---
#         self.conf_threshold = 0.5  # 信心值門檻：低於 0.5 的直接無視
#         self.class_names = { 0: "left", 1: "right", 2: "straight" }
#         self.class_colors = { 0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0) }

#     # def image_callback(self, msg):
#     #     image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#     #     self.process_and_display(image)
#     def image_callback(self, msg: Image):
#         try:
#             # Convert ROS Image to OpenCV BGR image
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             # Perform inference on the frame
#             results = self.model(frame, imgsz=320)
#             # Annotate
#             annotated = results[0].plot()
#             cv2.imshow('YOLO Detection', annotated)
#             cv2.waitKey(1)
#         finally:
#             # Cleanup per-frame to release memory
#             del results
#             torch.cuda.empty_cache()
#             #gc.collect()

#     def process_and_display(self, image):
#         results = self.model(image, verbose=False)
#         detections = results[0].boxes
        
#         img_h, img_w = image.shape[:2]
        
#         # 1. 建立一個清單來存放所有「合格」的候選人
#         candidates = []

#         for result in detections:
#             if result.xyxy is None or result.conf is None or result.cls is None:
#                 continue

#             confidence = result.conf[0].item()
#             class_id = int(result.cls[0].item())

#             # 過濾 1: 類別是否在名單內
#             if class_id not in self.class_names:
#                 continue
            
#             # 過濾 2: 信心值是否足夠 (本局新增)
#             if confidence < self.conf_threshold:
#                 continue

#             bbox = result.xyxy[0].tolist()
#             x_min, y_min, x_max, y_max = map(int, bbox)
            
#             # 計算面積與中心
#             width = x_max - x_min
#             height = y_max - y_min
#             area = width * height
#             center_x = int((x_min + x_max) / 2)
#             center_y = int((y_min + y_max) / 2)

#             # 將資訊存包起來
#             candidates.append({
#                 'class_id': class_id,
#                 'confidence': confidence,
#                 'bbox': (x_min, y_min, x_max, y_max),
#                 'center': (center_x, center_y),
#                 'area': area,
#                 'name': self.class_names[class_id]
#             })

#         # 2. 比較並選出「主角」
#         target_sign = None
        
#         if len(candidates) > 0:
#             # 使用 lambda 語法，依照 'area' 找出最大的一個
#             target_sign = max(candidates, key=lambda x: x['area'])

#         # 3. 處理與顯示
#         # 如果有找到目標，才發佈訊息
#         if target_sign:
#             # --- 終端機顯示 (只顯示最大的那個) ---
#             log_msg = (f"鎖定目標: {target_sign['name']} | "
#                        f"座標: {target_sign['center']} | "
#                        f"面積: {target_sign['area']} | "
#                        f"Conf: {target_sign['confidence']:.2f}")
#             self.get_logger().info(log_msg)

#             # --- 發佈 Topic ---
#             msg = String()
#             # msg.data = str(target_sign['class_id'])
#             # msg.data = f"{target_sign['name']},{target_sign['center'][0]},{target_sign['bbox'][3]}"
#             # 格式: "類別名稱,中心x,底部y,面積"
#             # 例如: "left,160,200,4500"
#             msg.data = f"{target_sign['name']},{target_sign['center'][0]},{target_sign['bbox'][3]},{target_sign['area']}"

#             self.class_id_pub.publish(msg)

#         # --- 畫面繪製 (Loop 畫出所有候選人，但特別標註主角) ---
#         for cand in candidates:
#             x1, y1, x2, y2 = cand['bbox']
#             cx, cy = cand['center']
#             cls_color = self.class_colors.get(cand['class_id'], (255, 255, 255))
            
#             is_target = (cand == target_sign)

#             if is_target:
#                 # === 主角 (最大) 的畫法：粗框、顯示座標、連線 ===
#                 cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 3) # 框變粗
#                 cv2.circle(image, (cx, cy), 8, (0, 255, 255), -1)      # 圓點變大
                
#                 # 標示座標與面積
#                 info_text = f"Target ({cx}, {cy}) A:{cand['area']}"
#                 cv2.putText(image, info_text, (x1, y1 - 25), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
#                 # 畫一條引導線 (從畫面下方連到目標)
#                 cv2.line(image, (img_w // 2, img_h), (cx, cy), (0, 255, 255), 2)
                
#                 # 標示類別
#                 cv2.putText(image, cand['name'], (x1, y1 - 5), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, cls_color, 2)
#             else:
#                 # === 其他 (較遠/較小) 的畫法：細框、灰階處理 ===
#                 # 這樣你可以知道「有看到」，但程式正在「忽略」它
#                 gray_color = (128, 128, 128) 
#                 cv2.rectangle(image, (x1, y1), (x2, y2), gray_color, 1) # 細框
#                 cv2.putText(image, f"Ignored ({cand['area']})", (x1, y1 - 5), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray_color, 1)

#         # cv2.imshow("Sign Tracking (Priority)", image)
#         # cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = YOLOSignTracker()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point  # <--- 新增這裡：引入 Point 格式
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOSignTracker(Node):
    def __init__(self):
        super().__init__('yolo_sign_tracker')

        # 訂閱相機影像
        self.image_sub = self.create_subscription(
            Image, 
            '/camera1/image_raw', 
            self.image_callback, 
            10
        )
        # --- 新增這裡：座標專用的 Publisher ---
        # 使用 Point 格式，Topic 名稱叫做 'sign_coordinates'
        self.coord_pub = self.create_publisher(Point, 'sign_coordinates', 1)

        # 發布策略結果 (給運動控制用)
        self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)
        
        # # 發布除錯影像 (給人類看畫面用)
        # self.debug_img_pub = self.create_publisher(Image, '/yolo/debug_image', 10)

        self.bridge = CvBridge()
        
        # 載入模型
        self.get_logger().info('正在載入 YOLO 模型...')
        # 請確認路徑是否正確
        self.model = YOLO('strategy/strategy/mar/best1.onnx') 
        self.get_logger().info('✅ 模型載入完成！')

        # --- 設定區 ---
        self.conf_threshold = 0.5  # 信心值門檻
        self.class_names = { 0: "left", 1: "right", 2: "straight" }
        self.class_colors = { 0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0) }

    def image_callback(self, msg: Image):
        try:
            # 將 ROS 影像轉為 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 呼叫核心邏輯
            self.process_and_display(frame)
        except Exception as e:
            self.get_logger().error(f'影像處理發生錯誤: {str(e)}')

    def process_and_display(self, image):
        # 1. 執行推論
        results = self.model(image, imgsz=640, verbose=False)
        detections = results[0].boxes
        
        img_h, img_w = image.shape[:2]
        
        # 2. 建立一個清單來存放所有「合格」的候選人
        candidates = []

        for result in detections:
            if result.xyxy is None or result.conf is None or result.cls is None:
                continue

            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())

            # 過濾
            if class_id not in self.class_names:
                continue
            if confidence < self.conf_threshold:
                continue

            bbox = result.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # 計算面積與中心
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            candidates.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': (x_min, y_min, x_max, y_max),
                'center': (center_x, center_y),
                'area': area,
                'name': self.class_names[class_id]
            })

        # 3. 比較並選出「主角」
        target_sign = None
        if len(candidates) > 0:
            target_sign = max(candidates, key=lambda x: x['area'])

        # 4. 處理與發佈控制訊號
        if target_sign:
            # 取得座標
            cx, cy = target_sign['center']

            # --- [修改處] 終端機顯示座標 ---
            log_msg = (f"鎖定: {target_sign['name']} | "
                       f"座標: ({cx:3d}, {cy:3d}) | "  # 這裡加入了座標顯示
                       f"Area: {target_sign['area']}")
            self.get_logger().info(log_msg)

            # --- 發佈 Topic ---
            msg = String()
            msg.data = f"{target_sign['name']},{cx},{target_sign['bbox'][3]},{target_sign['area']}"
            self.class_id_pub.publish(msg)
            # 3. --- 新增這裡：發布座標 Point Topic ---
            msg_point = Point()
            # Point 格式是 float，所以轉型一下比較安全
            # x 對應 center_x, y 對應 center_y
            msg_point.x = float(target_sign['center'][0]) 
            msg_point.y = float(target_sign['center'][1])
            msg_point.z = 0.0  # 2D 影像沒有深度，設為 0
            
            self.coord_pub.publish(msg_point)


        # 5. 畫面繪製 (準備傳回筆電看)
        for cand in candidates:
            x1, y1, x2, y2 = cand['bbox']
            cx, cy = cand['center']
            cls_color = self.class_colors.get(cand['class_id'], (255, 255, 255))
            
            is_target = (cand == target_sign)

            if is_target:
                # === 主角 ===
                cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 3) 
                cv2.circle(image, (cx, cy), 8, (0, 255, 255), -1)     
                
                info_text = f"Target ({cx}, {cy})"
                cv2.putText(image, info_text, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.line(image, (img_w // 2, img_h), (cx, cy), (0, 255, 255), 2)
                cv2.putText(image, cand['name'], (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cls_color, 2)
            else:
                # === 其他 ===
                gray_color = (128, 128, 128) 
                cv2.rectangle(image, (x1, y1), (x2, y2), gray_color, 1)
                cv2.putText(image, f"Ignored", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray_color, 1)

        # # Publish 除錯影像
        # debug_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        # self.debug_img_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOSignTracker()
    try:
        print("--- YOLO 節點執行中，終端機將顯示目標座標 ---")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()

if __name__ == '__main__':
    main()
