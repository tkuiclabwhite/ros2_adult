import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from geometry_msgs.msg import Point # [移除] 不需要了
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOSignTracker(Node):
    def __init__(self):
        super().__init__('yolo_sign_tracker')

        self.image_sub = self.create_subscription(Image, '/image', self.image_callback, 10)
        
        # Publishers
        # [修改] 只保留 class_id_topic，因為它已經包含了 X 座標和面積
        self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)
        
        # [移除] 不需要再發布 Point 了
        # self.coord_pub = self.create_publisher(Point, 'sign_coordinates', 1)

        self.bridge = CvBridge()
        # 請確認你的權重檔名正確 (例如 'best.pt' 或 'yolov8n.pt')
        self.model = YOLO('strategy/strategy/mar/best(4).pt') 

        # Settings
        self.conf_threshold = 0.5
        self.class_names = { 0: "left", 1: "right", 2: "straight" }
        self.class_colors = { 0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0) }

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_and_display(image)
        except Exception as e:
            self.get_logger().error(f'Image processing failed: {e}')

    def process_and_display(self, image):
        results = self.model(image, verbose=False)
        detections = results[0].boxes
        
        img_h, img_w = image.shape[:2]
        candidates = []

        for result in detections:
            if result.xyxy is None or result.conf is None or result.cls is None:
                continue

            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())

            if class_id not in self.class_names:
                continue
            
            if confidence < self.conf_threshold:
                continue

            bbox = result.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, bbox)
            
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

        target_sign = None
        
        # 選擇面積最大的作為目標
        if len(candidates) > 0:
            target_sign = max(candidates, key=lambda x: x['area'])

        # === 發布邏輯 ===
        msg_str = String()

        if target_sign:
            # 有偵測到號誌：發送 "名稱,中心X,底部Y,面積"
            # 底部Y (bbox[3]) 雖然目前控制端沒用到，但保留著以免未來想判斷距離
            msg_str.data = f"{target_sign['name']},{target_sign['center'][0]},{target_sign['bbox'][3]},{target_sign['area']}"
            
            log_msg = (f"Target: {target_sign['name']} | "
                       f"X: {target_sign['center'][0]} | "
                       f"Area: {target_sign['area']}")
            self.get_logger().info(log_msg)
        else:
            # 沒有偵測到號誌：發送 "None,0,0,0"
            msg_str.data = "None,0,0,0"

        # 統一發布
        self.class_id_pub.publish(msg_str)

        # === 畫面繪製 (Debug 用) ===
        for cand in candidates:
            x1, y1, x2, y2 = cand['bbox']
            cx, cy = cand['center']
            cls_color = self.class_colors.get(cand['class_id'], (255, 255, 255))
            
            is_target = (cand == target_sign)

            if is_target:
                # 繪製目標 (粗框 + 標示)
                cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 3)
                cv2.circle(image, (cx, cy), 8, (0, 255, 255), -1)
                
                info_text = f"Target ({cx}, {cy}) A:{cand['area']}"
                cv2.putText(image, info_text, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 畫一條線指引中心
                cv2.line(image, (img_w // 2, img_h), (cx, cy), (0, 255, 255), 2)
                
                cv2.putText(image, cand['name'], (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cls_color, 2)
            else:
                # 繪製被忽略的物件 (細框 + 灰色)
                gray_color = (128, 128, 128) 
                cv2.rectangle(image, (x1, y1), (x2, y2), gray_color, 1)
                cv2.putText(image, f"Ignored ({cand['area']})", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray_color, 1)

        cv2.imshow("Sign Tracking", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOSignTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()