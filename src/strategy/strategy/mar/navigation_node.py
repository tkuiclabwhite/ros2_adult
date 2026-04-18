import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from pupil_apriltags import Detector
from ultralytics import YOLO

april_detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.5,
    refine_edges=True,
    decode_sharpening=0.25,
)
TAG_ACTION = {1: "FORWARD", 2: "RIGHT", 3: "LEFT"}

MIN_ARROW_AREA = 2000
MIN_DARK_AREA  = 1500
BLACK_THRESH   = 80
WHITE_THRESH   = 150
CONFIRM_FRAMES = 3


def find_arrow_boxes(frame, masked, black_thresh, min_dark_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_f, w_f = frame.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    median_val = np.median(gray_eq)
    canny_lo = int(max(0, 0.5 * median_val))
    canny_hi = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray_eq, canny_lo, canny_hi)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_dark_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > w_f * 0.9 or h > h_f * 0.9:
            continue
        if w * h > w_f * h_f * 0.4:
            continue
        if not (0.4 < w / h < 2.5):
            continue
        roi_gray = gray[y:y+h, x:x+w]
        if np.mean(roi_gray) > 180:
            continue
        dark_px = np.sum(roi_gray < np.percentile(roi_gray, 50))
        if dark_px / (w * h) < 0.25:
            continue
        roi_bgr_check = masked[y:y+h, x:x+w]
        gray_mean = cv2.mean(cv2.cvtColor(roi_bgr_check, cv2.COLOR_BGR2GRAY))[0]
        if 115 < gray_mean < 140:
            continue
        boxes.append((x, y, w, h))
    return boxes


def detect_arrow(roi_bgr, min_arrow_area):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    top_bright = np.percentile(gray, 95)
    bottom_dark = np.percentile(gray, 5)
    diff = top_bright - bottom_dark
    if diff < 30:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold = bottom_dark + diff * 0.6
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary) < min_arrow_area:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_arrow_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    h, w = binary.shape
    cx_i, cy_i = int(cx), int(cy)

    w_top    = cv2.countNonZero(binary[:cy_i, :])
    w_bottom = cv2.countNonZero(binary[cy_i:, :])
    tb_diff  = w_top - w_bottom

    inv = cv2.bitwise_not(binary)
    b_bottom_left  = cv2.countNonZero(inv[cy_i:, :cx_i])
    b_bottom_right = cv2.countNonZero(inv[cy_i:, cx_i:])

    center_band  = binary[:, w//3 : 2*w//3]
    center_ratio = cv2.countNonZero(center_band) / center_band.size

    if center_ratio > 0.25 and tb_diff > 0:
        return "FORWARD"
    if b_bottom_right > b_bottom_left:
        return "RIGHT"
    return "LEFT"


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.bridge = CvBridge()
        self.action_history = deque(maxlen=CONFIRM_FRAMES)
        self.last_confirmed_action = None

        # 載入 YOLO（備援）
        self.get_logger().info('載入 YOLO 模型...')
        self.yolo_model = YOLO('strategy/strategy/mar/best1.onnx')
        self.yolo_class_names = {0: "left", 1: "right", 2: "straight"}
        self.yolo_to_action  = {"left": "LEFT", "right": "RIGHT", "straight": "FORWARD"}
        self.conf_threshold  = 0.5
        self.get_logger().info('✅ 模型載入完成')

        # 訂閱
        self.image_sub = self.create_subscription(
            Image, '/camera1/image_raw', self.image_callback, 10)

        # 發布（與原 YOLO 節點相同 topic）
        self.class_id_pub = self.create_publisher(String, 'class_id_topic', 1)
        self.coord_pub    = self.create_publisher(Point,  'sign_coordinates', 1)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        cx_frame, cy_frame = w // 2, h // 2

        # ── 1. AprilTag（最高優先）────────────────────────
        april_results = april_detector.detect(gray)
        april_action = None
        april_corners_list = []
        for r in april_results:
            action = TAG_ACTION.get(r.tag_id)
            if action:
                april_action = action
                april_corners_list.append(r.corners.astype(int))

        # 遮蔽 AprilTag 區域
        masked = frame.copy()
        for corners in april_corners_list:
            cv2.fillPoly(masked, [corners], (127, 127, 127))

        # ── 2. YOLO（AprilTag 無時備援）──────────────────
        yolo_action = None
        yolo_center = None
        yolo_area   = None
        yolo_bbox_bottom = None
        if not april_action:
            results = self.yolo_model(frame, imgsz=640, verbose=False)
            candidates = []
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                if cls not in self.yolo_class_names or conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                candidates.append({
                    'name': self.yolo_class_names[cls],
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'area': area,
                    'bbox_bottom': y2,
                })
            if candidates:
                best = max(candidates, key=lambda x: x['area'])
                yolo_action      = self.yolo_to_action.get(best['name'])
                yolo_center      = best['center']
                yolo_area        = best['area']
                yolo_bbox_bottom = best['bbox_bottom']

        # ── 決策 ─────────────────────────────────────────
        current_action = april_action or yolo_action

        # 連續幀確認
        self.action_history.append(current_action)
        if (len(self.action_history) == CONFIRM_FRAMES and
                len(set(self.action_history)) == 1):
            confirmed = self.action_history[0]
            if confirmed and confirmed != self.last_confirmed_action:
                self.last_confirmed_action = confirmed
                source = "TAG" if april_action else "YOLO"
                self.get_logger().info(
                    f'ACTION: {confirmed}  [{source}]'
                )

                # 發布 class_id_topic
                if yolo_center and yolo_area:
                    cx, cy = yolo_center
                    data = f"{confirmed.lower()},{cx},{yolo_bbox_bottom},{yolo_area}"
                else:
                    data = f"{confirmed.lower()},{cx_frame},{cy_frame},0"

                msg_str = String()
                msg_str.data = data
                self.class_id_pub.publish(msg_str)

                # 發布 sign_coordinates
                msg_pt = Point()
                msg_pt.x = float(yolo_center[0]) if yolo_center else float(cx_frame)
                msg_pt.y = float(yolo_center[1]) if yolo_center else float(cy_frame)
                msg_pt.z = 0.0
                self.coord_pub.publish(msg_pt)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()