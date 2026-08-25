#!/usr/bin/env python3
"""拍照節點：訂閱各路影像，收到 /capture_photo 觸發訊息（內容是要拍哪一路的
topic 名稱）後，把該路目前最新一張影像存到 ros2_adult/snap/。

原始畫面的 topic 名稱會隨相機來源改變：
    usb -> /camera1/image_raw        zed -> /zed/zed_node/left/image_rect_color
所以「原始畫面」是一個代號，實際 topic 由 camera_source 參數決定。網頁沿用舊的
'camera1/image_raw' 字串也能運作（見 RAW_ALIASES），不需要跟著改。

參數（launch 用 --ros-args -p 傳入）：
    camera_source  'usb' 或 'zed'，決定原始畫面對應哪個 topic。預設 zed
    raw_topic      直接指定原始畫面 topic，非空字串時蓋過 camera_source
"""
import os
import re
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

SAVE_DIR = os.path.expanduser('~/ros2_adult/snap')

# 相機來源 -> 原始（未裁切、未處理）影像 topic
RAW_TOPICS = {
    'usb': '/camera1/image_raw',
    'zed': '/zed/zed_node/left/image_rect_color',
}

# 網頁送來這些字串時一律當成「原始畫面」，改相機來源不用動網頁
RAW_ALIASES = ('camera1/image_raw', '/camera1/image_raw', 'raw', 'image_raw')

# 除了原始畫面以外，固定會先訂閱好的影像（按下快門才訂閱會漏掉第一張）
STATIC_TOPICS = {
    '/processed_image':       'processed_image',   # 色模結果
    '/zoom_in':               'zoom_in',           # 裁切放大後的畫面
    '/depth_view':            'depth_view',        # 深度偽彩色（僅 zed）
    '/depth_processed_image': 'depth_processed',   # 深度分類結果（僅 zed）
    '/overlap_image':         'overlap',           # 疊合結果（僅 zed）
}


class PhotoCaptureNode(Node):
    def __init__(self):
        super().__init__('photo_capture_node')
        self.bridge = CvBridge()
        self.latest_frames = {}
        self._extra_subs = {}      # 動態訂閱的 topic -> Subscription

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.declare_parameter('camera_source', 'zed')
        self.declare_parameter('raw_topic', '')

        source = str(self.get_parameter('camera_source').value).strip().lower()
        override = str(self.get_parameter('raw_topic').value).strip()
        self.raw_topic = override or RAW_TOPICS.get(source, RAW_TOPICS['zed'])

        self.topic_labels = dict(STATIC_TOPICS)
        self.topic_labels[self.raw_topic] = 'image_raw'

        # depth=1：只要最新一張。原本的 10 在 960x600 下等於替每一路影像多押
        # 十幾 MB 的緩衝，而拍照永遠只用得到最後一幀。
        self._qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        for topic in self.topic_labels:
            self.create_subscription(
                Image, topic, self._make_image_callback(topic), self._qos)

        self.create_subscription(String, '/capture_photo', self.capture_callback, 10)

        self.get_logger().info(
            'PhotoCaptureNode ready (camera_source=%s, raw=%s), saving to %s'
            % (source, self.raw_topic, SAVE_DIR))

    def _make_image_callback(self, topic):
        def callback(msg):
            self.latest_frames[topic] = msg
        return callback

    def _resolve(self, requested):
        """把網頁送來的字串換成實際 topic 名稱。

        比對時忽略開頭的 '/'，因為網頁有些地方寫 'camera1/image_raw'、
        有些寫 '/processed_image'，兩種都要收。
        """
        req = requested.strip()
        if req in RAW_ALIASES:
            return self.raw_topic
        for topic in self.topic_labels:
            if topic.lstrip('/') == req.lstrip('/'):
                return topic
        return req if req.startswith('/') else '/' + req

    def _label_for(self, topic):
        label = self.topic_labels.get(topic)
        if label:
            return label
        # 動態訂閱的 topic：拿 topic 名稱湊一個能當檔名的標籤
        return re.sub(r'[^0-9A-Za-z]+', '_', topic).strip('_') or 'image'

    def capture_callback(self, msg):
        topic = self._resolve(msg.data)

        frame_msg = self.latest_frames.get(topic)
        if frame_msg is None:
            if topic not in self.topic_labels and topic not in self._extra_subs:
                # 沒預先訂閱的 topic：現在補訂閱，下一次按快門就有畫面
                self._extra_subs[topic] = self.create_subscription(
                    Image, topic, self._make_image_callback(topic), self._qos)
                self.get_logger().warn(
                    '%s not subscribed yet, subscribing now — press again' % topic)
            else:
                self.get_logger().warn('No frame received yet for %s' % topic)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(frame_msg, 'bgr8')
        except Exception as e:
            # 深度圖之類的非彩色編碼轉 bgr8 會失敗，原樣存檔至少留得下資料
            self.get_logger().warn('bgr8 convert failed for %s (%s), passthrough'
                                   % (topic, e))
            try:
                cv_image = self.bridge.imgmsg_to_cv2(frame_msg, 'passthrough')
            except Exception as e2:
                self.get_logger().error('Cannot decode %s: %s' % (topic, e2))
                return

        filename = 'photo_%s_%d.jpg' % (self._label_for(topic), int(time.time()))
        save_path = os.path.join(SAVE_DIR, filename)
        if cv2.imwrite(save_path, cv_image):
            self.get_logger().info('Saved photo to %s' % save_path)
        else:
            self.get_logger().error('cv2.imwrite failed for %s' % save_path)


def main(args=None):
    rclpy.init(args=args)
    node = PhotoCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
