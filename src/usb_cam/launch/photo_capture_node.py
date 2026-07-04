#!/usr/bin/env python3
"""拍照節點：訂閱 zoom_in 跟 processed_image 兩路影像，收到 /capture_photo
觸發訊息（內容是要拍哪一路的 topic 名稱）後，把該路目前最新一張影像存到
ros2_adult/snap/。"""
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

SAVE_DIR = os.path.expanduser('~/ros2_adult/snap')

TOPIC_LABELS = {
    '/zoom_in': 'zoom_in',
    '/processed_image': 'processed_image',
}


class PhotoCaptureNode(Node):
    def __init__(self):
        super().__init__('photo_capture_node')
        self.bridge = CvBridge()
        self.latest_frames = {}

        os.makedirs(SAVE_DIR, exist_ok=True)

        for topic in TOPIC_LABELS:
            self.create_subscription(
                Image, topic,
                self._make_image_callback(topic),
                10,
            )
        self.create_subscription(String, '/capture_photo', self.capture_callback, 10)

        self.get_logger().info('PhotoCaptureNode ready, saving to %s' % SAVE_DIR)

    def _make_image_callback(self, topic):
        def callback(msg):
            self.latest_frames[topic] = msg
        return callback

    def capture_callback(self, msg):
        topic = msg.data
        label = TOPIC_LABELS.get(topic)
        if label is None:
            self.get_logger().warn('Unknown capture topic requested: %s' % topic)
            return

        frame_msg = self.latest_frames.get(topic)
        if frame_msg is None:
            self.get_logger().warn('No frame received yet for %s' % topic)
            return

        cv_image = self.bridge.imgmsg_to_cv2(frame_msg, 'bgr8')
        filename = 'photo_%s_%d.jpg' % (label, int(time.time()))
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, cv_image)
        self.get_logger().info('Saved photo to %s' % save_path)


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
