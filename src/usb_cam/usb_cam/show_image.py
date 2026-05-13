"""簡易影像檢視工具：訂閱 image_raw 並用 cv2.imshow 顯示。

以 entry_point 安裝為 ``ros2 run usb_cam show_image``。
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ExamineImage(Node):
    """訂閱 image_raw，依 encoding 解析後顯示。"""

    def __init__(self):
        super().__init__('examine_image')
        self.mat = None
        self.sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, 100
        )

    def image_callback(self, msg):
        sz = (msg.height, msg.width)
        if msg.step * msg.height != len(msg.data):
            print('bad step/height/data size')
            return

        if msg.encoding == 'rgb8':
            # rgb8: 三通道交錯；R=0, G=1, B=2 後再 reshape
            if (self.mat is None or msg.width != self.mat.shape[1]
                    or msg.height != self.mat.shape[0]
                    or len(self.mat.shape) < 2 or self.mat.shape[2] != 3):
                self.mat = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            # imshow 預期 BGR，所以反向放
            self.mat[:, :, 2] = np.array(msg.data[0::3]).reshape(sz)
            self.mat[:, :, 1] = np.array(msg.data[1::3]).reshape(sz)
            self.mat[:, :, 0] = np.array(msg.data[2::3]).reshape(sz)
        elif msg.encoding == 'bgr8':
            # bgr8: 三通道交錯，順序與 cv2 imshow 期望一致
            self.mat = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
        elif msg.encoding == 'mono8':
            self.mat = np.array(msg.data).reshape(sz)
        else:
            print(f'unsupported encoding {msg.encoding}')
            return

        if self.mat is not None:
            cv2.imshow('image', self.mat)
            cv2.waitKey(5)


def main(args=None):
    rclpy.init(args=args)
    examine_image = ExamineImage()
    try:
        rclpy.spin(examine_image)
    except KeyboardInterrupt:
        pass
    examine_image.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
