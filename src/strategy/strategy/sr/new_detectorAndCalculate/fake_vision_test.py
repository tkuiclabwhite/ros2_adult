#!/usr/bin/env python3
# coding=utf-8
"""
fake_vision_test.py — 沒有真實相機/硬體時,拿來測試 sr.py 整條 pipeline 的假資料產生器

流程:
    1. 持續發布 Dio(strategy=True)到 /package/dioarray,讓 sr.py 的 is_start 變 True
    2. 持續發布 /processed_image(bgr8),裡面畫一條帶斜率的色塊模擬板子邊緣,
       近邊距離逐幀變近(模擬機器人正在接近板子),讓 sr.py 的量測 -> 濾波 ->
       決策 -> 觸發整條 pipeline 都能被實際跑到
    3. Ctrl+C 結束時,會先發一次 Dio(strategy=False),方便順便驗證
       sr.py 收到「關閉」訊號後有沒有正確停止步態

用法(開兩個 terminal):
    # terminal A
    source install/setup.bash
    ros2 run strategy sr

    # terminal B
    source install/setup.bash
    python3 src/strategy/strategy/sr/fake_vision_test.py

觀察方式:
    - 直接看 terminal A(sr 節點自己的 log,已經印 layer/confidence/distance/skipped)
    - 或另開 terminal C: ros2 topic echo /ChangeContinuousValue_Topic
    - 或: ros2 topic echo /SendBodyAuto_Topic (看單步步態有沒有被觸發)
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from tku_msgs.msg import Dio

from strategy.sr.board_detector import COLOR_BGR, COLOR_LABEL

# 跟 sr.py 目前的 BOARD_COLOR[0] 一致,預設第一層是 Green;
# 如果你改了 sr.py 的 BOARD_COLOR[0],這裡也要跟著改,不然量到的顏色對不上
FIRST_LAYER_COLOR = 'green'
BOARD_BGR = COLOR_BGR[COLOR_LABEL[FIRST_LAYER_COLOR]]

IMG_W, IMG_H = 320, 240
OUTSET = 215          # 跟 sr.py 的 FOOTBOARD_LINE 一致
BOARD_DEPTH_PX = 30   # 板面深度(pixel),固定值方便觀察 board_depth_px
SLOPE = 0.02          # 模擬板子邊緣的斜率


class FakeVision(Node):
    def __init__(self):
        super().__init__('fake_vision_test')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(RosImage, 'processed_image', 10)
        self.dio_pub = self.create_publisher(Dio, '/package/dioarray', 10)
        self.frame = 0

        self.create_timer(0.05, self.tick)       # 20Hz,跟 sr.py LOOP_PERIOD_SEC 一致
        self.create_timer(0.5, self._send_start)  # 反覆送 is_start=True,確保被收到
        self.get_logger().info(
            f'開始送假資料,模擬顏色={FIRST_LAYER_COLOR} BGR={BOARD_BGR}\033[K'
        )

    def _send_start(self) -> None:
        msg = Dio()
        msg.data = 0
        msg.strategy = True
        self.dio_pub.publish(msg)

    def _send_stop(self) -> None:
        msg = Dio()
        msg.data = 0
        msg.strategy = False
        self.dio_pub.publish(msg)
        self.get_logger().info('已送出 is_start=False,驗證 sr.py 是否停止步態\033[K')

    def tick(self) -> None:
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # 近邊距離模擬:從遠(near_y=120,距離約 95px)逐幀逼近到接近門檻(near_y=205,距離約 10px)
        near_y_center = int(min(120 + self.frame * 0.5, 205))

        for x in range(40, 280):
            edge_y = int(near_y_center + SLOPE * (x - 160))
            edge_y = max(BOARD_DEPTH_PX, min(IMG_H - 1, edge_y))
            far_y = edge_y - BOARD_DEPTH_PX
            img[far_y:edge_y + 1, x] = BOARD_BGR

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.image_pub.publish(msg)

        if self.frame % 20 == 0:
            near_dist = OUTSET - near_y_center
            self.get_logger().info(f'frame={self.frame} 模擬近邊距離(pixel)≈{near_dist}\033[K')
        self.frame += 1


def main(args=None):
    rclpy.init(args=args)
    node = FakeVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
