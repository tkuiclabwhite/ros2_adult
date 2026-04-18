#!/usr/bin/env python3
# coding=utf-8

import rclpy
from rclpy.node import Node

import numpy as np
# from hello1 import Sendmessage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tku_msgs.msg import Camera
import cv2
import sys
import time
import math
from strategy.API import API

class deep_calculate(Node):
    def __init__(self):
        super().__init__('deep_calculate')

        self.bridge = CvBridge()

        self.Image_compress_sub = self.create_subscription(
            Image,
            'processed_image',
            self.convert,
            10
        )

        self.first_red = True
        self.ya = [0] * 32
        self.aa = [0] * 32
        self.ba = [0] * 32
        self.x1 = 0
        self.y1 = 0
        self.x2 = 1
        self.y2 = 0
        self.cnt = 0
        self.a = True
        self.b = True
        self.slope = 0
        self.degree = 0
        self.red_width = 0
        self.Y_Dy = 24

    def convert(self, imgmsg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        cv_image = cv2.resize(cv_image, (320, 240))
        cv_image_2 = cv2.resize(cv_image, (32, 24), interpolation=cv2.INTER_NEAREST)

        self.red_width = 0
        self.R_Deep_Matrix = []

        for compress_width in range(0, 32):
            self.r = True
            self.R_Deep_Matrix.append(0)
            for compress_height in range(23, -1, -1):
                blue = cv_image_2.item(compress_height, compress_width, 0)
                green = cv_image_2.item(compress_height, compress_width, 1)
                red = cv_image_2.item(compress_height, compress_width, 2)

                if (blue == 255 and green == 255 and red == 0) and self.r:
                    self.red_width += 1
                    self.r = False

                if (blue == 255 and green == 255 and red == 0):
                    self.R_Deep_Matrix[compress_width] = 23 - compress_height
                    break

                if compress_height == 0:
                    self.R_Deep_Matrix[compress_width] = 24

        self.x1 = 0
        self.y1 = 0
        self.x2 = 1
        self.y2 = 0
        self.cnt = 0
        self.a0 = self.a1 = self.a2 = self.a3 = self.a4 = 0
        self.Xa = self.Ya = 0
        self.Xmin = self.Ymin = 0
        flag = True
        self.redsize = False

        for compress_width in range(0, 32):
            self.a = True
            for compress_height in range(23, -1, -1):
                blue = cv_image_2.item(compress_height, compress_width, 0)
                green = cv_image_2.item(compress_height, compress_width, 1)
                red = cv_image_2.item(compress_height, compress_width, 2)

                if compress_height == 6 and (blue, green, red) == (128, 128, 0):
                    self.a0 += 1
                elif compress_height == 7 and (blue, green, red) == (128, 128, 0):
                    self.a1 += 1
                elif compress_height == 8 and (blue, green, red) == (128, 128, 0):
                    self.a2 += 1
                elif compress_height == 9 and (blue, green, red) == (128, 128, 0):
                    self.a3 += 1
                elif compress_height == 10 and (blue, green, red) == (128, 128, 0):
                    self.a4 += 1

                if (blue, green, red) == (255, 255, 0):
                    self.redsize = True
                    if self.a:
                        self.Xa = compress_width
                        self.Ya = 23 - compress_height
                        self.cnt += 1
                        self.a = False

                if (blue, green, red) == (255, 255, 0) and self.first_red:
                    self.first_red = False
                    self.x1 = compress_width
                    self.y1 = 23 - compress_height

                if abs(self.red_width - self.cnt) <= 2 and self.b:
                    self.b = False
                    self.x2 = compress_width
                    self.y2 = 23 - compress_height

                if self.Ya == min(self.R_Deep_Matrix):
                    if self.y1 > self.y2 and flag:
                        self.Xmin = self.Xa
                        self.Ymin = self.Ya
                        flag = False
                    elif self.y2 >= self.y1:
                        self.Xmin = self.Xa
                        self.Ymin = self.Ya

        if abs(self.x1 - self.x2) < 1:
            self.slope = 0
            self.degree = 0
        else:
            if abs(self.Xmin - self.x1) <= abs(self.Xmin - self.x2):
                self.slope = (self.y2 - self.Ymin) / (self.x2 - self.Xmin)
            else:
                self.slope = (self.Ymin - self.y1) / (self.Xmin - self.x1)

            self.degree = int(math.degrees(self.slope))

        self.first_red = True
        self.b = True

        self.Deep_Matrix = []
        for w in range(32):
            self.Deep_Matrix.append(24)
            for h in range(23, -1, -1):
                b, g, r = cv_image_2.item(h, w, 0), cv_image_2.item(h, w, 1), cv_image_2.item(h, w, 2)
                if (b, g, r) in [(128, 0, 128), (128, 128, 0)]:
                    self.Deep_Matrix[w] = 23 - h
                    break

        self.aa = self.Deep_Matrix
        self.filter_sum_aa = sum(x for x in self.aa if x != 24)


        self.Y_Deep_Matrix = []
        for w in range(32):
            self.Y_Deep_Matrix.append(24)
            for h in range(23, -1, -1):
                if cv_image_2.item(h, w, 0) == 128 and cv_image_2.item(h, w, 1) == 128:
                    self.Y_Deep_Matrix[w] = 23 - h
                    print(self.Y_Deep_Matrix)
                    break

        self.ya = self.Y_Deep_Matrix

        self.B_Deep_Matrix = []
        for w in range(32):
            self.B_Deep_Matrix.append(24)
            for h in range(23, -1, -1):
                if cv_image_2.item(h, w, 0) == 128 and cv_image_2.item(h, w, 2) == 128:
                    self.B_Deep_Matrix[w] = 23 - h
                    break

        self.ba = self.B_Deep_Matrix

def main(args=None):
    rclpy.init(args=args)

    try:
        while rclpy.ok():
            send = API()
            if send.Web:
                pass
            else:
                node = deep_calculate()
                rclpy.spin(node)
                node.destroy_node()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()