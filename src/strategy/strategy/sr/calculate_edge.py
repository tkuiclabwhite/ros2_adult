#!/usr/bin/env python
#coding=utf-8
import rclpy
import numpy as np
from strategy.API import API                  #PythonAPI
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tku_msgs.msg import Camera
import cv2 
import sys
import time
import math
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

MULT = 1
LENTH = 320/MULT
WIDTH = 240/MULT

#             Orange   Yellow      Blue        Green     Black   Red         White
COLOR_MASK = [[0,0,0],[128,128,0],[128,0,128],[0,0,128],[0,0,0],[255,255,0],[0,0,0]]
class deep_calculate(Node):
    def __init__(self,color,layer):
        super().__init__('deep_calculate')
        self.bridge = CvBridge()
        # 訂閱攝像頭資訊 #"/kidsize/camera/image_raw" #"compress_image" #"/usb_cam/image_raw"
        #colormodel_image  orign_image
        self.Image_compress_sub = self.create_subscription(
            Image,
            'processed_image',
            self.convert,
            10
        )		
        self.init()
        self.color = color
        self.layer = layer
        self.output =[]
        self.new_label_matrix_flatten = []


    def init(self):
        self.now_line_coordinate    = [-9999,-9999,-9999,-9999]
        self.last_line_coordinate   = [-9999,-9999,-9999,-9999]

    def generate_custom_label_matrix(self, img):
        # 確保影像是 320x240 大小
        img_320 = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # 建立一張全黑的標籤矩陣 (全部填 0)
        label_matrix = np.zeros((240, 320), dtype=np.int32)
        
        # 根據你的色模表建立映射字典 (Tuple 格式)
        # 注意: cv2 讀進來的通常是 BGR，但為了避免排列問題，我們直接比對你的數值
        pixel_to_param = {
            (128,   0,   0): 1,   # Orange (深紅)
            (128, 128,   0): 2,   # Yellow (黃綠)
            (128,   0, 128): 4,   # Blue   (紫)
            (  0,   0, 128): 8,   # Green  (深藍)
            (255,   0, 255): 16,  # Black  (粉)
            (255, 255,   0): 32,  # Red    (黃)
            (  0, 255, 255): 64   # White  (青綠)
        }
        
        # 快速掃描整張圖片，把符合顏色的地方填上 parameter
        for pixel_val, param in pixel_to_param.items():
            # 尋找影像中像素值等於 pixel_val 的位置
            mask = np.all(img_320 == pixel_val, axis=-1)
            label_matrix[mask] = param
            
        # 攤平為一維陣列回傳給主程式
        return label_matrix.flatten().tolist()

    # 影像判斷更新
    def convert(self, imgmsg):
        try:                             #影像通訊
            img = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return # 加個 return 避免報錯後還繼續跑
        
        if getattr(self, 'api', None) is not None:
            self.new_label_matrix_flatten = self.generate_custom_label_matrix(img)

        img  = img[115:215, 90:225]
        
        # img = cv2.resize(img, (int(LENTH),int(WIDTH)))
        self.edge(img,self.color)

        ##----測試用---##
        # cv2.imshow("Image_show",self.output)
        # cv2.waitKey(1)
        ##------------##
        
        self.api.drawImageFunction(999,1,self.now_line_coordinate[0],self.now_line_coordinate[2],self.now_line_coordinate[1],self.now_line_coordinate[3],255,255,0,1)

        if hasattr(self, 'debug_scan_points'):
            for idx, (px, py) in enumerate(self.debug_scan_points[:40]):
                draw_x = px + 90
                draw_y = py + 115
                self.api.drawImageFunction(950+idx, 2, draw_x-2, draw_x+2, draw_y-2, draw_y+2, 0, 255, 0, 1)

        #計算斜率
        if abs(self.now_line_coordinate[1] - self.now_line_coordinate[3]) == 0:
            self.slope = 0
        else:
            self.slope = (self.now_line_coordinate[1] - self.now_line_coordinate[3]) / abs(self.now_line_coordinate[0] - self.now_line_coordinate[2])*10

    def calc_slope(self, x0, y0, x1, y1):
        if x0 == x1:
            #無限e
            return float("inf")
        return (y1 - y0) / (x1 - x0)

    def edge(self, img, color):
        if getattr(self, 'api', None) is None: return
        for i in range(0, self.api.color_counts[color]):
            if self.api.object_sizes[color][i] > 5000:
                break
            else:
                self.now_line_coordinate  = [-9999, -9999, -9999, -9999]
                self.last_line_coordinate = [-9999,-9999,-9999,-9999]
                return

        h, w = img.shape[0], img.shape[1]

        #---- 遮罩:向量化寫法,取代雙層 for 迴圈逐像素 .item() ----
        target_color = np.array(COLOR_MASK[color], dtype=img.dtype)
        match_mask = np.all(img == target_color, axis=-1)   # shape (h, w),一次算完整張
        mask = match_mask
        # img[~match_mask] = (0, 0, 0)

        # output = cv2.medianBlur(img, 15)

        #---- 多欄掃描 ----
        num_columns = self.SLOPE_SCAN_COLUMNS if hasattr(self, 'SLOPE_SCAN_COLUMNS') else 20
        min_run     = self.SLOPE_MIN_RUN      if hasattr(self, 'SLOPE_MIN_RUN')      else 5

        xs = np.linspace(0, w-1, num_columns, dtype=int)   
        points = []
        y_seq = range(h-1, -1, -1) if self.layer < 4 else range(0, h, 1) 

        for x in xs:
            col = mask[:, x]
            run = 0
            found_y = None
            for y in y_seq:
                if col[y]:
                    run += 1
                    if run >= min_run:
                        found_y = y
                        break
                else:
                    run = 0
            if found_y is not None:
                points.append((x, found_y))

        if len(points) >= 4:
            pts = np.array(points, dtype=float)
            xs_pt, ys_pt = pts[:, 0], pts[:, 1]
            A = np.vstack([xs_pt, np.ones_like(xs_pt)]).T
            slope_px, intercept = np.linalg.lstsq(A, ys_pt, rcond=None)[0]

            residual = np.abs(ys_pt - (slope_px * xs_pt + intercept))
            std = residual.std() if residual.std() > 1e-6 else 1.0
            inliers = residual < max(2.0 * std, 3.0)
            if np.sum(inliers) >= 4:
                xs_in, ys_in = xs_pt[inliers], ys_pt[inliers]
                A_in = np.vstack([xs_in, np.ones_like(xs_in)]).T
                slope_px, intercept = np.linalg.lstsq(A_in, ys_in, rcond=None)[0]
                xs_pt, ys_pt = xs_in, ys_in

            x1, x2 = float(xs_pt.min()), float(xs_pt.max())
            y1, y2 = slope_px * x1 + intercept, slope_px * x2 + intercept

            # 裁切為 img[115:215, 90:225]:x 偏移改成 +90(原本是+120)
            self.now_line_coordinate[0] = int(round(x1 + 90))
            self.now_line_coordinate[2] = int(round(x2 + 90))
            self.now_line_coordinate[1] = int(round(y1 + 115))
            self.now_line_coordinate[3] = int(round(y2 + 115))

            self.debug_scan_points = [(int(px), int(py)) for px, py in points]
        else:
            self.now_line_coordinate = self.last_line_coordinate
            self.debug_scan_points = []

        self.last_line_coordinate = self.now_line_coordinate
        # self.output = output

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