#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
#import gc
from ultralytics import YOLO


class YoloRosNode(Node):
    def __init__(self):
        super().__init__('yolo_ros_node')
        # Path to the TensorRT engine file
        self.engine_path = "/home/iclab/ros2_adult/src/strategy/strategy/mar/s8_newFinal.engine"
        # self.engine_path = "/workspace/towen/src/strategy/strategy/mar/s8_newFinal.engine"
        # Initialize the YOLO model
        self.model = YOLO(self.engine_path)
        # CvBridge for converting ROS images to OpenCV
        self.bridge = CvBridge()
        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,          #'/image_raw',  # change to your topic
            '/camera1/image_raw',
            self.image_callback,
            10
        )
        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image to OpenCV BGR image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Perform inference on the frame
            results = self.model(frame, imgsz=320)
            # Annotate
            annotated = results[0].plot()
            cv2.imshow('YOLO Detection', annotated)
            cv2.waitKey(1)
        finally:
            # Cleanup per-frame to release memory
            del results
            torch.cuda.empty_cache()
            #gc.collect()

    def destroy_node(self):
        # Close YOLO model and windows before shutdown
        self.model.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

