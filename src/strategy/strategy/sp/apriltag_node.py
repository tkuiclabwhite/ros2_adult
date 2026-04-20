import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from pupil_apriltags import Detector


class AprilTagNode(Node):
    def __init__(self):
        super().__init__('apriltag_node')

        self.bridge = CvBridge()
        self.image_topic = '/camera1/image_raw'

        self.detector = Detector(
            families='tag36h11',
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=1.2,
            refine_edges=1,
            decode_sharpening=0.7,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.tag_pub = self.create_publisher(
            Float32MultiArray,
            '/apriltag_info',
            10
        )

        self.get_logger().info(f'apriltag_node started, subscribe: {self.image_topic}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        if frame is None:
            self.get_logger().warn('frame is None')
            return

        h, w = frame.shape[:2]
        self.get_logger().info(f'[IMG] received frame: {w}x{h}')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        results = self.detector.detect(gray)

        out = Float32MultiArray()

        if len(results) == 0:
            print('[PUB] NOT FOUND')
            out.data = [0.0, -1.0, 0.0, 0.0, 0.0]
            self.tag_pub.publish(out)
            return

        tag = results[0]
        center_x, center_y = tag.center
        area = cv2.contourArea(tag.corners.astype('float32'))

        self.get_logger().info(
            f'[TAG] id={tag.tag_id} cx={center_x:.1f} cy={center_y:.1f} area={area:.1f}'
        )

        out.data = [
            1.0,
            float(tag.tag_id),
            float(center_x),
            float(center_y),
            float(area)
        ]

        print(
            f'[PUB] found={out.data[0]} '
            f'id={out.data[1]} '
            f'cx={out.data[2]:.1f} '
            f'cy={out.data[3]:.1f} '
            f'area={out.data[4]:.1f}'
        )

        self.tag_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagNode()
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