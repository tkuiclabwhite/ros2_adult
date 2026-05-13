"""usb_cam 主節點。

ROS2 介面
─────────────────────────────────────────────────────────────────────
發佈 (Publisher)
  - image_raw            sensor_msgs/Image
  - camera_info          sensor_msgs/CameraInfo  （與 image_raw 同步發佈）
  - image_raw/compressed sensor_msgs/CompressedImage  （pixel_format=='mjpeg' 時）

訂閱 (Subscriber)
  - /Camera_Topic        tku_msgs/Camera         → 即時調整曝光 / 白平衡 / 亮度
  - /Camera_Save         tku_msgs/CameraSave     → 寫 CameraSet.ini
  - /location            tku_msgs/Location       → 切換 strategy location 並重新載入

服務 (Service)
  - set_capture          std_srvs/SetBool        → 啟停影像擷取
  - /CameraInfo          tku_msgs/srv/CameraInfo → 回傳目前 CameraSet 內容

ROS 參數
  camera_name, camera_info_url, framerate, frame_id, image_height, image_width,
  io_method, pixel_format, av_device_format, video_device, brightness, contrast,
  saturation, sharpness, gain, auto_white_balance, white_balance, autoexposure,
  exposure, autofocus, focus, save_dir
"""
from __future__ import annotations

import threading
from typing import List

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_srvs.srv import SetBool
from tku_msgs.msg import Camera as CameraMsg
from tku_msgs.msg import CameraSave as CameraSaveMsg
from tku_msgs.msg import Location as LocationMsg
from tku_msgs.srv import CameraInfo as CameraInfoSrv

from .camera_io import (
    CameraParameters,
    UsbCam,
    resolve_device_path,
)
from .camera_info_loader import load_camera_info
from .settings_io import (
    CameraSet,
    load_camera_set,
    load_strategy_location,
    save_camera_set,
)


# 預設 topic 名稱；可由 launch remap 改成例如 camera1/image_raw
BASE_TOPIC_NAME = "image_raw"


class UsbCamNode(Node):
    """usb_cam 主節點。"""

    # ------------------------------------------------------------------
    # 建構：宣告參數、建立 publisher / subscription / service，啟動 timer
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__("usb_cam")

        # 相機抓取物件
        self._camera = UsbCam()
        self._cv_bridge = CvBridge()
        self._params = CameraParameters()

        # CameraSet：亮度/對比/飽和度/白平衡/自動白平衡/自動曝光的執行階段值。
        # 控制項可被 /Camera_Topic 即時改寫，因此另外維護而非直接用 ROS param。
        self._camera_set = CameraSet()

        # location：由 strategy.ini 或 /location topic 提供
        self._location: str = ""

        # 訊息物件重複使用，避免每次重新分配
        self._image_msg = Image()
        self._compressed_img_msg: CompressedImage | None = None
        self._camera_info_msg: CameraInfo | None = None

        # 發佈端：image_raw + camera_info 預設一律建立（兩條 topic 同步發佈）
        qos = QoSProfile(depth=100)
        self._pub_image = self.create_publisher(Image, BASE_TOPIC_NAME, qos)
        self._pub_camera_info = self.create_publisher(
            CameraInfo, "camera_info", qos
        )
        # mjpeg 模式時才建立的 compressed publisher，於 _init_camera_and_publishers 中決定
        self._pub_compressed: rclpy.publisher.Publisher | None = None
        self._pub_compressed_cam_info: rclpy.publisher.Publisher | None = None

        # 控制項變更需要序列化，避免 timer callback 與 /Camera_Topic 競爭
        self._ctrl_lock = threading.Lock()

        # ---- 訂閱 / 服務 ----
        self._sub_camera_param = self.create_subscription(
            CameraMsg, "/Camera_Topic", self._camera_param_callback, 1
        )
        self._sub_camera_save = self.create_subscription(
            CameraSaveMsg, "/Camera_Save", self._camera_save_callback, 1
        )
        self._sub_location = self.create_subscription(
            LocationMsg, "/location", self._location_callback, 1
        )
        self._srv_set_capture = self.create_service(
            SetBool, "set_capture", self._service_capture
        )
        self._srv_camera_info = self.create_service(
            CameraInfoSrv, "/CameraInfo", self._load_camera_set_file_service
        )

        # ---- 參數宣告 ----
        self.declare_parameter("camera_name", "default_cam")
        self.declare_parameter("camera_info_url", "")
        self.declare_parameter("framerate", 30.0)
        self.declare_parameter("frame_id", "default_cam")
        self.declare_parameter("image_height", 240)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("io_method", "mmap")
        self.declare_parameter("pixel_format", "yuyv")
        self.declare_parameter("av_device_format", "YUV422P")
        self.declare_parameter("video_device", "/dev/video2")
        self.declare_parameter("brightness", 140)
        self.declare_parameter("contrast", 200)
        self.declare_parameter("saturation", 100)
        self.declare_parameter("sharpness", -1)
        self.declare_parameter("gain", -1)
        self.declare_parameter("auto_white_balance", False)
        self.declare_parameter("white_balance", -1)
        self.declare_parameter("autoexposure", False)
        self.declare_parameter("exposure", -1)
        self.declare_parameter("autofocus", False)
        self.declare_parameter("focus", -1)
        self.declare_parameter("save_dir", "")

        # 把宣告好的 ROS 參數讀進 self._params / self._camera_set
        self._get_params()
        # 開相機 + 套用初始控制項
        self._init_camera_and_publishers()
        # 動態參數變更 callback
        self.add_on_set_parameters_callback(self._parameters_callback)

    # ------------------------------------------------------------------
    # 參數 → 內部資料結構
    # ------------------------------------------------------------------
    def _get_params(self) -> None:
        """把 ROS 參數讀進 self._params。"""
        p = self._params
        gp = self.get_parameter
        p.camera_name = gp("camera_name").get_parameter_value().string_value
        p.camera_info_url = (
            gp("camera_info_url").get_parameter_value().string_value
        )
        p.framerate = gp("framerate").get_parameter_value().double_value
        p.frame_id = gp("frame_id").get_parameter_value().string_value
        p.image_height = gp("image_height").get_parameter_value().integer_value
        p.image_width = gp("image_width").get_parameter_value().integer_value
        p.io_method_name = gp("io_method").get_parameter_value().string_value
        p.pixel_format_name = (
            gp("pixel_format").get_parameter_value().string_value
        )
        p.av_device_format = (
            gp("av_device_format").get_parameter_value().string_value
        )
        # video_device 同樣要做 symlink 解析
        p.device_name = resolve_device_path(
            gp("video_device").get_parameter_value().string_value
        )
        p.brightness = gp("brightness").get_parameter_value().integer_value
        p.contrast = gp("contrast").get_parameter_value().integer_value
        p.saturation = gp("saturation").get_parameter_value().integer_value
        p.sharpness = gp("sharpness").get_parameter_value().integer_value
        p.gain = gp("gain").get_parameter_value().integer_value
        p.auto_white_balance = (
            gp("auto_white_balance").get_parameter_value().bool_value
        )
        p.white_balance = (
            gp("white_balance").get_parameter_value().integer_value
        )
        p.autoexposure = gp("autoexposure").get_parameter_value().bool_value
        p.exposure = gp("exposure").get_parameter_value().integer_value
        p.autofocus = gp("autofocus").get_parameter_value().bool_value
        p.focus = gp("focus").get_parameter_value().integer_value

        # 把 ROS 參數的初始亮度等同步到 CameraSet（之後被 INI 與 /Camera_Topic 覆蓋）
        self._camera_set.brightness = p.brightness
        self._camera_set.contrast = p.contrast
        self._camera_set.saturation = p.saturation
        self._camera_set.white_balance = (
            p.white_balance if p.white_balance != -1 else 0
        )
        self._camera_set.auto_white_balance = p.auto_white_balance
        self._camera_set.auto_exposure = p.autoexposure

    def _assign_params(self, parameters: List[Parameter]) -> None:
        """把動態變更的參數套到 self._params。

        只接受我們認識的 key，未知的 key 給警告。
        """
        p = self._params
        for param in parameters:
            name = param.name
            if name == "camera_name":
                p.camera_name = param.value
            elif name == "camera_info_url":
                p.camera_info_url = param.value
            elif name == "frame_id":
                p.frame_id = param.value
            elif name == "framerate":
                p.framerate = float(param.value)
                self.get_logger().warn(f"framerate: {p.framerate}")
            elif name == "image_height":
                p.image_height = int(param.value)
            elif name == "image_width":
                p.image_width = int(param.value)
            elif name == "io_method":
                p.io_method_name = param.value
            elif name == "pixel_format":
                p.pixel_format_name = param.value
            elif name == "av_device_format":
                p.av_device_format = param.value
            elif name == "video_device":
                p.device_name = resolve_device_path(param.value)
            elif name == "brightness":
                p.brightness = int(param.value)
            elif name == "contrast":
                p.contrast = int(param.value)
            elif name == "saturation":
                p.saturation = int(param.value)
            elif name == "sharpness":
                p.sharpness = int(param.value)
            elif name == "gain":
                p.gain = int(param.value)
            elif name == "auto_white_balance":
                p.auto_white_balance = bool(param.value)
            elif name == "white_balance":
                p.white_balance = int(param.value)
            elif name == "autoexposure":
                p.autoexposure = bool(param.value)
            elif name == "exposure":
                p.exposure = int(param.value)
            elif name == "autofocus":
                p.autofocus = bool(param.value)
            elif name == "focus":
                p.focus = int(param.value)
            elif name == "save_dir":
                # save_dir 不在 _params 中（屬於設定檔搜尋），由 get_parameter 直接讀
                pass
            else:
                self.get_logger().warn(f"Invalid parameter name: {name}")

    # ------------------------------------------------------------------
    # 啟動：開相機、套參數、建 timer
    # ------------------------------------------------------------------
    def _init_camera_and_publishers(self) -> None:
        """開相機、載入校正、套用初始控制項、建立 timer。"""
        # frame_id 必填；參數宣告時已給預設，這裡只是再保險一次
        if not self._params.frame_id:
            self.get_logger().warn(
                "frame_id is empty; using 'default_cam'"
            )
            self._params.frame_id = "default_cam"

        # 載入 camera intrinsics（若有設 camera_info_url）
        self._camera_info_msg = load_camera_info(
            self._params.camera_info_url,
            self._params.frame_id,
            self._params.image_width,
            self._params.image_height,
        )

        # mjpeg 模式：另外建立 compressed publisher
        if self._params.pixel_format_name == "mjpeg":
            self._compressed_img_msg = CompressedImage()
            self._compressed_img_msg.header.frame_id = self._params.frame_id
            self._pub_compressed = self.create_publisher(
                CompressedImage, BASE_TOPIC_NAME + "/compressed", 100
            )
            self._pub_compressed_cam_info = self.create_publisher(
                CameraInfo, "camera_info", 100
            )

        self._image_msg.header.frame_id = self._params.frame_id

        self.get_logger().info(
            f"Starting '{self._params.camera_name}' "
            f"({self._params.device_name}) at "
            f"{self._params.image_width}x{self._params.image_height} via "
            f"{self._params.io_method_name} ({self._params.pixel_format_name}) "
            f"at {self._params.framerate} FPS"
        )

        # 嘗試開相機；失敗就直接停掉 node
        try:
            self._camera.configure(self._params, self._params.io_method_name)
            # 載入 location 與對應的 CameraSet.ini
            self._location = load_strategy_location(
                lambda m: self.get_logger().info(m)
            )
            self._camera_set = load_camera_set(
                self._location,
                self._get_save_dir(),
                self._camera_set,
                lambda m: self.get_logger().info(m),
            )
            self._set_v4l2_params()
            self._camera.start()
        except Exception as e:
            self.get_logger().error(
                f"[init] camera fatal error: {e}"
            )
            rclpy.shutdown()
            return

        # 建立 timer（period_ms = 1000 / framerate）
        period_s = 1.0 / max(self._params.framerate, 1e-6)
        self._timer = self.create_timer(period_s, self._update)
        self.get_logger().info(
            f"Timer triggering every {period_s * 1000:.1f} ms"
        )

    def _get_save_dir(self) -> str:
        """取得 save_dir 參數；若尚未宣告會自動補宣告。"""
        if not self.has_parameter("save_dir"):
            self.declare_parameter("save_dir", "")
        return self.get_parameter("save_dir").get_parameter_value().string_value

    # ------------------------------------------------------------------
    # 控制項：套到 V4L2
    # ------------------------------------------------------------------
    def _set_v4l2_params(self) -> None:
        """把 self._camera_set 套到實體相機。

        映射式：
          - auto_exposure: False → v4l2 1（手動）, True → 3（光圈優先）
          - brightness >= -64    → v4l2 = (brightness + 70) * 2
          - contrast / saturation → v4l2 = value * 2
          - auto_white_balance True  → white_balance_automatic = 1
            False → 0 並另外設 white_balance_temperature = white_balance
        """
        with self._ctrl_lock:
            cs = self._camera_set
            cam = self._camera

            # 自動曝光：1=手動, 3=光圈優先（V4L2 對 UVC 的常見編碼）
            if not cs.auto_exposure:
                cam.set_v4l_parameter("auto_exposure", 1)
            else:
                cam.set_v4l_parameter("auto_exposure", 3)

            if cs.brightness >= -64:
                cam.set_v4l_parameter(
                    "brightness", (cs.brightness + 70) * 2
                )
            if cs.contrast >= 0:
                cam.set_v4l_parameter("contrast", cs.contrast * 2)
            if cs.saturation >= 0:
                cam.set_v4l_parameter("saturation", cs.saturation * 2)

            if cs.auto_white_balance:
                cam.set_v4l_parameter("white_balance_automatic", 1)
            else:
                cam.set_v4l_parameter("white_balance_automatic", 0)
                cam.set_v4l_parameter(
                    "white_balance_temperature", cs.white_balance
                )

    # ------------------------------------------------------------------
    # 訂閱 callback
    # ------------------------------------------------------------------
    def _camera_param_callback(self, msg: CameraMsg) -> None:
        """``/Camera_Topic`` callback：即時調整控制項。

        注意：tku_msgs/Camera 的 whitebalance / autowhitebalance 欄位無底線，
        與我們內部 white_balance 命名不同，要明確對映。
        """
        cs = self._camera_set
        cs.brightness = int(msg.brightness)
        cs.contrast = int(msg.contrast)
        cs.saturation = int(msg.saturation)
        cs.auto_white_balance = bool(msg.autowhitebalance)

        if int(msg.whitebalance) != -1:
            cs.white_balance = int(msg.whitebalance)
        else:
            cs.white_balance = 0

        cs.auto_exposure = bool(msg.auto_exposure)
        cs.zoomin = float(msg.zoomin)

        # 直接套用以下幾項到實體相機（不重新跑整個 _set_v4l2_params）
        with self._ctrl_lock:
            cam = self._camera
            if not cs.auto_exposure:
                cam.set_v4l_parameter("auto_exposure", 1)
            else:
                cam.set_v4l_parameter("auto_exposure", 3)

            if cs.brightness >= -64:
                cam.set_v4l_parameter(
                    "brightness", (cs.brightness + 70) * 2
                )
            if cs.contrast >= 0:
                cam.set_v4l_parameter("contrast", cs.contrast * 2)
            if cs.saturation >= 0:
                cam.set_v4l_parameter("saturation", cs.saturation * 2)

            if cs.auto_white_balance:
                cam.set_v4l_parameter("white_balance_automatic", 1)
            else:
                cam.set_v4l_parameter("white_balance_automatic", 0)
                cam.set_v4l_parameter(
                    "white_balance_temperature", cs.white_balance
                )

    def _location_callback(self, msg: LocationMsg) -> None:
        """``/location`` callback：切換 strategy location 並重新載入設定。"""
        self._location = msg.data or ""
        self._camera_set = load_camera_set(
            self._location,
            self._get_save_dir(),
            self._camera_set,
            lambda m: self.get_logger().info(m),
        )
        self._set_v4l2_params()

    def _camera_save_callback(self, msg: CameraSaveMsg) -> None:
        """``/Camera_Save`` callback：寫 CameraSet.ini。

        寫入順序：primary → save_dir/CameraSet.ini →
        ~/.ros/usb_cam/config/CameraSet.ini。
        """
        cs = self._camera_set
        cs.brightness = int(msg.brightness)
        cs.contrast = int(msg.contrast)
        cs.saturation = int(msg.saturation)
        cs.white_balance = int(msg.whitebalance)
        cs.auto_white_balance = bool(msg.autowhitebalance)
        cs.auto_exposure = bool(msg.auto_exposure)
        cs.zoomin = float(msg.zoomin)

        self.get_logger().info(
            f"[camera_save_callback] b={cs.brightness} c={cs.contrast} "
            f"s={cs.saturation} wb={cs.white_balance} "
            f"awb={int(cs.auto_white_balance)} ae={int(cs.auto_exposure)} "
            f"zi={cs.zoomin:.1f}"
        )

        if not self._location:
            self.get_logger().error(
                "[camera_save_callback] location is empty; "
                "cannot resolve target path."
            )
            return

        save_camera_set(
            cs,
            self._location,
            self._get_save_dir(),
            lambda m: self.get_logger().info(m),
        )

    # ------------------------------------------------------------------
    # 服務 callback
    # ------------------------------------------------------------------
    def _service_capture(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """``set_capture`` 服務：True 啟動串流、False 暫停串流。"""
        if request.data:
            self._camera.start_capturing()
            response.message = "Start Capturing"
        else:
            self._camera.stop_capturing()
            response.message = "Stop Capturing"
        response.success = True
        return response

    def _load_camera_set_file_service(
        self,
        request: CameraInfoSrv.Request,
        response: CameraInfoSrv.Response,
    ) -> CameraInfoSrv.Response:
        """``/CameraInfo`` 服務：強制重新讀 CameraSet.ini 並回覆現值。

        tku_msgs/srv/CameraInfo 的 request 欄位 ``load`` 為 bool flag，
        本實作未使用其內容。
        """
        del request  # request.load 目前未使用
        self._camera_set = load_camera_set(
            self._location,
            self._get_save_dir(),
            self._camera_set,
            lambda m: self.get_logger().info(m),
        )
        cs = self._camera_set
        response.brightness = cs.brightness
        response.contrast = cs.contrast
        response.saturation = cs.saturation
        response.white_balance = cs.white_balance
        response.auto_white_balance = cs.auto_white_balance
        response.auto_exposure = cs.auto_exposure
        response.zoomin = cs.zoomin
        # auto_backlight_compensation 目前未追蹤，回覆 False
        response.auto_backlight_compensation = False
        return response

    # ------------------------------------------------------------------
    # 動態參數 callback
    # ------------------------------------------------------------------
    def _parameters_callback(
        self, parameters: List[Parameter]
    ) -> SetParametersResult:
        """ROS 動態參數變更 → 套用到實體相機。會 reset timer。"""
        self.get_logger().debug(
            f"Setting parameters for {self._params.camera_name}"
        )
        if hasattr(self, "_timer"):
            self._timer.reset()
        self._assign_params(parameters)
        # 把 ROS 參數同步到 CameraSet 後再套到 V4L2
        # 注意：CameraSet 的 brightness 等與 ROS 參數同名但語意一致
        self._sync_params_to_camera_set()
        self._set_v4l2_params()
        return SetParametersResult(successful=True, reason="success")

    def _sync_params_to_camera_set(self) -> None:
        """ROS 參數 → CameraSet（只覆蓋 V4L2 控制項相關欄位）。"""
        p = self._params
        cs = self._camera_set
        cs.brightness = p.brightness
        cs.contrast = p.contrast
        cs.saturation = p.saturation
        if p.white_balance != -1:
            cs.white_balance = p.white_balance
        cs.auto_white_balance = p.auto_white_balance
        cs.auto_exposure = p.autoexposure

    # ------------------------------------------------------------------
    # 主迴圈：抓 frame → 發佈
    # ------------------------------------------------------------------
    def _update(self) -> None:
        """timer callback：抓 frame → 發佈。"""
        if not self._camera.is_capturing():
            return
        try:
            if self._params.pixel_format_name == "mjpeg":
                self._take_and_send_image_mjpeg()
            else:
                self._take_and_send_image()
        except Exception as e:
            self.get_logger().error(f"[update] camera exception: {e}")
            self._camera.stop_capturing()

    def _take_and_send_image(self) -> None:
        """擷取並發佈 sensor_msgs/Image + sensor_msgs/CameraInfo。

        OpenCV 一律輸出 BGR，因此 encoding 填 ``bgr8``；下游若需要 RGB 可在
        消費端用 cv_bridge 取得 ``rgb8``。
        """
        cap = self._camera.get_image()
        if cap is None:
            return
        # 用 cv_bridge 把 numpy 包成 sensor_msgs/Image（會處理 step / data）
        msg = self._cv_bridge.cv2_to_imgmsg(cap.image, encoding="bgr8")
        msg.header.frame_id = self._params.frame_id
        msg.header.stamp.sec = cap.stamp_sec
        msg.header.stamp.nanosec = cap.stamp_nsec

        # CameraInfo header 與 Image 同步
        if self._camera_info_msg is None:
            self._camera_info_msg = CameraInfo()
        self._camera_info_msg.header = msg.header
        self._camera_info_msg.width = msg.width
        self._camera_info_msg.height = msg.height

        self._pub_image.publish(msg)
        self._pub_camera_info.publish(self._camera_info_msg)

    def _take_and_send_image_mjpeg(self) -> None:
        """擷取並發佈 sensor_msgs/CompressedImage + sensor_msgs/CameraInfo。

        OpenCV 不會回傳 driver 給的 raw MJPEG bytes，而是 BGR；因此用
        ``cv2.imencode`` 把 BGR 重新編成 JPEG 再放入 CompressedImage。
        對下游視覺等效，僅多一次 JPEG 編碼成本。
        """
        cap = self._camera.get_image()
        if cap is None:
            return
        ok, jpg = cv2.imencode(".jpg", cap.image)
        if not ok:
            self.get_logger().warn("cv2.imencode JPEG failed")
            return

        if self._compressed_img_msg is None:
            self._compressed_img_msg = CompressedImage()
            self._compressed_img_msg.header.frame_id = self._params.frame_id
        self._compressed_img_msg.format = "jpeg"
        self._compressed_img_msg.header.stamp.sec = cap.stamp_sec
        self._compressed_img_msg.header.stamp.nanosec = cap.stamp_nsec
        # tobytes() 回傳 bytes；CompressedImage.data 是 array<uint8>
        self._compressed_img_msg.data = jpg.tobytes()

        if self._camera_info_msg is None:
            self._camera_info_msg = CameraInfo()
        self._camera_info_msg.header = self._compressed_img_msg.header

        if self._pub_compressed is not None:
            self._pub_compressed.publish(self._compressed_img_msg)
        if self._pub_compressed_cam_info is not None:
            self._pub_compressed_cam_info.publish(self._camera_info_msg)

    # ------------------------------------------------------------------
    # 解構
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        """關 timer 與相機。"""
        self.get_logger().warn("Shutting down")
        if hasattr(self, "_timer") and self._timer is not None:
            self._timer.cancel()
        try:
            self._camera.shutdown()
        except Exception as e:
            self.get_logger().warn(f"camera shutdown error: {e}")
        return super().destroy_node()


# ---------------------------------------------------------------------------
# 進入點
# ---------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = UsbCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
