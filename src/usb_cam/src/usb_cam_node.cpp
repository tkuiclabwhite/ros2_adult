// Copyright 2014 Robert Bosch, LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Robert Bosch, LLC nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "usb_cam/usb_cam_node.hpp"
#include "usb_cam/utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tku_msgs/msg/camera.hpp"
#include "tku_msgs/msg/camera_save.hpp"
#include "tku_msgs/srv/camera_info.hpp"
#include "tku_msgs/msg/location.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

const char BASE_TOPIC_NAME[] = "image_raw";

namespace usb_cam
{

UsbCamNode::UsbCamNode(const rclcpp::NodeOptions & node_options)
: Node("usb_cam", node_options),
  m_camera(new usb_cam::UsbCam()),
  m_image_msg(new sensor_msgs::msg::Image()),
  m_compressed_img_msg(nullptr),
  m_image_publisher(std::make_shared<image_transport::CameraPublisher>(
      image_transport::create_camera_publisher(this, BASE_TOPIC_NAME,
      rclcpp::QoS {100}.get_rmw_qos_profile()))),
  m_compressed_image_publisher(nullptr),
  m_compressed_cam_info_publisher(nullptr),
  m_parameters(),
  m_camera_info_msg(new sensor_msgs::msg::CameraInfo()),
  m_service_capture(
    this->create_service<std_srvs::srv::SetBool>(
      "set_capture",
      std::bind(
        &UsbCamNode::service_capture,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3)))
{
  sub_camera_param_ = create_subscription<tku_msgs::msg::Camera>(
  "/Camera_Topic", 1, std::bind(&UsbCamNode::camera_param_callback, this, std::placeholders::_1));

  sub_camera_save_ = create_subscription<tku_msgs::msg::CameraSave>(
  "/Camera_Save", 1, std::bind(&UsbCamNode::camera_save_callback, this, std::placeholders::_1));

  srv_camera_info_ = create_service<tku_msgs::srv::CameraInfo>(
  "/CameraInfo", std::bind(&UsbCamNode::LoadCameraSetFile, this, std::placeholders::_1, std::placeholders::_2));

  sub_location_ = create_subscription<tku_msgs::msg::Location>(
  "/location", 1, std::bind(&UsbCamNode::location_callback, this, std::placeholders::_1));
  // declare params
  this->declare_parameter("camera_name", "default_cam");
  this->declare_parameter("camera_info_url", "");
  this->declare_parameter("framerate", 30.0);
  this->declare_parameter("frame_id", "default_cam");
  this->declare_parameter("image_height", 240);
  this->declare_parameter("image_width", 320);
  this->declare_parameter("io_method", "mmap");
  this->declare_parameter("pixel_format", "yuyv");
  this->declare_parameter("av_device_format", "YUV422P");
  this->declare_parameter("video_device", "/dev/video2");
  // this->declare_parameter("video_device", "usb-Azurewave_USB2.0_HD_UVC_WebCam_0x0001-video-index0");
  // this->declare_parameter("video_device", "usb-046d_罗技高清网络摄像机_C930c_09C55EAE-video-index0");  
  this->declare_parameter("brightness", 140);  // 0-255, -1 "leave alone"
  this->declare_parameter("contrast", 200);    // 0-255, -1 "leave alone"
  this->declare_parameter("saturation", 100);  // 0-255, -1 "leave alone"
  this->declare_parameter("sharpness", -1);   // 0-255, -1 "leave alone"
  this->declare_parameter("gain", -1);        // 0-100?, -1 "leave alone"
  this->declare_parameter("auto_white_balance", false);
  this->declare_parameter("white_balance", -1);
  this->declare_parameter("autoexposure", false);
  this->declare_parameter("exposure", -1);
  this->declare_parameter("autofocus", false);
  this->declare_parameter("focus", -1);  // 0-255, -1 "leave alone"

  this->declare_parameter<std::string>("save_dir", "");

  get_params();
  init();
  m_parameters_callback_handle = add_on_set_parameters_callback(
    std::bind(
      &UsbCamNode::parameters_callback, this,
      std::placeholders::_1));
}

UsbCamNode::~UsbCamNode()
{
  RCLCPP_WARN(this->get_logger(), "Shutting down");
  m_image_msg.reset();
  m_compressed_img_msg.reset();
  m_camera_info_msg.reset();
  m_camera_info.reset();
  m_timer.reset();
  m_service_capture.reset();
  m_parameters_callback_handle.reset();

  delete (m_camera);
}

void UsbCamNode::service_capture(
  const std::shared_ptr<rmw_request_id_t> request_header,
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  (void) request_header;
  if (request->data) {
    m_camera->start_capturing();
    response->message = "Start Capturing";
  } else {
    m_camera->stop_capturing();
    response->message = "Stop Capturing";
  }
}

std::string resolve_device_path(const std::string & path)
{
  if (std::filesystem::is_symlink(path)) {
    std::filesystem::path target_path = std::filesystem::read_symlink(path);

    // if the target path is relative, resolve it
    if (target_path.is_relative()) {
      target_path = std::filesystem::absolute(path).parent_path() / target_path;
      target_path = std::filesystem::canonical(target_path);
    }

    return target_path.string();
  }
  return path;
}

void UsbCamNode::load_strategy_location()
{
    namespace fs = std::filesystem;
    fs::path src_root = fs::path(__FILE__).parent_path()    // .../usb_cam/src
                                       .parent_path()       // .../usb_cam
                                       .parent_path();      // .../tku/src

    // 真正要的 ini 路徑：.../tku/src/strategy/strategy/strategy.ini
    fs::path ini_path = src_root / "strategy" / "strategy" / "strategy.ini";

    std::ifstream in(ini_path);
    if (!in.is_open()) {
        // 用 ROS2 的 logger 也可以
        std::cout << "[usb_cam] 無法開啟檔案: " << ini_path << std::endl;
        location_.clear();
        return;
    }

    std::string line;
    std::getline(in, line);  // 檔案就一行，像 /bb/Parameter

    // 去掉前後空白與換行
    auto first = line.find_first_not_of(" \t\r\n");
    auto last  = line.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) {
        location_.clear();
    } else {
        location_ = line.substr(first, last - first + 1);
    }

    std::cout << "[usb_cam] location_: " << location_ << std::endl;
}

void UsbCamNode::init()
{
  while (m_parameters.frame_id == "") {
    RCLCPP_WARN_ONCE(
      this->get_logger(), "Required Parameters not set...waiting until they are set");
    get_params();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // load the camera info
  m_camera_info.reset(
    new camera_info_manager::CameraInfoManager(
      this, m_parameters.camera_name, m_parameters.camera_info_url));

  if (!m_camera_info->isCalibrated()) {
    m_camera_info->setCameraName(m_parameters.device_name);
    m_camera_info_msg->header.frame_id = m_parameters.frame_id;
    m_camera_info_msg->width = m_parameters.image_width;
    m_camera_info_msg->height = m_parameters.image_height;
    m_camera_info->setCameraInfo(*m_camera_info_msg);
  }

  // Check if given device name is an available v4l2 device
  auto available_devices = usb_cam::utils::available_devices();
  if (available_devices.find(m_parameters.device_name) == available_devices.end()) {
    RCLCPP_ERROR_STREAM(
      this->get_logger(),
      "Device specified is not available or is not a vaild V4L2 device: `" <<
        m_parameters.device_name << "`"
    );
    RCLCPP_INFO(this->get_logger(), "Available V4L2 devices are:");
    for (const auto & device : available_devices) {
      RCLCPP_INFO_STREAM(this->get_logger(), "    " << device.first);
      RCLCPP_INFO_STREAM(this->get_logger(), "        " << device.second.card);
    }
    rclcpp::shutdown();
    return;
  }

  // if pixel format is equal to 'mjpeg', init compressed publisher
  if (m_parameters.pixel_format_name == "mjpeg") {
    m_compressed_img_msg.reset(new sensor_msgs::msg::CompressedImage());
    m_compressed_img_msg->header.frame_id = m_parameters.frame_id;
    m_compressed_image_publisher =
      this->create_publisher<sensor_msgs::msg::CompressedImage>(
        std::string(BASE_TOPIC_NAME) + "/compressed", rclcpp::QoS(100));
    m_compressed_cam_info_publisher =
      this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::QoS(100));
  }

  m_image_msg->header.frame_id = m_parameters.frame_id;
  RCLCPP_INFO(
    this->get_logger(), "Starting '%s' (%s) at %dx%d via %s (%s) at %i FPS",
    m_parameters.camera_name.c_str(), m_parameters.device_name.c_str(),
    m_parameters.image_width, m_parameters.image_height, m_parameters.io_method_name.c_str(),
    m_parameters.pixel_format_name.c_str(), m_parameters.framerate);

  // set the IO method
  io_method_t io_method =
    usb_cam::utils::io_method_from_string(m_parameters.io_method_name);
  if (io_method == usb_cam::utils::IO_METHOD_UNKNOWN) {
    RCLCPP_ERROR_ONCE(
      this->get_logger(),
      "Unknown IO method '%s'", m_parameters.io_method_name.c_str());
    rclcpp::shutdown();
    return;
  }

  // ===== 這一段就是你要加的 try/catch =====
  try {
    m_camera->configure(m_parameters, io_method);
    load_strategy_location();
    LoadSet();
    set_v4l2_params();
    m_camera->start();
  }
  catch (const char* msg) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::init] camera fatal error (char*): %s", msg);
    rclcpp::shutdown();
    return;
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::init] camera fatal std::exception: %s", e.what());
    rclcpp::shutdown();
    return;
  }
  catch (...) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::init] camera fatal unknown exception");
    rclcpp::shutdown();
    return;
  }
  // ===== 這裡以下維持原本的 timer 設定 =====

  const int period_ms = 1000.0 / m_parameters.framerate;
  m_timer = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int64_t>(period_ms)),
    std::bind(&UsbCamNode::update, this));
  RCLCPP_INFO_STREAM(this->get_logger(), "Timer triggering every " << period_ms << " ms");
}


void UsbCamNode::get_params()
{
  auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(this);
  auto parameters = parameters_client->get_parameters(
    {
      "camera_name", "camera_info_url", "frame_id", "framerate", "image_height", "image_width",
      "io_method", "pixel_format", "av_device_format", "video_device", "brightness", "contrast",
      "saturation", "sharpness", "gain", "auto_white_balance", "white_balance", "autoexposure",
      "exposure", "autofocus", "focus"
    }
  );

  assign_params(parameters);
}

void UsbCamNode::assign_params(const std::vector<rclcpp::Parameter> & parameters)
{
  for (auto & parameter : parameters) {
    if (parameter.get_name() == "camera_name") {
      RCLCPP_INFO(this->get_logger(), "camera_name value: %s", parameter.value_to_string().c_str());
      m_parameters.camera_name = parameter.value_to_string();
    } else if (parameter.get_name() == "camera_info_url") {
      m_parameters.camera_info_url = parameter.value_to_string();
    } else if (parameter.get_name() == "frame_id") {
      m_parameters.frame_id = parameter.value_to_string();
    } else if (parameter.get_name() == "framerate") {
      RCLCPP_WARN(this->get_logger(), "framerate: %f", parameter.as_double());
      m_parameters.framerate = parameter.as_double();
    } else if (parameter.get_name() == "image_height") {
      m_parameters.image_height = parameter.as_int();
    } else if (parameter.get_name() == "image_width") {
      m_parameters.image_width = parameter.as_int();
    } else if (parameter.get_name() == "io_method") {
      m_parameters.io_method_name = parameter.value_to_string();
    } else if (parameter.get_name() == "pixel_format") {
      m_parameters.pixel_format_name = parameter.value_to_string();
    } else if (parameter.get_name() == "av_device_format") {
      m_parameters.av_device_format = parameter.value_to_string();
    } else if (parameter.get_name() == "video_device") {
      m_parameters.device_name = resolve_device_path(parameter.value_to_string());
    } else if (parameter.get_name() == "brightness") {
      m_parameters.brightness = parameter.as_int();
    } else if (parameter.get_name() == "contrast") {
      m_parameters.contrast = parameter.as_int();
    } else if (parameter.get_name() == "saturation") {
      m_parameters.saturation = parameter.as_int();
    } else if (parameter.get_name() == "sharpness") {
      m_parameters.sharpness = parameter.as_int();
    } else if (parameter.get_name() == "gain") {
      m_parameters.gain = parameter.as_int();
    } else if (parameter.get_name() == "auto_white_balance") {
      m_parameters.auto_white_balance = parameter.as_bool();
    } else if (parameter.get_name() == "white_balance") {
      m_parameters.white_balance = parameter.as_int();
    } else if (parameter.get_name() == "autoexposure") {
      m_parameters.autoexposure = parameter.as_bool();
    } else if (parameter.get_name() == "exposure") {
      m_parameters.exposure = parameter.as_int();
    } else if (parameter.get_name() == "autofocus") {
      m_parameters.autofocus = parameter.as_bool();
    } else if (parameter.get_name() == "focus") {
      m_parameters.focus = parameter.as_int();
    } else {
      RCLCPP_WARN(this->get_logger(), "Invalid parameter name: %s", parameter.get_name().c_str());
    }
  }
}

void UsbCamNode::camera_param_callback(const tku_msgs::msg::Camera &msg) {
  // std::cout << "Accessing file_path_ from camera_param_callback: " << file_path_ << std::endl;
  // RCLCPP_ERROR_ONCE_STREAM(this->get_logger(), "Setting camera parameters");
  // 設置參數
  // std::cout << "camera_param_callback_brightness: " << msg.brightness << std::endl;
  brightness_ = int(msg.brightness);
  // std::cout << "camera_param_callback_contrast: " << msg.contrast << std::endl;
  contrast_ = int(msg.contrast);
  // std::cout << "camera_param_callback_saturation: " << msg.saturation << std::endl;
  saturation_ = int(msg.saturation);
  // std::cout << "camera_param_callback_white_balance: " << msg.whitebalance << std::endl;
  auto_white_balance_ = msg.autowhitebalance;

  if (msg.whitebalance != -1) {
    // std::cout << "camera_param_callback_white_balance: " << msg.whitebalance << std::endl;
    white_balance_ = int(msg.whitebalance);
  } else {
    white_balance_ = 0;  // 預設值
  }
  // std::cout << "camera_param_callback_auto_exposure: " << msg.auto_exposure << std::endl;
  auto_exposure_ = msg.auto_exposure;

  // 設置相機參數
  if (!auto_exposure_) {
    m_camera->set_v4l_parameter("auto_exposure", 1);
  } else {
    m_camera->set_v4l_parameter("auto_exposure", 3);
  }

  if (brightness_ >= -64) {
    m_camera->set_v4l_parameter("brightness", (brightness_ + 70) * 2);
  }

  if (contrast_ >= 0) {
    m_camera->set_v4l_parameter("contrast", contrast_ * 2);
  }

  if (saturation_ >= 0) {
    m_camera->set_v4l_parameter("saturation", saturation_ * 2);
  }

  if (auto_white_balance_) {
    m_camera->set_v4l_parameter("white_balance_automatic", 1);
  } else {
    m_camera->set_v4l_parameter("white_balance_automatic", 0);
    m_camera->set_v4l_parameter("white_balance_temperature", white_balance_);
  }
}

void UsbCamNode::location_callback(const tku_msgs::msg::Location &msg){
    location_ = msg.data;
    LoadSet();
    set_v4l2_params();
}

void UsbCamNode::camera_save_callback(const tku_msgs::msg::CameraSave &msg) {
    namespace fs = std::filesystem;

    // 1) 更新目前參數
    brightness_         = msg.brightness;
    contrast_           = msg.contrast;
    saturation_         = msg.saturation;
    white_balance_      = msg.whitebalance;
    auto_white_balance_ = msg.autowhitebalance;
    auto_exposure_      = msg.auto_exposure;

    RCLCPP_INFO(this->get_logger(),
        "[camera_save_callback] b=%d c=%d s=%d wb=%d awb=%d ae=%d",
        brightness_, contrast_, saturation_, white_balance_,
        (int)auto_white_balance_, (int)auto_exposure_);

    // ---- 小工具：trim / 展開 ~ ----
    auto trim_copy = [](std::string s) {
        auto not_ws = [](unsigned char c){ return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_ws));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_ws).base(), s.end());
        return s;
    };
    auto expand_user = [](const std::string& p) -> fs::path {
        if (p.rfind("~/", 0) == 0) {
            if (const char* home = std::getenv("HOME")) return fs::path(home) / p.substr(2);
        }
        return fs::path(p);
    };

    // 2) 讀取 save_dir（當 primary 無法使用時的 fallback）
    if (!this->has_parameter("save_dir")) {
        this->declare_parameter<std::string>("save_dir", "");
    }
    std::string save_dir = this->get_parameter("save_dir").as_string();

    // 3) 正規化 location_：你的輸入會像 "/bb/Parameter"
    auto normalize_loc = [&](std::string s) -> std::string {
        s = trim_copy(std::move(s));
        while (!s.empty() && s.front() == '/') s.erase(s.begin()); // 去掉所有前導 '/'
        while (!s.empty() && s.back()  == '/') s.pop_back();       // 去掉尾端 '/'
        return s; // 變成 "bb/Parameter"
    };
    const std::string loc_clean = normalize_loc(location_);
    if (loc_clean.empty()) {
        RCLCPP_ERROR(this->get_logger(), "[camera_save_callback] location_ is empty; cannot resolve target path.");
        return;
    }

    // 4) 找 strategy 根目錄：優先 tku_STRATEGY_ROOT，否則 ~/tku/src/strategy/strategy，再往上找
    auto find_strategy_root = [&]() -> fs::path {
        if (const char* env = std::getenv("tku_STRATEGY_ROOT")) {
            fs::path p = fs::path(env);
            if (fs::exists(p)) return p;
        }
        if (const char* home = std::getenv("HOME")) {
            fs::path p1 = fs::path(home) / "ros2_adult" / "src" / "strategy" / "strategy";
            if (fs::exists(p1)) return p1;
            fs::path p2 = fs::path(home) / "workspace" / "ros2_adult" / "src" / "strategy" / "strategy";
            if (fs::exists(p2)) return p2;
        }
        // 往上走嘗試找到 tku/src/strategy/strategy
        for (fs::path up = fs::current_path(); !up.empty(); up = up.parent_path()) {
            fs::path cand = up / "ros2_adult" / "src" / "strategy" / "strategy";
            if (fs::exists(cand)) return cand;
        }
        return {};
    };

    fs::path primary_path;
    if (fs::path strategy_root = find_strategy_root(); !strategy_root.empty()) {
        primary_path = strategy_root / loc_clean / "CameraSet.ini";
    }

    // 5) 計算 fallback 路徑（save_dir/CameraSet.ini），以及最終保底 (~/.ros/usb_cam/config/CameraSet.ini)
    fs::path fallback_path;
    if (!save_dir.empty()) {
        fallback_path = expand_user(save_dir) / "CameraSet.ini";
    }
    const char* home = std::getenv("HOME");
    fs::path last_resort = (home ? fs::path(home) : fs::path(".")) / ".ros" / "usb_cam" / "config" / "CameraSet.ini";

    // 6) 依序嘗試寫入（primary → fallback → last_resort）
    auto try_write = [&](const fs::path& path) -> bool {
        if (path.empty()) return false;
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        std::ofstream OutFile(path, std::ios::trunc);
        if (!OutFile.is_open()) return false;

        try {
            OutFile << "[Camera Set Parameter]\n";
            OutFile << "brightness = " << brightness_ << "\n";
            OutFile << "contrast = " << contrast_ << "\n";
            OutFile << "saturation = " << saturation_ << "\n";
            OutFile << "white_balance = " << white_balance_ << "\n";
            OutFile << "auto_white_balance = " << (auto_white_balance_ ? 1 : 0) << "\n";
            OutFile << "auto_exposure = " << (auto_exposure_ ? 1 : 0) << "\n";
            OutFile.close();
            RCLCPP_INFO(this->get_logger(), "[camera_save_callback] Saved to: %s", path.string().c_str());
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[camera_save_callback] Write error to %s: %s",
                         path.string().c_str(), e.what());
            return false;
        }
    };

    if (try_write(primary_path)) return;
    RCLCPP_WARN(this->get_logger(), "[camera_save_callback] Primary failed: %s",
                primary_path.empty() ? "(none)" : primary_path.string().c_str());

    if (try_write(fallback_path)) return;
    RCLCPP_WARN(this->get_logger(), "[camera_save_callback] Fallback(save_dir) failed: %s",
                fallback_path.empty() ? "(none)" : fallback_path.string().c_str());

    if (try_write(last_resort)) return;
    RCLCPP_ERROR(this->get_logger(), "[camera_save_callback] All write attempts failed. Last tried: %s",
                 last_resort.string().c_str());
}
std::string trim(const std::string& str)
{
    auto start = str.begin();
    while (start != str.end() && std::isspace(*start)) {
        start++;
    }
    auto end = str.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}
void UsbCamNode::LoadSet() {
    namespace fs = std::filesystem;

    // ---- 0) 取得 save_dir 參數（當策略路徑失敗時作為 fallback）----
    if (!this->has_parameter("save_dir")) {
        this->declare_parameter<std::string>("save_dir", "");
    }
    std::string save_dir = this->get_parameter("save_dir").as_string();

    auto expand_user = [](const std::string& p) -> fs::path {
        if (p.rfind("~/", 0) == 0) {
            const char* home = std::getenv("HOME");
            if (home) return fs::path(home) / p.substr(2);
        }
        return fs::path(p);
    };

    // ---- 1) 正規化 location_：去空白、移除所有前導'/'、尾端'/' ----
    auto trim_copy = [](std::string s) {
        auto not_ws = [](unsigned char c){ return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_ws));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_ws).base(), s.end());
        while (!s.empty() && s.front() == '/') s.erase(s.begin());  // 重要：移除開頭 '/'
        while (!s.empty() && s.back()  == '/') s.pop_back();
        return s;
    };

    std::string loc_clean = trim_copy(location_);
    if (loc_clean.empty()) {
        RCLCPP_WARN(this->get_logger(), "[LoadSet] location_ is empty; skip loading.");
        return;
    }

    // ---- 2) 取得 strategy_root ----
    fs::path strategy_root;
    if (const char* env = std::getenv("tku_STRATEGY_ROOT")) {
        strategy_root = fs::path(env);
    } else if (const char* home = std::getenv("HOME")) {
        strategy_root = fs::path(home) / "ros2_adult" / "src" / "strategy" / "strategy";
    }

    // ---- 3) 組成主要讀取路徑（Primary）與備援路徑（Fallback: save_dir）----
    fs::path primary_path;
    if (!strategy_root.empty()) {
        primary_path = strategy_root / loc_clean / "CameraSet.ini";
    }
    fs::path fallback_path;
    if (!save_dir.empty()) {
        fallback_path = expand_user(save_dir) / "CameraSet.ini";
    }

    // ---- 4) 嘗試開檔：先 primary，再 fallback ----
    std::ifstream fin;
    fs::path final_path;

    if (!primary_path.empty()) {
        fin.open(primary_path);
        if (fin.is_open()) {
            final_path = primary_path;
            RCLCPP_INFO(this->get_logger(), "[LoadSet] Using strategy path: %s",
                        final_path.string().c_str());
        }
    }

    if (!fin.is_open()) {
        if (!fallback_path.empty()) {
            fin.clear();
            fin.open(fallback_path);
            if (fin.is_open()) {
                final_path = fallback_path;
                RCLCPP_WARN(this->get_logger(),
                            "[LoadSet] Primary missing/unreadable. Fallback to save_dir: %s",
                            final_path.string().c_str());
            }
        }
    }

    if (!fin.is_open()) {
        RCLCPP_ERROR(this->get_logger(),
                     "[LoadSet] Cannot open CameraSet.ini. Tried: %s%s%s",
                     primary_path.empty() ? "(none)" : primary_path.string().c_str(),
                     (!primary_path.empty() && !fallback_path.empty()) ? " and " : "",
                     fallback_path.empty() ? "" : fallback_path.string().c_str());
        return;
    }

    // ---- 5) 讀檔並套用參數 ----
    try {
        std::string line;
        while (std::getline(fin, line)) {
            line = trim_copy(std::move(line));
            if (line.empty() || line[0] == '#' || line[0] == '[') continue;

            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key   = trim_copy(line.substr(0, pos));
            std::string value = trim_copy(line.substr(pos + 1));

            try {
                if      (key == "brightness")         brightness_         = std::stoi(value);
                else if (key == "contrast")           contrast_           = std::stoi(value);
                else if (key == "saturation")         saturation_         = std::stoi(value);
                else if (key == "white_balance")      white_balance_      = std::stoi(value);
                else if (key == "auto_white_balance") auto_white_balance_ = std::stoi(value);
                else if (key == "auto_exposure")      auto_exposure_      = std::stoi(value);
                else {
                    RCLCPP_WARN(this->get_logger(), "[LoadSet] Unknown key: %s", key.c_str());
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "[LoadSet] Bad value for '%s': %s", key.c_str(), e.what());
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[LoadSet] Read error: %s", e.what());
    }

    RCLCPP_INFO(this->get_logger(), "Load_Set");
}

bool UsbCamNode::LoadCameraSetFile(const std::shared_ptr<tku_msgs::srv::CameraInfo::Request> request,
    std::shared_ptr<tku_msgs::srv::CameraInfo::Response> response) {
      (void)request;
    // ----------------------------------------------------------------
    // 複製自 camera_save_callback 的路徑確定邏輯
    // ----------------------------------------------------------------
    LoadSet();
    response->brightness = brightness_;
    response->contrast = contrast_;
    response->saturation = saturation_;
    response->white_balance = white_balance_;
    response->auto_white_balance = auto_white_balance_;
    response->auto_exposure = auto_exposure_;
    return true;
}
/// @brief Send current parameters to V4L2 device
/// TODO(flynneva): should this actuaully be part of UsbCam class?
void UsbCamNode::set_v4l2_params()
{
  // set camera parameters
  // if (m_parameters.brightness >= 0) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'brightness' to %d", m_parameters.brightness);
  //   m_camera->set_v4l_parameter("brightness", m_parameters.brightness);
  // }

  // if (m_parameters.contrast >= 0) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'contrast' to %d", m_parameters.contrast);
  //   m_camera->set_v4l_parameter("contrast", m_parameters.contrast);
  // }

  // if (m_parameters.saturation >= 0) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'saturation' to %d", m_parameters.saturation);
  //   m_camera->set_v4l_parameter("saturation", m_parameters.saturation);
  // }

  // if (m_parameters.sharpness >= 0) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'sharpness' to %d", m_parameters.sharpness);
  //   m_camera->set_v4l_parameter("sharpness", m_parameters.sharpness);
  // }

  // if (m_parameters.gain >= 0) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'gain' to %d", m_parameters.gain);
  //   m_camera->set_v4l_parameter("gain", m_parameters.gain);
  // }

  // // check auto white balance
  // if (m_parameters.auto_white_balance) {
  //   m_camera->set_v4l_parameter("white_balance_automatic", 1);
  //   RCLCPP_INFO(this->get_logger(), "Setting 'white_balance_temperature_auto' to %d", 1);
  // } else {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'white_balance' to %d", m_parameters.white_balance);
  //   m_camera->set_v4l_parameter("white_balance_automatic", 0);
  //   m_camera->set_v4l_parameter("white_balance_temperature", m_parameters.white_balance);
  // }

  // // check auto exposure
  // if (!m_parameters.autoexposure) {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'exposure_auto' to %d", 1);
  //   RCLCPP_INFO(this->get_logger(), "Setting 'exposure' to %d", m_parameters.exposure);
  //   // turn down exposure control (from max of 3)
  //   m_camera->set_v4l_parameter("auto_exposure", 1);
  //   // change the exposure level
  //   m_camera->set_v4l_parameter("exposure_absolute", m_parameters.exposure);
  // } else {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'exposure_auto' to %d", 3);
  //   m_camera->set_v4l_parameter("auto_exposure", 3);
  // }

  // // check auto focus
  // if (m_parameters.autofocus) {
  //   m_camera->set_auto_focus(1);
  //   RCLCPP_INFO(this->get_logger(), "Setting 'focus_auto' to %d", 1);
  //   m_camera->set_v4l_parameter("focus_automatic_continuous", 1);
  // } else {
  //   RCLCPP_INFO(this->get_logger(), "Setting 'focus_auto' to %d", 0);
  //   m_camera->set_v4l_parameter("focus_automatic_continuous", 0);
  //   if (m_parameters.focus >= 0) {
  //     RCLCPP_INFO(this->get_logger(), "Setting 'focus_absolute' to %d", m_parameters.focus);
  //     m_camera->set_v4l_parameter("focus_absolute", m_parameters.focus);
  //   }
  // }


  // 設置相機參數
  if (!auto_exposure_) {
    m_camera->set_v4l_parameter("auto_exposure", 1);
  } else {
    m_camera->set_v4l_parameter("auto_exposure", 3);
  }

  if (brightness_ >= -64) {
    m_camera->set_v4l_parameter("brightness", (brightness_ + 70) * 2);
  }

  if (contrast_ >= 0) {
    m_camera->set_v4l_parameter("contrast", contrast_ * 2);
  }

  if (saturation_ >= 0) {
    m_camera->set_v4l_parameter("saturation", saturation_ * 2);
  }

  if (auto_white_balance_) {
    m_camera->set_v4l_parameter("white_balance_automatic", 1);
  } else {
    m_camera->set_v4l_parameter("white_balance_automatic", 0);
    m_camera->set_v4l_parameter("white_balance_temperature", white_balance_);
  }
}

bool UsbCamNode::take_and_send_image()
{
  // Only resize if required
  if (m_image_msg->data.size() != m_camera->get_image_size_in_bytes()) {
    m_image_msg->width = m_camera->get_image_width();
    m_image_msg->height = m_camera->get_image_height();
    m_image_msg->encoding = m_camera->get_pixel_format()->ros();
    m_image_msg->step = m_camera->get_image_step();
    if (m_image_msg->step == 0) {
      // Some formats don't have a linesize specified by v4l2
      // Fall back to manually calculating it step = size / height
      m_image_msg->step = m_camera->get_image_size_in_bytes() / m_image_msg->height;
    }
    m_image_msg->data.resize(m_camera->get_image_size_in_bytes());
  }

  // grab the image, pass image msg buffer to fill
  m_camera->get_image(reinterpret_cast<char *>(&m_image_msg->data[0]));

  auto stamp = m_camera->get_image_timestamp();
  m_image_msg->header.stamp.sec = stamp.tv_sec;
  m_image_msg->header.stamp.nanosec = stamp.tv_nsec;

  *m_camera_info_msg = m_camera_info->getCameraInfo();
  m_camera_info_msg->header = m_image_msg->header;
  m_image_publisher->publish(*m_image_msg, *m_camera_info_msg);
  return true;
}

bool UsbCamNode::take_and_send_image_mjpeg()
{
  // Only resize if required
  if (sizeof(m_compressed_img_msg->data) != m_camera->get_image_size_in_bytes()) {
    m_compressed_img_msg->format = "jpeg";
    m_compressed_img_msg->data.resize(m_camera->get_image_size_in_bytes());
  }

  // grab the image, pass image msg buffer to fill
  m_camera->get_image(reinterpret_cast<char *>(&m_compressed_img_msg->data[0]));

  auto stamp = m_camera->get_image_timestamp();
  m_compressed_img_msg->header.stamp.sec = stamp.tv_sec;
  m_compressed_img_msg->header.stamp.nanosec = stamp.tv_nsec;

  *m_camera_info_msg = m_camera_info->getCameraInfo();
  m_camera_info_msg->header = m_compressed_img_msg->header;

  m_compressed_image_publisher->publish(*m_compressed_img_msg);
  m_compressed_cam_info_publisher->publish(*m_camera_info_msg);
  return true;
}

rcl_interfaces::msg::SetParametersResult UsbCamNode::parameters_callback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  RCLCPP_DEBUG(this->get_logger(), "Setting parameters for %s", m_parameters.camera_name.c_str());
  m_timer->reset();
  assign_params(parameters);
  set_v4l2_params();
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void UsbCamNode::update()
{
  if (!m_camera->is_capturing()) return;

  try {
    bool ok = (m_parameters.pixel_format_name == "mjpeg") ?
              take_and_send_image_mjpeg() :
              take_and_send_image();
    if (!ok) {
      RCLCPP_WARN_ONCE(this->get_logger(),
                       "USB camera did not respond in time.");
    }
  }
  catch (const char* msg) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::update] camera exception (char*): %s", msg);
    m_camera->stop_capturing();
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::update] camera std::exception: %s", e.what());
    m_camera->stop_capturing();
  }
  catch (...) {
    RCLCPP_ERROR(this->get_logger(),
                 "[UsbCamNode::update] camera unknown exception");
    m_camera->stop_capturing();
  }
}

}  // namespace usb_cam



#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(usb_cam::UsbCamNode)