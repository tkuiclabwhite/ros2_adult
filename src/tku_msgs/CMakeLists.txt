cmake_minimum_required(VERSION 3.8)
project(tku_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


# find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  #########     vision/msg    ##################
  "msg/ButtonColorForm.msg"
  "msg/ColorArray.msg"
  "msg/ColorData.msg"
  "msg/DrawImage.msg"
  "msg/HSVValue.msg"
  "msg/LabelModelObjectList.msg"
  "msg/ObjectList.msg"
  "msg/Camera.msg"
  "msg/CameraSave.msg"
  "msg/Location.msg"
  "msg/Dio.msg"
  "msg/Zoom.msg"
  "msg/OpenCvOrder.msg"
  "msg/YUVValue.msg"
  ##########  walkingparameter/msg  ################
  "msg/Parameter.msg"
  "msg/Interface.msg"
  "msg/Parametermessage.msg"

  ##########  motionpackage/msg  ################
  "msg/Callback.msg"
  "msg/HeadPackage.msg"
  "msg/InterfaceSend2Sector.msg"
  "msg/PIDpackage.msg"
  "msg/SandHandSpeed.msg"
  "msg/SaveMotion.msg"
  "msg/SaveMotionVector.msg"
  "msg/SensorPackage.msg"
  "msg/SensorSet.msg"
  "msg/SingleMotorData.msg"
  "msg/Walkingmessage.msg"
  #########     vision/srv    ##################
  "srv/BuildModel.srv"
  "srv/HSVInfo.srv"
  "srv/YUVInfo.srv"
  "srv/OpenCvInfo.srv"
  "srv/SaveYUV.srv"
  "srv/SaveOpenCv.srv"
  "srv/SaveHSV.srv"
  "srv/Add.srv"
  "srv/CameraInfo.srv"
  ##########  motionpackage/srv  ################
  "srv/ReadMotion.srv"
  "srv/CheckSector.srv"
  "srv/SendHandSpeed.srv"
  ##########  walkingparameter/srv  ################
  "srv/WalkingGaitParameter.srv"

  DEPENDENCIES std_msgs
  # DEPENDENCIES geometry_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
