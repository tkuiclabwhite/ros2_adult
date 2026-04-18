#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略：找不到球 -> 頭上下掃描找球
      找到球 -> 頭追球(用 center.y) + 走歪用球中心 x 偏移修正(theta)
並且每圈用「面板格式」輸出狀態（像你截圖那種形式）
"""

import rclpy
import time
from strategy.API import API

# =========================
# 影像尺寸（依你圖：320x240）
# =========================
IMG_W = 320
IMG_H = 240
CX_TARGET = IMG_W // 2          # 160
DEADBAND_X = 20                 # 球中心偏移容忍範圍（px）
KP_X = 0.03                     # 影像對準增益（要調）
THETA_MAX = 6                   # 最大轉向（要調）

# =========================
# 走路速度 / 轉向相關參數
# =========================
FORWARD_START_SPEED = 2500
BACK_START_SPEED = -2000
FORWARD_MAX_SPEED = 2500
FORWARD_MIN_SPEED = 2000
BACK_MAX_SPEED = -2500

FORWARD_SPEED_ADD = 100
FORWARD_SPEED_SUB = -300
BACK_SPEED_ADD = -100

FORWARD_ORIGIN_THETA = 0
BACK_ORIGIN_THETA = -1

# =========================
# 頭部馬達上下限
# =========================
HEAD_Y_HIGH = 1800
HEAD_Y_LOW = 1400


# -------------------------
# 小工具：清畫面 + bool格式
# -------------------------
def clear_terminal():
    # 讓輸出像監控面板一樣固定刷新
    print("\033[2J\033[H", end="")  # ANSI clear screen + move cursor to top-left


def fmt_bool(v):
    return "True" if v else "False"


# -------------------------
# 資料結構：把 speed/theta 綁一起
# -------------------------
class parameter:
    def __init__(self, speed, theta):
        self.speed = speed
        self.theta = theta


# ============================================================
# SP：策略主體（頭掃描/追球、行走狀態機、轉向修正、面板輸出）
# ============================================================
class SP:
    def __init__(self, tku_ros_api):
        self.tku_ros_api = tku_ros_api

        # 前進 / 後退 的速度參數
        self.forward = parameter(FORWARD_START_SPEED, FORWARD_ORIGIN_THETA)
        self.backward = parameter(BACK_START_SPEED, BACK_ORIGIN_THETA)

        # 找球器
        self.sp_ball = SprintBall(tku_ros_api)

        # 掃描模式狀態
        self.scan_dir = 1
        self.scan_step = 15

        # 本圈是否找到球（避免同圈 find 兩次）
        self.ball_found = False

        # 本圈 theta（送給 sendContinuousValue 的轉向量）
        self.theta = 0

        # 額外 debug（可在面板顯示）
        self.find_left = False
        self.find_right = False
        self.center_diff_y = 0

        self.init()

    # -------------------------
    # 狀態機判斷（靠 size）
    # -------------------------
    def status_check(self):
        # 沒球：保持 Forward（不要亂後退）
        if not self.ball_found:
            return 'Forward'

        size = self.sp_ball.size

        # ✅ 這裡門檻要依你實測調整
        # 下面是比較「常見」的示範：
        if 1600 >= size >= 1500:
            return 'Decelerating'
        elif size > 1600:
            return 'Backward'
        return 'Forward'

    # -------------------------
    # 頭追球（依 center.y）
    # -------------------------
    def head_control(self):
        cy = int(self.sp_ball.center.y)

        # 球偏上：抬頭
        if cy < 110:
            self.head_y += 20
            self.head_y = min(HEAD_Y_HIGH, self.head_y)
            self.tku_ros_api.get_logger().info('[HEAD] UP (ball high)')

        # 球偏下：低頭
        elif cy > 130:
            self.head_y -= 20
            self.head_y = max(HEAD_Y_LOW, self.head_y)
            self.tku_ros_api.get_logger().info('[HEAD] DOWN (ball low)')

        else:
            self.tku_ros_api.get_logger().info('[HEAD] HOLD (ball mid)')

        self.tku_ros_api.sendHeadMotor(2, int(self.head_y), 100)
        self.tku_ros_api.sendbodyAuto(1)

        time.sleep(0.01)

    # -------------------------
    # IMU yaw 修正（沒球用）
    # -------------------------
    def angle_control_by_yaw(self, right_theta, left_theta, straight_theta, original_theta):
        yaw = float(getattr(self.tku_ros_api, "yaw", 0.0))

        if yaw > 3:
            base = right_theta
        elif yaw < -3:
            base = left_theta
        else:
            base = straight_theta

        self.theta = int(base + original_theta)
        self.tku_ros_api.get_logger().info(f'[YAW] yaw={yaw:.2f} theta(int)={self.theta}')

    # -------------------------
    # 影像偏移修正（有球用）
    # -------------------------
    def angle_control_by_ball(self, original_theta):
        cx = int(self.sp_ball.center.x)
        err = cx - CX_TARGET

        # deadband 避免抖動
        if abs(err) <= DEADBAND_X:
            corr = 0.0
        else:
            corr = KP_X * err

        # 限幅
        corr = max(-THETA_MAX, min(THETA_MAX, corr))

        # ✅ 必須轉 int，避免 Interface.theta assert
        self.theta = int(round(corr + original_theta))

        self.tku_ros_api.get_logger().info(
            f'[BALL_ALIGN] cx={cx} err={err} corr={corr:.2f} theta(int)={self.theta}'
        )

    # -------------------------
    # 速度控制（Forward / Decelerating / Backward）
    # -------------------------
    def speed_control(self, speed, speed_variation, speed_limit, status):
        if status == 'Forward':
            speed = min(speed_limit, speed + speed_variation)
            self.tku_ros_api.get_logger().info('Forward')
        else:
            speed = max(speed_limit, speed + speed_variation)
            self.tku_ros_api.get_logger().info('Decelerating or Backward')

        self.tku_ros_api.get_logger().info(f'speed = {speed}')
        return speed

    # -------------------------
    # 頭更新：有球追球；沒球掃描
    # -------------------------
    def head_motor_update(self):
        # 本圈只找一次
        self.ball_found = self.sp_ball.find()

        # 把 SprintBall.find() 裡的 debug 結果拉出來顯示
        self.find_left = self.sp_ball.find_left
        self.find_right = self.sp_ball.find_right
        self.center_diff_y = self.sp_ball.center_diff_y

        if self.ball_found:
            self.head_control()
            return

        # 沒球：掃描
        self.head_y += self.scan_dir * self.scan_step

        if self.head_y >= HEAD_Y_HIGH:
            self.head_y = HEAD_Y_HIGH
            self.scan_dir = -1
        elif self.head_y <= HEAD_Y_LOW:
            self.head_y = HEAD_Y_LOW
            self.scan_dir = 1

        self.tku_ros_api.sendHeadMotor(2, int(self.head_y), 80)

    # -------------------------
    # 初始化
    # -------------------------
    def init(self):
        self.head_y = 1800
        self.sp_ball.size = 0
        self.forward.speed = FORWARD_START_SPEED
        self.backward.speed = BACK_START_SPEED
        self.theta = 0
        self.ball_found = False

        # 頭左右(id=1)回正 2048；頭上下(id=2)到 1800
        self.tku_ros_api.sendHeadMotor(1, 2048, 50)
        self.tku_ros_api.sendHeadMotor(2, 1800, 50)
        time.sleep(0.01)

    # -------------------------
    # 面板輸出：像你截圖那種格式
    # -------------------------
    def print_status(self, is_start, walk_status):
        yaw_api = float(getattr(self.tku_ros_api, "yaw", 0.0))
        yaw_imu = yaw_api  # 你目前沒 imu rpy 就先同 yaw

        ball_size = float(getattr(self.sp_ball, "size", 0.0))
        cx = int(getattr(self.sp_ball.center, "x", 0))
        cy = int(getattr(self.sp_ball.center, "y", 0))

        head_y = int(getattr(self, "head_y", 0))
        theta = int(getattr(self, "theta", 0))

        clear_terminal()

        print("#==============機器人狀態==============#")
        print(f"is_start        : {fmt_bool(is_start)}")
        print(f"walk_status      : {walk_status}")
        print(f"yaw(API)         : {yaw_api:.2f}")
        print(f"imu yaw(imu rpy)  : {yaw_imu:.2f}")

        print("#==============SprintBall==============#")
        print(f"left_found       : {fmt_bool(self.find_left)}")
        print(f"right_found      : {fmt_bool(self.find_right)}")
        print(f"ball_found       : {fmt_bool(self.ball_found)}")
        print(f"center_diff_y    : {int(self.center_diff_y)}")
        print(f"ball_size        : {ball_size:.1f}")
        print(f"ball_center      : ({cx}, {cy})")

        print("#=============控制輸出===============#")
        print(f"head_y           : {head_y}")
        print(f"theta            : {theta}")


# ============================================================
# Coordinate / SprintBall / ObjectInfo：找球相關
# ============================================================
class Coordinate:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        return Coordinate(self.x / other, self.y / other)

    def __abs__(self):
        return Coordinate(abs(self.x), abs(self.y))

    def __lt__(self, other):
        return self.x < other.x


class SprintBall:
    """
    同時抓到藍球(左) + 紅球(右)，且兩球中心 y 差小、左右順序合理，才算找到有效球
    """
    def __init__(self, tku_ros_api):
        self.tku_ros_api = tku_ros_api
        self.left_side = ObjectInfo('Blue', tku_ros_api)
        self.right_side = ObjectInfo('Red', tku_ros_api)

        self.size = 0
        self.center = Coordinate(0, 0)

        # debug：讓 SP.print_status 能顯示
        self.find_left = False
        self.find_right = False
        self.center_diff_y = 0

    def find(self):
        self.find_left = self.left_side.update()
        self.find_right = self.right_side.update()

        if self.find_left and self.find_right:
            center_diff = abs(self.left_side.center - self.right_side.center)
            self.center_diff_y = int(center_diff.y)

            if self.center_diff_y <= 5 and (self.left_side.edge_min < self.right_side.edge_min) \
               and (self.left_side.edge_max < self.right_side.edge_max):

                # 畫框：藍(left)、紅(right)
                self.tku_ros_api.drawImageFunction(1, 1, *self.left_side.boundary_box, 0, 0, 255)
                self.tku_ros_api.drawImageFunction(2, 1, *self.right_side.boundary_box, 255, 0, 0)

                self.size = float(self.left_side.size + self.right_side.size)
                self.center = (self.left_side.center + self.right_side.center) / 2
                return True

        # 沒找到：清一下 debug（避免顯示舊值）
        self.size = 0
        self.center = Coordinate(0, 0)
        self.center_diff_y = 0
        return False


class ObjectInfo:
    color_dict = {'Orange': 0, 'Yellow': 1, 'Blue': 2, 'Green': 3, 'Black': 4, 'Red': 5, 'White': 6}

    def __init__(self, color, tku_ros_api):
        self.tku_ros_api = tku_ros_api
        self.color = self.color_dict[color]

        self.edge_max = Coordinate(0, 0)
        self.edge_min = Coordinate(0, 0)
        self.center = Coordinate(0, 0)
        self.size = 0.0

    @property
    def boundary_box(self):
        return (self.edge_min.x, self.edge_max.x, self.edge_min.y, self.edge_max.y)

    def update(self):
        color_name = ['orange', 'yellow', 'blue', 'green', 'black', 'red', 'white', 'others'][self.color]
        objs = self.tku_ros_api.get_objects(color_name)

        if not objs:
            return False

        for o in objs:
            try:
                x, y, w, h = o['bbox']
                size = float(o.get('area', w * h))
            except Exception:
                continue

            # ✅ 你要求：10~8000
            if 10 < size < 8000:
                self.edge_min.x = int(x)
                self.edge_max.x = int(x + w)
                self.edge_min.y = int(y)
                self.edge_max.y = int(y + h)

                self.center.x = int(x + w / 2)
                self.center.y = int(y + h / 2)
                self.size = size
                return True

        return False


# ============================================================
# main：ROS2 loop
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    self = API()
    sp = SP(self)

    rate_hz = 30.0
    dt = 1.0 / rate_hz

    self.is_start = True
    first_in = True
    walk_status = 'Forward'

    try:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.is_start:
                if first_in:
                    sp.init()
                    self.sendbodyAuto(1)
                    self.sendSensorReset(True)
                    first_in = False

                # 先更新頭（同時更新 sp.ball_found）
                sp.head_motor_update()

                # -------------------------
                # Forward
                # -------------------------
                if walk_status == 'Forward':
                    # 有球：用影像對準；沒球：用 yaw 基本修正
                    print('aa')
                    if sp.ball_found:
                        sp.angle_control_by_ball(FORWARD_ORIGIN_THETA)
                    else:
                        sp.angle_control_by_yaw(-2, 2, 0, FORWARD_ORIGIN_THETA)

                    sp.forward.speed = sp.speed_control(
                        sp.forward.speed, FORWARD_SPEED_ADD,
                        FORWARD_MAX_SPEED, walk_status
                    )

                    self.sendContinuousValue(int(sp.forward.speed), 0, int(sp.theta))
                    walk_status = sp.status_check()

                # -------------------------
                # Decelerating
                # -------------------------
                elif walk_status == 'Decelerating':
                    if sp.ball_found:
                        sp.angle_control_by_ball(FORWARD_ORIGIN_THETA)
                    else:
                        sp.angle_control_by_yaw(-2, 2, 0, FORWARD_ORIGIN_THETA)

                    sp.forward.speed = sp.speed_control(
                        sp.forward.speed, FORWARD_SPEED_SUB,
                        FORWARD_MIN_SPEED, walk_status
                    )

                    self.sendContinuousValue(int(sp.forward.speed), 0, int(sp.theta))
                    walk_status = sp.status_check()

                # -------------------------
                # Backward
                # -------------------------
                else:
                    if sp.ball_found:
                        sp.angle_control_by_ball(BACK_ORIGIN_THETA)
                    else:
                        sp.angle_control_by_yaw(-2, 2, 0, BACK_ORIGIN_THETA)

                    sp.backward.speed = sp.speed_control(
                        sp.backward.speed, BACK_SPEED_ADD,
                        BACK_MAX_SPEED, walk_status
                    )

                    self.sendContinuousValue(int(sp.backward.speed), 0, int(sp.theta))

            else:
                if not first_in:
                    self.sendbodyAuto(1)
                walk_status = 'Forward'
                sp.init()
                first_in = True

            # ✅ 每圈用「面板格式」輸出
            # sp.print_status(api.is_start, walk_status)

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            self.destroy_node()
        except Exception:
            pass

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
