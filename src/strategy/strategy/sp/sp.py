# 找不到球時，頭上下移動找球
import rclpy
import numpy as np
import sys
from std_msgs.msg import Float32MultiArray
# sys.path.append('/home/iclab/Desktop/kid_hurocup/src/strategy')  # 如果 strategy 不在 PYTHONPATH，才需要手動加路徑
from strategy.API import API
import time

# =========================
# 走路速度 / 轉向相關參數
# =========================
FORWARD_START_SPEED = -1000     # 起步前進速度 2000
BACK_START_SPEED = 3000       # 起步後退速度（負值代表反方向) 
FORWARD_MAX_SPEED = -3000       # 前進速度上限
FORWARD_MIN_SPEED = -1000       # 前進減速階段的下限（這裡設成 2000 => 等於不會真的降到更慢）
BACK_MAX_SPEED = 8000         # 後退速度上限（越接近 0 越慢；-2000 是最快後退）

# 每圈更新速度的變化量
FORWARD_SPEED_ADD = -100        # 前進加速量
FORWARD_SPEED_SUB = -50       # 減速量（負值代表速度往小變）
BACK_SPEED_ADD = 500          # 後退加速量（更負 => 更快後退）

# theta(轉向)的基準偏移
FORWARD_ORIGIN_THETA = 0     # 前進的基準修正
BACK_ORIGIN_THETA = 0      # 後退的基準修正（通常後退要稍微修方向）

# =========================
# 頭部馬達上下限
# =========================
HEAD_Y_HIGH = 2800             # 抬頭最高
HEAD_Y_LOW = 2800              # 低頭最低

# 用來把 speed / theta 綁在一起（前進一組、後退一組）
class parameter:
    def __init__(self, speed, theta):
        self.speed = speed
        self.theta = theta

class SP():

    #這支策略的核心概念：
    #1) 用 SprintBall.find() 找到藍球+紅球 => 得到 center 與 size
    #2) size 當作「距離」指標：越大代表越近
    #3) 用三段式狀態機：
    #  -Forward：正常衝向球
    #  -Decelerating：接近球，準備放慢（避免撞太大力）
    #  -Backward：太近了，開始後退（通常是要越線/保持距離）
    #4) 用 yaw 修正走路方向（theta）
    #5) 用球中心 y 控頭上下（讓球一直在畫面中間）

    def __init__(self, tku_ros_api):
        self.tku_ros_api = tku_ros_api

        # 前進 / 後退的速度與基準 theta
        self.forward = parameter(FORWARD_START_SPEED, FORWARD_ORIGIN_THETA)
        self.backward = parameter(BACK_START_SPEED, BACK_ORIGIN_THETA)

        # SprintBall：用影像找左右兩個顏色球（Blue/Red），合成中心與 size
        self.sp_ball = SprintBall(tku_ros_api)

                # ===== AprilTag 狀態 =====
        self.tag_found = False
        self.tag_id = -1
        self.tag_center_x = 0.0
        self.tag_center_y = 0.0
        self.tag_area = 0.0
        self.target_tag_id = 0   # 要追的 tag id，可自行改

        self.apriltag_sub = self.tku_ros_api.create_subscription(
            Float32MultiArray,
            '/apriltag_info',
            self.apriltag_callback,
            10
        )

        # ===== 掃描模式用的狀態變數 =====
        self.scan_dir = 1        # 掃描方向：1 = 往上，-1 = 往下
        self.scan_step = 15      # 每次掃描移動量（可依相機/比賽調）
        self.ball_found = False   # 本圈是否找到球（避免同圈 find 兩次）
        # 初始化狀態（頭角度、速度等）
        self.init()
    
    def apriltag_callback(self, msg):
        data = msg.data

        if len(data) < 5:
            self.tag_found = False
            self.tag_id = -1
            self.tag_center_x = 0.0
            self.tag_center_y = 0.0
            self.tag_area = 0.0
            return

        self.tag_found = bool(int(data[0]))
        self.tag_id = int(data[1])
        self.tag_center_x = float(data[2])
        self.tag_center_y = float(data[3])
        self.tag_area = float(data[4])

        # self.tku_ros_api.get_logger().info(
        #     f'[TAG_CB] found={self.tag_found} id={self.tag_id} '
        #     f'cx={self.tag_center_x:.1f} cy={self.tag_center_y:.1f} area={self.tag_area:.1f}'
        # )

        
    def status_check(self):
        print("tag_area = ", self.tag_area)
        print("tag_id = ", self.tag_id)
        print("target_tag_id = ", self.target_tag_id)

        # 沒找到 tag，先維持 Forward
        if not self.tag_found:
            return 'Forward'

        # 找到但不是目標 id，也先當沒找到
        # if self.tag_id != self.target_tag_id:
        #     return 'Forward'

        # 門檻之後一定要現場調整
        if 500 >= self.tag_area >= 3000:
            return 'Decelerating'
        elif self.tag_area > 5000:
            return 'Backward'
        
    def head_control(self):
        # 用 AprilTag 中心 y 控頭
        if self.tag_center_y < 110:   # tag 在畫面偏上
            self.head_y += 20
            self.head_y = min(HEAD_Y_HIGH, self.head_y)
            # self.tku_ros_api.get_logger().info('[HEAD] UP (tag high)')

        if self.tag_center_y > 130:   # tag 在畫面偏下
            self.head_y -= 20
            self.head_y = max(HEAD_Y_LOW, self.head_y)
            # self.tku_ros_api.get_logger().info('[HEAD] DOWN (tag low)')

        print("head_y", self.head_y, "tag_center_y", self.tag_center_y)

        self.tku_ros_api.sendHeadMotor(2, self.head_y, 100)
        time.sleep(0.01)

    def angle_control(self, right_theta, left_theta, straight_theta, original_theta):
        
        #用 IMU 的 yaw 來修正走路方向：
        #- yaw > 3  ：右偏 => 右轉（theta = right_theta）
        #- yaw < -3 ：左偏 => 左轉（theta = left_theta）
        #- 否則     ：直走
        #最後再加上 original_theta 當作基準偏移
        
        yaw = self.tku_ros_api.yaw

        if yaw > 3:
            self.theta = right_theta    # 右轉
           # self.tku_ros_api.get_logger().info('Turn Right')
        elif yaw < -3:
            self.theta = left_theta     # 左轉
            #self.tku_ros_api.get_logger().info('Turn Left')
        else:
            self.theta = straight_theta # 直走
            #self.tku_ros_api.get_logger().info('Go Straight')

        # 加上前進/後退的基準補償
        self.theta += original_theta
        # self.tku_ros_api.get_logger().info(f'theta = {self.theta}')

    def speed_control(self, speed, speed_variation, speed_limit, status):
        
        #根據狀態調速度：
        #-Forward：速度(speed)會往上加（取 min，避免超過 speed_limit）
        #- Decelerating/Backward：速度往下（或往更負）變化（取 max，避免小於/超過 speed_limit）
        
        if status == 'Forward':
            # 前進：限制在 speed_limit 以內（上限）
             self.tku_ros_api.get_logger().info('Forward')
             speed = max(speed_limit, speed + speed_variation)
        elif status == 'Decelerating':
             self.tku_ros_api.get_logger().info('Decelerating or Backward')
            # 減速/後退：限制在 speed_limit 以內（對負值來說，max 才是「不要比 -2000 更負」）
             speed = max(speed_limit, speed + speed_variation)
        elif status == 'Backward':
             speed = min(speed_limit, speed + speed_variation)
        self.tku_ros_api.get_logger().info(f'speed = {speed}')
        return speed

    def head_motor_update(self):
        # self.tku_ros_api.get_logger().info('=== HEAD_MOTOR_UPDATE APRILTAG VERSION ===')

        self.ball_found = self.tag_found and (self.tag_id == self.target_tag_id)

        if self.ball_found:
            self.head_control()
            # self.tku_ros_api.get_logger().info(f'[TRACK] head_y = {self.head_y}')
            return

        # 找不到 tag：掃描
        self.head_y += self.scan_dir * self.scan_step

        if self.head_y >= HEAD_Y_HIGH:
            self.head_y = HEAD_Y_HIGH
            self.scan_dir = -1
        elif self.head_y <= HEAD_Y_LOW:
            self.head_y = HEAD_Y_LOW
            self.scan_dir = 1

        self.tku_ros_api.sendHeadMotor(2, self.head_y, 80)
        # self.tku_ros_api.get_logger().info(f'[SCAN] head_y = {self.head_y}')
        
    def init(self):
        
        #初始化策略狀態：
        #-head_y 回到預設
        #-球 size 清 0
        #-前進/後退速度回起始值
        #-頭部馬達回到初始角度
        self.walk_status = 'Forward'
        self.head_y = 1800
        self.sp_ball.size = 0
        self.forward.speed = FORWARD_START_SPEED
        self.backward.speed = BACK_START_SPEED

        # 頭左右 (id=1) 回正 2048；頭上下 (id=2) 設 1800
        self.tku_ros_api.sendHeadMotor(1, 2048, 50)
        self.tku_ros_api.sendHeadMotor(2, 2800, 50)
        time.sleep(0.01)
    

def main(args=None):
    # ROS2 初始化
    rclpy.init(args=args)

    self = API()

    # SprintBall 策略物件
    sp = SP(self)

    # 主迴圈頻率設定
    rate_hz = 30.0
    dt = 1.0 / rate_hz

    # 控制是否開始策略（由外部可能會改 self.is_start）
    self.is_start = False

    first_in =  True      # 用來判斷是否剛進入 start 狀態（只做一次初始化）
    walk_status = 'Forward'

    try:
        while rclpy.ok():
            # 不阻塞地處理一次 ROS callback（收感測器、影像結果、IMU yaw...）
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.is_start:
                # 第一次進入 start：只做一次的初始化
                if first_in:
                    print("123")
                    sp.init()
                    # self.sendbodyAuto(1)         # 開啟自動走路模式
                    first_in = False
                    self.sendSensorReset(True)    # 重置感測器（例如 IMU 或步態狀態）
                # self.sendbodyAuto(1)         # 開啟自動走路模式

                # 若有看到球就更新頭部追蹤
                sp.head_motor_update()

                # ==============
                # 三種行走狀態
                # ==============
                
                if walk_status == 'Forward':
                    # 用 yaw 修正方向 + 設定前進 theta
                    sp.angle_control(-2, 2, 0, FORWARD_ORIGIN_THETA)

                    # 前進速度逐步加
                    sp.forward.speed = sp.speed_control(
                        sp.forward.speed, FORWARD_SPEED_ADD,
                        FORWARD_MAX_SPEED, walk_status
                    )

                    # 送走路指令：speed, side(0)(可能是側移), theta(轉向)
                    self.sendContinuousValue(sp.forward.speed, 0, sp.theta)

                    # 依球大小決定下一狀態
                    walk_status = sp.status_check()

                elif walk_status == 'Decelerating':
                    # 方向修正仍同前進
                    sp.angle_control(-2, 2, 0, FORWARD_ORIGIN_THETA)

                    # 速度逐步減（FORWARD_SPEED_SUB 是負值）
                    sp.forward.speed = sp.speed_control(
                        sp.forward.speed, FORWARD_SPEED_SUB,
                        FORWARD_MIN_SPEED, walk_status
                    )

                    self.sendContinuousValue(sp.forward.speed, 0, sp.theta)
                    walk_status = sp.status_check()

                elif walk_status == 'Backward':
                    # Backward：後退時使用另一個 origin_theta
                    sp.angle_control(-2, 2, 0, BACK_ORIGIN_THETA)

                    # 後退速度逐步增加（更負），但不要超過 BACK_MAX_SPEED
                    sp.backward.speed = sp.speed_control(
                        sp.backward.speed, BACK_SPEED_ADD,
                        BACK_MAX_SPEED, walk_status
                    )
                    self.sendContinuousValue(sp.backward.speed, 0, sp.theta)
                    # walk_status = sp.status_check()
                    
            else:
                # 如果 is_start 被關掉：重置策略，等待下次開始
                if not first_in:
                    self.sendbodyAuto(0)
                walk_status = 'Forward'
                sp.init()
                first_in = True

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            self.destroy_node()
        except Exception:
            pass

    # 關閉 ROS2
    if rclpy.ok():
        rclpy.shutdown()


class Coordinate:
    #自己寫的座標類別，方便做 + - / abs 等運算
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        # 這裡 other 會是數字（例如 /2）
        return Coordinate(self.x / other, self.y / other)

    def __abs__(self):
        return Coordinate(abs(self.x), abs(self.y))

    def __lt__(self, other):
        # 用在排序（但你這份程式好像沒用到）
        return self.x < other.x


class SprintBall:
    
    #SprintBall 的找球方式：
    #- 同時抓到藍球(左) + 紅球(右)
    #- 並且兩球中心 y 高度差很小（同一水平）
    #- 且藍球在左、紅球在右
    # 符合才算找到「有效球」
    
    def __init__(self, tku_ros_api):
        self.tku_ros_api = tku_ros_api

        # 兩種顏色的球
        self.left_side = ObjectInfo('Blue', tku_ros_api)
        self.right_side = ObjectInfo('Red', tku_ros_api)

        self.size = 0
        self.center = Coordinate(0, 0)

    def find(self):
        
        #同時偵測到藍球和紅球，且兩者高度差很小、位置順序合理，才算找到有效目標。
        #回傳 True/False。
        
        find_left = self.left_side.update()
        find_right = self.right_side.update()
        print("left :", find_left)
        print("right :", find_right)

        if find_left and find_right:
            # 兩球中心的差（取 abs）
            center_diff = abs(self.left_side.center - self.right_side.center)
            print("center_diff_y = ", center_diff.y)

            # 你這裡的條件：
            # 1) 兩邊 y 差 < 5（同高度）
            # 2) 左球的 edge_min < 右球 edge_min（左球在左）
            # 3) 左球的 edge_max < 右球 edge_max（左球整個都在右球左邊）
            if center_diff.y <= 5 and (self.left_side.edge_min < self.right_side.edge_min) \
               and (self.left_side.edge_max < self.right_side.edge_max):

                # 畫框：藍色框、紅色框
                self.tku_ros_api.drawImageFunction(1, 1, *self.left_side.boundary_box, 0, 0, 255)
                self.tku_ros_api.drawImageFunction(2, 1, *self.right_side.boundary_box, 255, 0, 0)

                # 紀錄 size 與 center
                # self.tku_ros_api.get_logger().info(
                #     f'left_ball_size = {self.left_side.size}, right_ball_size = {self.right_side.size}'
                # )
                # size 用兩邊面積相加
                self.size = self.left_side.size + self.right_side.size
                # center 用兩邊中心平均
                self.center = (self.left_side.center + self.right_side.center) / 2

                return True

            
            center_diff = 0

        return False


class ObjectInfo:
    
    #影像物件資訊：
    #- 由 API 的 get_objects(color_name) 取得 list
    #- 每個物件內含 bbox / area 等資料
    #- 篩選出 size 在一定範圍內的物件，並保存其邊界、中心、大小
    
    # 顏色字串對應到你的系統 color index
    color_dict = {'Orange': 0,
                  'Yellow': 1,
                  'Blue':   2,
                  'Green':  3,
                  'Black':  4,
                  'Red':    5,
                  'White':  6 }

    def __init__(self, color, tku_ros_api):
        self.tku_ros_api = tku_ros_api
        self.color = self.color_dict[color]

        self.edge_max = Coordinate(0, 0)  # 右下角/最大邊界
        self.edge_min = Coordinate(0, 0)  # 左上角/最小邊界
        self.center = Coordinate(0, 0)    # 中心點
        self.size = 0                     # 面積（或 bbox 面積）

    @property
    def boundary_box(self):
        # 供 drawImageFunction 使用：(xmin, xmax, ymin, ymax)
        return (self.edge_min.x, self.edge_max.x, self.edge_min.y, self.edge_max.y)

    def update(self):
    
        #從影像偵測結果中找指定顏色的物件：
        #- 如果找不到回 False
        #- 找到符合 size 範圍的第一個物件就更新資訊並回 True
        
        # 對應 API 需要的 color_name 字串
        color_name = ['orange', 'yellow', 'blue', 'green', 'black', 'red', 'white', 'others'][self.color]

        # 取得所有該顏色物件（list of dict）
        objs = self.tku_ros_api.get_objects(color_name)
        
        # 沒物件直接 False
        if not objs:
            return False

        for o in objs:
            try:
                # bbox = (x, y, w, h)
                x, y, w, h = o['bbox']
                # area 若有提供就用 area，沒有就用 w*h
                size = float(o.get('area', w * h))
            except Exception:
                # 當資料格式不正確就跳過
                continue

            # 篩選：只接受某個大小範圍的物件，避免雜訊
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


if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
