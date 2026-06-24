"""
ball_to_world.py — 影像像素 → 世界座標(ray-cast 法,座標系對齊舊 homography)

★ 此版輸出座標系與舊 pixel_to_world.py 完全一致:
    +X = 影像右方
    +Y = 往前(遠離相機,朝畫面上緣)
    原點 = 相機正下方地面點
  因此可直接取代 bb.py 裡的 PixelToWorld,IK 端(ik_solve.py)完全不用改。

相比舊 homography 的升級:
  - 用真實標定 K + 畸變校正(舊法無內參,邊緣誤差大)
  - ray-cast 到 z=h 平面,物體有高度時更準(舊法用 k=(H-h)/H 近似)

用法(bb.py):
  from ball_to_world import PixelToWorldRaycast
  mapper = PixelToWorldRaycast.load("camera_params.npz", ball_height_cm=6.5)
  Xw, Yw = mapper.bbox_to_world(x1, y1, x2, y2)   # 介面同舊版

★ 跑前確認:
  1. camera_params.npz 路徑
  2. ball_height_cm = 球心離地高度(舊法是 ball_radius_cm=6.5)
  3. 相機高/俯角若不同,改 CAMERA_HEIGHT / DEPRESSION_DEG
"""
import numpy as np
import cv2
import os

# ============================================================
# ★ 已確認參數 ★
# ============================================================
CAMERA_HEIGHT = 105.0      # 相機光心離地 (cm)
DEPRESSION_DEG = 66.1      # 俯角:從水平往下 (deg),已驗證
YAW_DEG = 0.0              # 左右偏擺
# ============================================================


def _build_extrinsics(height_cm, depression_deg, yaw_deg=0.0):
    """
    回傳 R_wc, C_w。
    ★座標系定義(對齊舊 homography)★:
      +X_w = 影像右方
      +Y_w = 往前(遠離相機,畫面上緣方向)
      +Z_w = 上, 地面 z=0
    俯角:從水平往下 depression_deg。
    """
    C_w = np.array([0.0, 0.0, height_cm])
    dep = np.deg2rad(depression_deg)
    yaw = np.deg2rad(yaw_deg)

    # 光軸:水平時指向 +Y(前),俯角後向下傾斜
    #   前方分量在 +Y,向下分量在 -Z
    z_c = np.array([np.sin(yaw) * np.cos(dep),   # 偏擺帶來的側向
                    np.cos(yaw) * np.cos(dep),   # 前方(+Y)
                    -np.sin(dep)])               # 向下(-Z)
    # 相機 X 軸(影像右)= 世界 +X 方向(水平面內,垂直光軸水平投影)
    x_c = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    # 相機 Y 軸(影像下)= Z_c × X_c
    y_c = np.cross(z_c, x_c)

    R_wc = np.column_stack([x_c, y_c, z_c])
    return R_wc, C_w


class PixelToWorldRaycast:
    """ray-cast 版,介面對齊舊 PixelToWorld(load / bbox_to_world)。"""

    def __init__(self, K, dist, ball_height_cm=6.5,
                 camera_height=CAMERA_HEIGHT,
                 depression_deg=DEPRESSION_DEG,
                 yaw_deg=YAW_DEG):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.R_wc, self.C_w = _build_extrinsics(
            camera_height, depression_deg, yaw_deg)
        self.ball_height = ball_height_cm

    @classmethod
    def load(cls, path, ball_height_cm=6.5, **kw):
        data = np.load(path)
        return cls(data["K"], data["dist"], ball_height_cm=ball_height_cm, **kw)

    def transform(self, u, v, plane_z=None):
        """像素 (u,v) -> 世界 (X, Y) cm。介面同舊版 transform。"""
        if plane_z is None:
            plane_z = self.ball_height

        # 去畸變(舊 homography 沒有,這是精度升級關鍵)
        pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        uu, vv = float(und[0, 0, 0]), float(und[0, 0, 1])

        # 反投影成射線
        d_c = self.Kinv @ np.array([uu, vv, 1.0])
        d_w = self.R_wc @ d_c
        n = np.linalg.norm(d_w)
        if n < 1e-12:
            raise ValueError("射線方向退化")
        d_w = d_w / n

        # 與 z=plane_z 平面求交
        if abs(d_w[2]) < 1e-9:
            raise ValueError("射線平行於平面")
        t = (plane_z - self.C_w[2]) / d_w[2]
        if t <= 0:
            raise ValueError("交點在相機背後")

        P = self.C_w + t * d_w
        return float(P[0]), float(P[1])   # (X右, Y前)

    def bbox_to_world(self, x1, y1, x2, y2, ball_radius_cm=None):
        """
        介面完全對齊舊版 bbox_to_world。
        ball_radius_cm: 若傳入則當作球心高度覆蓋(相容舊呼叫);否則用 self.ball_height
        """
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        plane_z = ball_radius_cm if ball_radius_cm is not None else self.ball_height
        return self.transform(u, v, plane_z=plane_z)


# ============================================================
# 直接跑 = 驗證座標系與舊 homography 方向一致
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("驗證:ray-cast 輸出方向是否對齊舊 homography")
    print("=" * 60)

    # 用你真實標定參數
    K = np.array([[154.32732187, 0, 159.89507983],
                  [0, 162.45946373, 131.80388869],
                  [0, 0, 1.0]])
    dist = np.array([0.07443338, -0.1800464, -0.00790302, 0.01420315, 0.06262373])

    m = PixelToWorldRaycast(K, dist, ball_height_cm=6.5)

    tests = [
        ("畫面中心", 160, 120),
        ("畫面右側", 240, 120),
        ("畫面左側", 80, 120),
        ("畫面上緣(遠)", 160, 20),
        ("畫面下緣(近)", 160, 220),
    ]
    print(f"\n{'位置':16s} {'像素':12s} {'X(右+)':>10s} {'Y(前+)':>10s}")
    for name, u, v in tests:
        X, Y = m.transform(u, v)
        print(f"{name:16s} ({u:3d},{v:3d})     {X:10.2f} {Y:10.2f}")

    print("\n方向特徵檢查(必須與舊法同向):")
    xr, _ = m.transform(240, 120)
    xl, _ = m.transform(80, 120)
    print(f"  右側X={xr:.1f} > 左側X={xl:.1f}  -> {'右為正 ✓ 對齊' if xr > xl else '✗ 方向反了'}")
    _, yf = m.transform(160, 20)
    _, yn = m.transform(160, 220)
    print(f"  上緣Y={yf:.1f} > 下緣Y={yn:.1f}  -> {'上緣(遠)Y大 ✓ 對齊' if yf > yn else '✗ 方向反了'}")
