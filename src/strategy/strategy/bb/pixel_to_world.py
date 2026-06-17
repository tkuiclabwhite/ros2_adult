#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地面平面 像素 <-> 世界座標 轉換 (單應矩陣法)

世界座標定義:
  原點 (0,0) = 相機鏡頭正下方的地面點
  +X = 影像右方
  +Y = 往前 (遠離相機,朝畫面上緣方向)
  單位 = cm

標定來源 = 實測梯形四角對應到影像四角,不需要相機內參。
若重新量測,只要改下方 CONFIG 的數字即可。
"""
import numpy as np

# ======================= CONFIG =======================
# 影像解析度
IMG_W = 320
IMG_H = 240

# 實測梯形 (cm)
TOP_WIDTH    = 147.5   # 上底  -> 對應畫面「上緣」(遠端)
BOTTOM_WIDTH = 120.0   # 下底  -> 對應畫面「下緣」(近端)
SLANT        = 104.0   # 左右斜邊
NEAR_DIST    = -8.0     # 下緣中點 到 相機正下方地面 的水平距離

# 影像左右是否被鏡像 (C930c 若開了 mirror 設 True)
MIRROR_X = False

# --- 以下僅供交叉驗算,不影響轉換結果 ---
CAMERA_HEIGHT       = 105.0
MOTOR_HORIZONTAL    = 2048    # 水平時的刻度
MOTOR_CURRENT       = 2770    # 目前刻度
MOTOR_TICKS_PER_REV = 4096    # Dynamixel 一圈
# ======================================================


def _build_world_corners():
    half_top = TOP_WIDTH / 2.0
    half_bot = BOTTOM_WIDTH / 2.0
    lateral_run = half_top - half_bot
    if SLANT**2 < lateral_run**2:
        raise ValueError("斜邊長度小於上下底差的一半,梯形不成立")
    depth = np.sqrt(SLANT**2 - lateral_run**2)
    far_y  = NEAR_DIST + depth
    near_y = NEAR_DIST
    s = -1.0 if MIRROR_X else 1.0
    world = np.array([
        [-s*half_top, far_y],   # 影像左上
        [ s*half_top, far_y],   # 影像右上
        [-s*half_bot, near_y],  # 影像左下
        [ s*half_bot, near_y],  # 影像右下
    ], dtype=float)
    return world, depth


def _homography(src, dst):
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y, -X])
        A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y, -Y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


# ---- 建立轉換矩陣 (import 時就算好) ----
_IMG_CORNERS = np.array([
    [0,     0],
    [IMG_W, 0],
    [0,     IMG_H],
    [IMG_W, IMG_H],
], dtype=float)
_WORLD_CORNERS, _DEPTH = _build_world_corners()
H     = _homography(_IMG_CORNERS, _WORLD_CORNERS)   # pixel -> world
H_INV = np.linalg.inv(H)                            # world -> pixel


def pixel_to_world(u, v, cyl_height=0.0):
    p = H @ np.array([float(u), float(v), 1.0])
    Xg, Yg = p[0] / p[2], p[1] / p[2]
    k = (CAMERA_HEIGHT - cyl_height) / CAMERA_HEIGHT
    return Xg * k, Yg * k


def world_to_pixel(X, Y):
    """世界座標 (X,Y) cm -> 影像像素 (u, v)"""
    p = H_INV @ np.array([float(X), float(Y), 1.0])
    return p[0] / p[2], p[1] / p[2]


def save_npz(path: str = "homography.npz") -> None:
    """將目前 CONFIG 算出的 H 矩陣存成 .npz 檔，供其他模組載入使用。"""
    np.savez(
        path,
        H=H,
        H_inv=H_INV,
        image_points=_IMG_CORNERS,
        world_points=_WORLD_CORNERS,
    )
    print(f"[已存檔] {path}")
    print(f"  H =\n{H}")


def load_npz(path: str = "homography.npz"):
    """從 .npz 載入 H，回傳 (H, H_inv)。"""
    data = np.load(path)
    return data["H"], data["H_inv"]


# ======================= 自我測試 =======================
def _self_test():
    np.set_printoptions(suppress=True, precision=8)
    print("=" * 56)
    print("CONFIG")
    print(f"  解析度        : {IMG_W} x {IMG_H}")
    print(f"  上底/下底/斜邊: {TOP_WIDTH} / {BOTTOM_WIDTH} / {SLANT} cm")
    print(f"  下緣距相機    : {NEAR_DIST} cm")
    print(f"  鏡像 MIRROR_X : {MIRROR_X}")
    print(f"  梯形縱深 D    : {_DEPTH:.4f} cm")
    print("=" * 56)

    print("\nHomography H (pixel -> world):")
    print(H)

    print("\n[四角還原驗算] (應等於標定值)")
    names = ["左上(遠左)", "右上(遠右)", "左下(近左)", "右下(近右)"]
    ok = True
    for (u, v), tgt, nm in zip(_IMG_CORNERS, _WORLD_CORNERS, names):
        X, Y = pixel_to_world(u, v)
        err = np.hypot(X - tgt[0], Y - tgt[1])
        ok &= err < 1e-6
        print(f"  {nm}  px({u:4.0f},{v:4.0f}) -> ({X:8.3f},{Y:8.3f})  "
              f"目標({tgt[0]:8.3f},{tgt[1]:8.3f})  誤差={err:.2e}")
    print(f"  => 四角還原 {'通過' if ok else '失敗'}")

    print("\n[往返一致性驗算] pixel -> world -> pixel")
    ok2 = True
    for u, v in [(160, 120), (50, 30), (300, 200), (10, 230)]:
        X, Y = pixel_to_world(u, v)
        u2, v2 = world_to_pixel(X, Y)
        err = np.hypot(u - u2, v - v2)
        ok2 &= err < 1e-6
        print(f"  px({u:3d},{v:3d}) -> ({X:7.2f},{Y:7.2f}) -> "
              f"px({u2:7.3f},{v2:7.3f})  誤差={err:.2e}")
    print(f"  => 往返一致 {'通過' if ok2 else '失敗'}")

    print("\n[取樣點]")
    for u, v in [(160, 120), (160, 0), (160, 240), (80, 120), (240, 60)]:
        X, Y = pixel_to_world(u, v)
        print(f"  px({u:3d},{v:3d}) -> X={X:8.3f}  Y={Y:8.3f}")

    print("\n[交叉驗算] 俯角/高度法 vs 梯形法")
    pitch = (MOTOR_CURRENT - MOTOR_HORIZONTAL) / MOTOR_TICKS_PER_REV * 360.0
    half_vfov = 90.0 - pitch          # NEAR_DIST=0 -> 下緣光線垂直
    top_dep = pitch - half_vfov
    far_by_angle = CAMERA_HEIGHT / np.tan(np.radians(top_dep))
    far_by_trap = NEAR_DIST + _DEPTH
    diff = abs(far_by_angle - far_by_trap)
    print(f"  馬達俯角 = ({MOTOR_CURRENT}-{MOTOR_HORIZONTAL})/{MOTOR_TICKS_PER_REV}*360 = {pitch:.4f} deg")
    print(f"  推得 VFOV = {2*half_vfov:.2f} deg, 上緣俯角 = {top_dep:.2f} deg")
    print(f"  遠端距離: 俯角法={far_by_angle:.2f} cm  梯形法={far_by_trap:.2f} cm  "
          f"差={diff:.2f} cm ({diff/far_by_trap*100:.1f}%)")

    print("\n" + "=" * 56)
    print(f"總結: 四角{'通過' if ok else '失敗'} / 往返{'通過' if ok2 else '失敗'}")
    print("=" * 56)


if __name__ == "__main__":
    import os
    _self_test()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography.npz")
    save_npz(out)