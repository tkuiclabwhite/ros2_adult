"""
adult_size_urdf 右臂 (7-DOF) 反向運動學
- 從 URDF 自動解析 joint origin / axis（不手填，避開 SolidWorks 鏡像匯出的軸號陷阱）
- 正向運動學 (FK) + Damped Least Squares (DLS) 數值反解
- 6D pose 目標，姿態權重可調 (pos_only 模式 = 退化成純位置)
- 純 NumPy，零相依，可直接放進比賽主迴圈

左臂用法：把 CHAIN 換成 L 開頭的關節名即可。
"""

import numpy as np
import xml.etree.ElementTree as ET


# ---------- 基礎幾何工具 ----------
def rpy_to_R(rpy):
    """URDF 的 rpy = 固定軸 X->Y->Z (即 R = Rz @ Ry @ Rx)"""
    r, p, y = rpy
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_to_R(axis, theta):
    """繞單位軸 axis 轉 theta 弧度 (Rodrigues)"""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------- 從 URDF 解析一條手臂鏈 ----------
class ArmChain:
    def __init__(self, urdf_path, joint_names):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        all_j = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = np.array([float(x) for x in o.get('xyz').split()])
            rpy = [float(x) for x in o.get('rpy').split()]
            ax = np.array([float(x) for x in j.find('axis').get('xyz').split()])
            all_j[j.get('name')] = (xyz, rpy, ax)

        self.names = joint_names
        self.origin_T = []   # 每個 joint 的固定 origin transform (parent->joint frame, theta=0)
        self.axes = []       # 每個 joint 的旋轉軸 (在自己 frame 下)
        for n in joint_names:
            xyz, rpy, ax = all_j[n]
            self.origin_T.append(make_T(rpy_to_R(rpy), xyz))
            self.axes.append(ax / (np.linalg.norm(ax) + 1e-12))
        self.origin_T = np.array(self.origin_T)
        self.axes = np.array(self.axes)
        self.n = len(joint_names)

    def fk(self, q, ee_offset=None):
        """正向運動學。回傳每個關節 frame 的世界 transform list + 末端 T。
        ee_offset: 末端再外加一個 4x4（手掌中心/TCP），可為 None。"""
        T = np.eye(4)
        frames = []
        for i in range(self.n):
            Ti = self.origin_T[i] @ make_T(axis_angle_to_R(self.axes[i], q[i]), np.zeros(3))
            T = T @ Ti
            frames.append(T.copy())
        if ee_offset is not None:
            T = T @ ee_offset
        return frames, T

    def jacobian(self, q, ee_offset=None):
        """幾何 Jacobian (6 x n)，世界座標系下。"""
        frames, T_ee = self.fk(q, ee_offset)
        p_ee = T_ee[:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            Ti = frames[i]
            # 旋轉軸在世界座標
            z_i = Ti[:3, :3] @ self.axes[i]
            p_i = Ti[:3, 3]
            J[:3, i] = np.cross(z_i, p_ee - p_i)  # 線速度貢獻
            J[3:, i] = z_i                        # 角速度貢獻
        return J, T_ee


# ---------- 姿態誤差 ----------
def rotation_error(R_cur, R_des):
    """回傳把 R_cur 轉到 R_des 的旋轉向量 (世界座標, 3,)"""
    Re = R_des @ R_cur.T
    cos_t = np.clip((np.trace(Re) - 1) / 2, -1, 1)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    w = np.array([Re[2, 1] - Re[1, 2],
                  Re[0, 2] - Re[2, 0],
                  Re[1, 0] - Re[0, 1]]) / (2 * np.sin(theta))
    return w * theta

# ---------- DLS 數值 IK ----------
def solve_ik(chain, T_target, q_init=None,
             ee_offset=None,
             ori_weight=1.0,      # 0 => 純位置；1 => 完整 6D
             damping=0.05,
             max_iters=200,
             pos_tol=1e-4, ori_tol=1e-3,
             step_clip=0.3,
             q_limits=None):       # (n,2) array 或 None
    """
    回傳 (q, info)。info 含 success / pos_err / ori_err / iters。
    T_target: 4x4 目標末端 pose (世界座標)。
    """
    n = chain.n
    q = np.zeros(n) if q_init is None else np.array(q_init, dtype=float)
    p_des = T_target[:3, 3]
    R_des = T_target[:3, :3]

    W = np.diag([1, 1, 1, ori_weight, ori_weight, ori_weight])
    info = {}
    for it in range(max_iters):
        J, T_ee = chain.jacobian(q, ee_offset)
        p_err = p_des - T_ee[:3, 3]
        o_err = rotation_error(T_ee[:3, :3], R_des) * ori_weight
        err = np.concatenate([p_err, o_err])

        pos_e = np.linalg.norm(p_err)
        ori_e = np.linalg.norm(o_err) / max(ori_weight, 1e-9)
        if pos_e < pos_tol and (ori_weight < 1e-9 or ori_e < ori_tol):
            info = dict(success=True, pos_err=pos_e, ori_err=ori_e, iters=it)
            return q, info

        Jw = W @ J
        # DLS: dq = J^T (J J^T + λ²I)^-1 e
        JJt = Jw @ Jw.T + (damping ** 2) * np.eye(6)
        dq = Jw.T @ np.linalg.solve(JJt, W @ err)

        # step 限制，避免大跳
        norm = np.linalg.norm(dq)
        if norm > step_clip:
            dq *= step_clip / norm
        q = q + dq

        if q_limits is not None:
            q = np.clip(q, q_limits[:, 0], q_limits[:, 1])

    J, T_ee = chain.jacobian(q, ee_offset)
    pos_e = np.linalg.norm(p_des - T_ee[:3, 3])
    ori_e = np.linalg.norm(rotation_error(T_ee[:3, :3], R_des))
    info = dict(success=False, pos_err=pos_e, ori_err=ori_e, iters=max_iters)
    return q, info

# ---------- 自我驗證 ----------
if __name__ == "__main__":
    URDF = "adult_size_urdf.urdf"
    CHAIN = ['RshouderJoint', 'RarmJoint', 'Rarm1Joint', 'RforarmJoint',
             'Rforarm1Joint', 'Rforarm2Joint', 'RhandJoint']
    arm = ArmChain(URDF, CHAIN)

    rng = np.random.default_rng(0)
    print("== round-trip 測試：隨機 q -> FK -> IK 是否回到同一個末端 pose ==")
    ok = 0
    for trial in range(20):
        q_true = rng.uniform(-1.0, 1.0, arm.n)
        _, T_goal = arm.fk(q_true)
        q_sol, info = solve_ik(arm, T_goal, q_init=np.zeros(arm.n), ori_weight=1.0)
        if info['success']:
            ok += 1
        if trial < 5:
            print(f"trial {trial}: success={info['success']} "
                  f"pos_err={info['pos_err']:.2e} ori_err={info['ori_err']:.2e} "
                  f"iters={info['iters']}")
    print(f"6D 收斂: {ok}/20")

    print("\n== 純位置模式 (ori_weight=0) ==")
    q_true = rng.uniform(-1, 1, arm.n)
    _, T_goal = arm.fk(q_true)
    q_sol, info = solve_ik(arm, T_goal, ori_weight=0.0)
    print(f"success={info['success']} pos_err={info['pos_err']:.2e} iters={info['iters']}")

    print("\n== 範例：指定一個世界座標目標點 ==")
    _, T0 = arm.fk(np.zeros(arm.n))
    print("zero-pose 末端位置:", np.round(T0[:3, 3], 4))
    T_t = T0.copy()
    T_t[:3, 3] += np.array([0.03, -0.02, 0.04])
    q_sol, info = solve_ik(arm, T_t, ori_weight=1.0)
    print(f"目標位置 {np.round(T_t[:3,3],4)} -> success={info['success']} "
          f"pos_err={info['pos_err']:.2e}")
    print("解出關節角(rad):", np.round(q_sol, 4))
