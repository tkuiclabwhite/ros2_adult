import cv2
import numpy as np
import os

home = os.path.expanduser("~")
folder = os.path.join(home, "ros2_adult/src/strategy/strategy/obs/RoutePlan")

image_paths = {
    45:  os.path.join(folder, 'save_1_scan_45.jpg'),
    0:   os.path.join(folder, 'save_1_scan_0.jpg'),
    -45: os.path.join(folder, 'save_1_scan_-45.jpg')
}

# 各角度設定：需手動點幾個點、點什麼、右/左側自動補角落
angle_config = {
    45:  {
        'clicks': 2,
        'desc': '邊界線【左上】、【左下】兩端點',
        'auto': 'right',   # 自動補 (320,0) 右上 和 (320,240) 右下
    },
    0:   {
        'clicks': 4,
        'desc': '【左上】、【右上】、【右下】、【左下】四個角',
        'auto': None,
    },
    -45: {
        'clicks': 2,
        'desc': '邊界線【右上】、【右下】兩端點',
        'auto': 'left',    # 自動補 (0,0) 左上 和 (0,240) 左下
    },
}

target_w, target_h = 320, 240
final_results = {}

def mouse_callback(event, x, y, flags, param):
    global img_display, clicked_points, n_clicks_needed
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= n_clicks_needed:
            return
        clicked_points.append([x, y])
        print(f"  點 {len(clicked_points)}: [{x}, {y}]")
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(clicked_points)), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Calibration", img_display)

for angle, path in image_paths.items():
    img_raw = cv2.imread(path)
    if img_raw is None:
        print(f"[警告] 找不到圖片：{path}，跳過角度 {angle}°")
        continue

    img_display = cv2.resize(img_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    clicked_points = []
    cfg = angle_config[angle]
    n_clicks_needed = cfg['clicks']

    # 先把自動補的角落畫在圖上讓使用者知道
    if cfg['auto'] == 'right':
        cv2.circle(img_display, (target_w, 0),          5, (0, 255, 0), -1)
        cv2.circle(img_display, (target_w, target_h),   5, (0, 255, 0), -1)
        cv2.putText(img_display, "auto", (target_w - 35, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(img_display, "auto", (target_w - 35, target_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    elif cfg['auto'] == 'left':
        cv2.circle(img_display, (0, 0),          5, (0, 255, 0), -1)
        cv2.circle(img_display, (0, target_h),   5, (0, 255, 0), -1)
        cv2.putText(img_display, "auto", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(img_display, "auto", (2, target_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    print(f"\n==========================================")
    print(f">>> 角度: {angle}°")
    print(f"    請點擊 {cfg['clicks']} 個點：{cfg['desc']}")
    if cfg['auto'] == 'right':
        print(f"    右側角落 (320,0) 和 (320,240) 會自動補上（綠點）")
    elif cfg['auto'] == 'left':
        print(f"    左側角落 (0,0) 和 (0,240) 會自動補上（綠點）")
    print(f"    點完後按任意鍵繼續")

    cv2.imshow("Calibration", img_display)
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.waitKey(0)

    if len(clicked_points) >= n_clicks_needed:
        pts = clicked_points[:n_clicks_needed]

        if cfg['auto'] == 'right':
            # 45°: 使用者點左上、左下；自動補右上、右下
            final_pts = [pts[0], [target_w, 0], [target_w, target_h], pts[1]]
        elif cfg['auto'] == 'left':
            # -45°: 使用者點右上、右下；自動補左上、左下
            final_pts = [[0, 0], pts[0], pts[1], [0, target_h]]
        else:
            # 0°: 直接用四個點
            final_pts = pts

        final_results[angle] = final_pts
        print(f"    完成！最終四點: {final_pts}")
    else:
        print(f"[警告] 只點了 {len(clicked_points)} 點（需要 {n_clicks_needed}），此角度跳過！")

cv2.destroyAllWindows()

print("\n=== 校正完成！複製以下內容取代 snap.py 的 roi_pts_dict ===\n")
print("        roi_pts_dict = {")
for angle, pts in final_results.items():
    print(f"            {angle}: np.array({pts}, np.int32),")
print("        }")
