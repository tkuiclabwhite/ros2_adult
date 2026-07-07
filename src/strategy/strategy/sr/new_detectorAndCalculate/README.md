# 專案背景

我在做 TKU IClab 大人型人形機器人(ros2_adult, repo: tkuiclabwhite/ros2_adult)的上下板策略程式,對應小型機器人(ros2_kid)的 `strategy/strategy/lc/` 模組,但這次是全新重寫。

**檔案位置**:新程式放在 `strategy/strategy/sr/`(不是 `lc/`,那是 KID 的路徑;大人型這邊命名是 `sr`)。

**關於「不 import 舊 package」的澄清**:指的是不 import KID 那邊 `strategy.lc.*` 或其他不相關的舊程式碼,**但 `API.py` 是可以用的**——策略類別本來就是繼承 `API`(例如 `class LiftandCarry(API)` 這種寫法),所以 `sendContinuousValue`、`sendBodySector`、`sendHeadMotor`、`set_head`、`get_objects` 這些 `API.py` 既有的介面都可以直接用,不算「引入其他 package」。

**額外發現**(從 `API.py` 讀到的,對接下來很有幫助):`API.py` 裡的 `_label_image_cb` 已經在維護 `self.label_matrix`(2D numpy array,mono8 編碼,剛好對應這次要用的顏色 bitmask 格式)跟 `self.label_matrix_flatten`(一維版本)。也就是說新策略類別繼承 `API` 之後,直接用 `self.label_matrix` 餵給 `BoardDetector.detect()` 就好,不需要另外處理影像轉換或訂閱/reshape 邏輯,`API.py` 已經做好了。

## 硬體與限制

- 平台:Jetson AGX Orin
- 相機:羅技一般 USB 相機(**比賽禁止使用深度相機**,所以不能靠深度感測器拿距離)
- 大人型站得比小型機器人高,相機視角更接近俯視板子,理論上能更準確判斷板子邊緣傾斜/哪邊比較平

## 移動方式的關鍵限制(這點很重要,決定了整個架構)

機器人只有兩種步態,而且**互相不連貫**:
1. **LIPM 連續步態**:邊走邊修正方向,可以每幀持續調整
2. **單步上板/下板步態**:一旦觸發就是固定動作序列,執行中不能像連續步態一樣邊走邊修

流程是:連續步態走近板子 → 停下 → 觸發一次上板步態 → 上完後切回連續步態尋找下一塊板子。

這代表視覺系統要處理兩個性質完全不同的任務:
- **任務 A(連續步態階段)**:持續輸出距離/角度修正量,容許誤差、每幀可以修。
- **任務 B(觸發時機判斷)**:單次決策,一旦觸發沒有回頭路,所以「觸發那一刻」的量測值必須是穩定共識,不能是單幀雜訊值。這也是為什麼濾波(Kalman)在這個架構下特別關鍵。

## 舊系統(ros2_kid `lc.py` / `calculate_edge.py`)的做法與問題

舊系統的「距離」不是深度相機,而是:
- 影像先經過 HSV 顏色分類,產生一個 `label_matrix`(320x240,每個像素是顏色的 bitmask 值)
- `return_real_board()`:固定 6 個 x 座標(`FOOT` 陣列,左右腳各 3 個採樣點),從基準列 `FOOTBOARD_LINE` 往上逐格掃描,找到連續 10 個像素都符合板子顏色的位置,回傳「基準列 - 該位置」當作距離。本質上是簡化版 IPM(Inverse Perspective Mapping):利用相機角度/高度固定,影像中的列(y)理論上對應固定的地面實際距離,所以**不需要知道板子實際尺寸**就能算距離,對「板子大小不一」本身有一定抵抗力。
- `calculate_edge.py` 額外用 Canny + HoughLinesP 抓一條邊緣線算斜率,拿來修正機器人朝向(theta)。

**已知問題**:
1. 斜率很容易亂跳——因為只信任「單一條」Hough 偵測到的線段,邊緣點抓不夠好、光線雜訊、板子邊緣本身不規則,都會讓這條線整個跳掉。
2. 舊系統用固定 sector/layer 參數表卡死每一層動作,沒有「動態路徑規劃」的空間,決策邏輯跟視覺量測黏死在同一個巨大 if-elif 狀態機裡。
3. 板子的「厚度」在舊架構下沒有真正處理(`LCParameter` ini 檔裡其實已經有 `board_high` 這個已知參數,例如 UpStair.ini board_high=3.0、DownStair.ini board_high=1.0,但目前只餵給步態、沒有回饋進視覺距離計算)。板子墊高後,pixel-row→實際距離的對應關係其實會跟著板高改變,這是舊系統距離漂移的真正原因,而不是長寬形狀不一。

## 這次重新設計的架構決定

視覺/決策要分三層,互相不依賴,各自可以獨立測試:

```
board_detector.py   → 只做視覺幾何量測
edge_fit.py 內的濾波 → 時序穩定性(Kalman filter,尚未開始)
path_planner.py      → 決策層,吃量測值輸出 forward/theta/translation(尚未開始)
```

## 目前狀態的澄清

`board_detector.py` 先前雖然已經寫出一版初稿(多欄掃描 + RANSAC,附在對話中),但**視為還沒定案**,需要依照下面「舊系統可重用常數」跟「待確認細節」調整過一輪才算數,不要直接照抄那份初稿的預設值。

草稿版設計重點(供參考,實際數字待確認後調整):
1. **多欄掃描取代 6 點**:比照舊系統的逐欄掃描精神,但欄數從 6 拉高到可調(建議 20~30),欄數越多,後續擬合越穩健。
2. **RANSAC 穩健線性擬合取代單一 Hough 線**:對每欄找到的邊緣點集合 `(x, y)` 做 RANSAC(自己寫的簡易版,純 numpy,不依賴外部套件),自動排除離群點,斜率是「整條邊的共識」而不是單一線段的巧合。
3. **同時抓近邊與遠邊,算出板面深度**:近邊(板子開始的位置)+ 遠邊(板子顏色結束的位置)相減 = 板面深度(pixel),用來判斷後腳落板空間夠不夠、要不要拆成兩步上板。
4. **左右分邊擬合,比殘差判斷哪邊比較平**:把點集合依 x 中位數切兩半分別擬合,RMSE 較小的一側代表邊緣偵測品質較好、板面較平整規則,可以優先信任那一側做角度修正或腳步對齊基準。
5. **`confidence` 欄位**:有效欄位比例,值太低時決策層應該直接跳過這幀、沿用上一幀,而不是拿雜訊結果去觸發步態。

草稿版介面(`detect()` 回傳,待確認後可能調整):
```python
{
    "confidence": float,        # 0~1,這幀量測可信度
    "near_slope": float,        # RANSAC 擬合後的近邊斜率
    "near_intercept": float,
    "near_fit_valid": bool,     # inlier 不足時 False,決策層應忽略這幀
    "flatter_side": "left"/"right"/"unknown",
    "board_depth_px": float,    # 近邊到遠邊的板面深度(pixel)
    "near_points": [(x,y),...],
    "far_points": [(x,y),...],
}
```

已經跑過合成資料的自我測試(假造一塊有已知斜率跟深度的 label_matrix),斜率跟深度數值都對得上預期值,但這只驗證了演算法邏輯本身沒錯,不代表常數/介面已經適合大人型實際情況。

## 舊系統(ros2_kid `lc.py` / `calculate_edge.py`)可重用的常數,分兩層列

**視覺層(`board_detector.py` 會直接用到)**
- 影像/label_matrix 尺寸:KID 是 320x240(`calculate_edge.py` 裡的 `LENTH`/`WIDTH`),大人型解析度待確認
- `FOOTBOARD_LINE = 215`:掃描基準列(對應新程式的 `outset`)
- `FOOT = [93,116,136,165,190,220]`:左右腳各 3 個固定 x 座標採樣點(左腳:左中右,右腳:左中右)
- 連續確認像素數:舊系統 `for i in range(1,11)`,也就是連續 10 個像素同色才算確認邊界
- `COLOR_DICT` / `COLOR_PARAMETER`(等同 KID `ObjectInfo.color_dict` / `parameter`):顏色 bitmask 對照表,直接複製沿用
- `BOARD_COLOR` 陣列的概念:每一層板子顏色是比賽現場才決定,不能寫死順序,必須是外部可傳入的參數,不能寫死在偵測層

**決策層(之後 `path_planner.py` 會用到,先列著避免漏掉)**
- `GO_UP_DISTANCE=20` / `GO_DOWN_DISTANCE=3`:觸發上/下板步態的距離門檻
- `UP_WARNING_DISTANCE=6` / `DOWN_WARNING_DISTANCE=0`:危險距離
- `FIRST/SECOND/THIRD_FORWORD_CHANGE_LINE`(50/100/150):依距離切換前進速度檔位
- `SLOPE_MIN/NORMAL/BIG`(2/5/12):斜率修正的檔位門檻
- `THETA_MIN/NORMAL/BIG`、`TRANSLATION_MIN/NORMAL/BIG`、`FORWARD_MIN/NORMAL/BIG/SUPER`、`BACK_MIN/NORMAL`:各種修正量的檔位值
- `LEFT_THETA=1` / `RIGHT_THETA=-1`、`FORWARD_PARAM=1` / `BACK_PARAM=-1`:方向基礎參數
- `START_LAYER=1`:起始層數
- `board_high`(LCParameter/ini 裡已有的板高參數,例如 UpStair board_high=3.0、DownStair board_high=1.0)→ 還沒接上視覺計算,是 IPM 校準要用的關鍵輸入

## 動手寫之前,需要跟使用者確認、不能自行假設的細節

1. **大人型的相機解析度**是多少?還是也跟 KID 一樣 320x240?(會影響 `ScanConfig` 預設值跟 min_run 要不要按比例調整)
2. **`FOOT` 六個固定 x 座標跟 `FOOTBOARD_LINE` 基準列**,是否已經實際量測過大人型的鏡頭視角?還是先留成參數,等實機架好再現場校正數字?
3. **架構問題(最關鍵)**:舊系統 `self.distance[0]`~`[5]` 是左右腳 6 個採樣點的獨立距離值,後面決策邏輯大量用「左腳三點 vs 右腳三點」分開判斷。新的 `board_detector` 除了整條邊的斜率/深度/信心值之外,**要不要保留「查詢某個特定 x 座標(某隻腳位置)的近邊/遠邊距離」這個功能**,讓決策層還能像舊系統一樣逐腳判斷?還是這次決策邏輯改用整條線的斜率去算,不需要逐腳獨立數值?
4. 大人型這次比賽**總共有幾層板子、上下板各幾層**?跟 KID 一樣是上 3 層下 3 層的結構嗎?會影響 `BOARD_COLOR` 之類陣列要留多長。
5. **min_run(連續多少像素才算確認邊界)**——如果解析度跟 KID 不同,這個閾值要不要跟著等比例調整,還是有自己想抓的容錯值?

## 接下來要做,還沒完成的部分(依優先順序)

時間有限,已經確認的優先順序:

1. **`board_detector.py` 多欄掃描 + RANSAC**——草稿已有,還需要:
   - 依上面「動手寫之前需要確認的細節」把常數/介面對齊大人型實際情況
   - 確認實際 `label_matrix` 的資料格式(`API.py` 已經提供 2D array,格式應該吻合,但要實測驗證)
   - 依大人型實際攝像頭 ROI 校正 `x_min`/`x_max`/`outset`/`y_top`
   - 拿實際影像驗證 `near_slope` 的正負號跟原本 `theta` 修正方向是否一致

2. **Homography IPM 校準**:用 `cv2.findHomography`,實地量幾個已知距離點 + 對應影像座標校準一次,把 pixel 距離換算成實際公分,不再是經驗式的像素數硬編碼。要利用已知的 `board_high` 參數,對不同板高做校正(不同板高等於地平面被墊高,需要不同的校準曲線或修正公式),藉此解決「板厚不一導致距離漂移」的問題。

3. **Kalman Filter(1D,狀態=[距離,速度] 或 [斜率,角速度])**:取代舊系統註解掉的 EMA 低通濾波。理由:EMA 的權重是固定常數,不管雜不雜訊都一樣;Kalman 對量測雜訊會動態調整信任權重,量測穩就跟得快,量測亂就多信預測,而且有「速度」狀態,對機器人等速前進的情境預測更準、延遲更小。這是「任務 B(觸發時機判斷)」穩定性的關鍵——決策層應該改成「Kalman 估計值連續 N 幀都小於門檻」才觸發上板步態,而不是單幀 `< threshold` 就觸發。

4. **板深(`board_depth_px`)換算成實際公分,決定要不要拆兩步上板**:已經有 pixel 深度,套上第 2 點的 IPM 校準後,可以跟機器人腳長/步距參數比較,決定:深度夠 → 一大步上板;深度不夠 → 先上前腳站穩再收後腳的兩段式動作。

5. **善用 Jetson AGX Orin 算力(`cv2.cuda`)**:硬體是 AGX Orin,算力不是瓶頸,可以用 OpenCV 的 CUDA 版本(`cv2.cuda_GpuMat` + CUDA 版的 HSV 轉換/中值濾波/Canny)把處理丟到 GPU 上跑,好處是可以把掃描欄數從現在的 20~30 拉更高(甚至 50+)、ROI 也可以開更大解析度,而不會拖慢整體 frame rate。這塊優先度排在功能性項目(IPM、Kalman)後面,但只要前面架構做完、時間允許就可以直接套用,改動量不大。

暫時跳過(時間不夠):訓練分割網路取代 HSV 顏色分類,先用 HSV + RANSAC 撐過這次比賽。

## 想請 Claude Code 協助的事

先跟使用者確認上面「動手寫之前需要確認的細節」那 5 個問題,再依實際回答重新調整/定案 `board_detector.py`。定案後依序往下做:homography IPM 校準模組、Kalman filter 濾波層,最後把這些串進一個獨立的 `path_planner.py`(決策層),輸出 forward/theta/translation 給連續步態,以及「是否/何時觸發上板步態」的判斷邏輯給離散步態。