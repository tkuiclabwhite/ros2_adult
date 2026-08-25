# 步長指令。x/y/z/theta 用 float64 對齊 walking_node 內部的浮點運算，
# 原本的 int32 會讓小數步長在傳輸時被截掉。
float64 x
float64 y
float64 z
float64 theta
int32 walking_mode
int32 walking_state
int32 sensor_mode
