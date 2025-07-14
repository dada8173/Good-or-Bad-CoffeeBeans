import os
import io
import shutil
from PIL import Image
import PySimpleGUI as sg

# 1. 選擇豆種
bean_types = ["ethiopia_washed", "kenya_natural", "honduras_natural"]
layout_select = [
    [sg.Text("請選擇要分類的咖啡豆種類：", font=("Arial", 14))],
    [sg.Combo(bean_types, key="-BEAN-", font=("Arial", 14))],
    [sg.Button("確定", font=("Arial", 14)), sg.Button("取消", font=("Arial", 14))]
]
window_select = sg.Window("選擇豆種", layout_select, modal=True)
event, values = window_select.read()
window_select.close()
if event != "確定" or not values.get("-BEAN-"):
    sg.popup("未選擇豆種，程式結束", font=("Arial", 14))
    exit()
bean_type = values["-BEAN-"]

# 根目錄為此程式所在資料夾
script_dir = os.path.dirname(os.path.abspath(__file__))
# crop 資料夾路徑
crop_dir = os.path.join(script_dir, "coffee_beans_data", bean_type, "crop")
# 固定使用 classByhands 資料夾
scheme_dir = os.path.join(crop_dir, "classByhands")
if not os.path.isdir(scheme_dir):
    sg.popup(f"未找到 classByhands 資料夾：{scheme_dir}", font=("Arial", 14))
    exit()

# 未分類資料夾在 crop 下
unclassified = os.path.join(crop_dir, "unclassified")
os.makedirs(unclassified, exist_ok=True)

# 定義主標籤順序與中文對應
master_labels = [
    ("good",     "好"),
    ("bad",      "壞"),
    ("back",     "背面"),
    ("idontknow","不知道"),
]
# 動態建立 classes dict，依序編號
classes = {}
key_counter = 1
for eng, chi in master_labels:
    folder = os.path.join(scheme_dir, eng)
    if os.path.isdir(folder):
        classes[str(key_counter)] = (chi, folder)
        key_counter += 1

# 確保分類資料夾存在
for _, path in classes.values():
    os.makedirs(path, exist_ok=True)

# 取得待分類圖片清單
imgs = [f for f in os.listdir(unclassified)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total = len(imgs)
if total == 0:
    sg.popup("未找到任何待分類的圖片", font=("Arial", 14))
    exit()

# 歷史記錄，用於「上一步」
history = []

# 計算各類數量
def get_counts():
    return {k: len([fn for fn in os.listdir(p)
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for k, (_, p) in classes.items()}

# 讀取並縮放圖片，回傳 PNG bytes
def get_image_data(path, maxsize=(800, 600)):
    img = Image.open(path)
    img.thumbnail(maxsize)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

# --- GUI 版面 ---
font = ("Arial", 14)
max_w, max_h = 800, 600
# 圖片區塊
image_col = sg.Column([[sg.Image(key='-IMAGE-', size=(max_w, max_h))]],
                      size=(max_w, max_h), element_justification='c')
# 按鈕區塊
btns = [sg.Button("上一步", key="prev", size=(8,1), font=font)]
for k, (label, _) in classes.items():
    btns.append(sg.Button(f"{label} (0)", key=k, size=(12,1), font=font))
btns.append(sg.Button("退出", size=(8,1), font=font))

layout = [
    [image_col],
    [sg.ProgressBar(total, orientation='h', size=(40, 20), key='-PROG_BAR-')],
    [sg.Text("", key='-PROGRESS-', font=font)],
    btns
]
window = sg.Window("咖啡豆分類介面", layout, element_justification='c', return_keyboard_events=True, finalize=True)

idx = 0

def update_ui():
    # 更新圖片顯示
    img_path = os.path.join(unclassified, imgs[idx])
    window['-IMAGE-'].update(data=get_image_data(img_path, maxsize=(max_w, max_h)))
    # 更新文字進度與進度條
    window['-PROGRESS-'].update(f"第 {idx+1}/{total} 張：{imgs[idx]}")
    window['-PROG_BAR-'].update_bar(idx+1)
    # 更新按鈕上的計數
    counts = get_counts()
    for k, (label, _) in classes.items():
        window[k].update(f"{label} ({counts[k]})")
    # 上一步按鈕狀態
    window['prev'].update(disabled=(len(history) == 0))

# 初始化 UI
update_ui()

# 事件迴圈
while True:
    event, _ = window.read()
    if event in (sg.WIN_CLOSED, "退出"):
        break
    # 捕捉快速鍵或按鈕
    key = None
    if event in classes:
        key = event
    elif event == 'prev':
        key = 'prev'

    # 上一步
    if key == 'prev' and history:
        last_idx, last_key = history.pop()
        fname = imgs[last_idx]
        _, dest_dir = classes[last_key]
        shutil.move(os.path.join(dest_dir, fname), os.path.join(unclassified, fname))
        idx = last_idx
        update_ui()
    # 分類
    elif key in classes:
        history.append((idx, key))
        src = os.path.join(unclassified, imgs[idx])
        dst = os.path.join(classes[key][1], imgs[idx])
        shutil.move(src, dst)
        idx += 1
        if idx >= total:
            sg.popup("分類完成！", font=font)
            break
        update_ui()

window.close()
