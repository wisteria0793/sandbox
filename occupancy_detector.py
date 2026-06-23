import os
import sys
import time
import subprocess
import csv
from datetime import datetime
from collections import deque
from pathlib import Path
import cv2
from ultralytics import YOLOWorld

# ================= 設定 =================
# Tapoカメラ接続情報
RTSP_USER = os.getenv("TAPO_USER", "admin")
RTSP_PASS = os.getenv("TAPO_PASS", "password123")
RTSP_IP = os.getenv("TAPO_IP", "192.168.1.100")
RTSP_PORT = os.getenv("TAPO_PORT", "554")
RTSP_STREAM = os.getenv("TAPO_STREAM", "stream1")

# スタッフ端末の固定IPアドレス一覧
STAFF_IPS = ["192.168.1.150"]
# スタッフのWi-Fi接続検知後の猶予時間 (秒) (スマホのスリープ対策: 15分)
STAFF_GRACE_PERIOD_SEC = 900

# 画像判定を行う間隔 (秒)
CHECK_INTERVAL_SEC = 2.0

# YOLO-Worldのクラス設定（細分化した靴の種類と荷物・人）
DETECT_CLASSES = [
    "sneakers", "leather shoes", "sandals", "slippers", "boots", "shoes", "footwear",
    "suitcase", "bag", "backpack", "umbrella", "person"
]
CONFIDENCE_THRESHOLD = 0.25
Y_MIN_LIMIT = 250  # 判定対象とする最小のY座標 (スリッパラックなどの除外用)

# 時系列バッファの設定 (直近15回の判定履歴を保持: 2秒おきなら計30秒分)
BUFFER_SIZE = 15
# 履歴の中で検出（靴や荷物など）があった割合が何割以上で「在室」とするか (例: 20%以上)
OCCUPANCY_RATIO_THRESHOLD = 0.20

# ログ保存先
LOG_DIR = Path("./data")
LOG_FILE = LOG_DIR / "occupancy_log.csv"

# VPS送信先URL (設定された場合のみ送信を実行します)
VPS_API_URL = os.getenv("VPS_API_URL", "")
# =======================================

# 状態管理用グローバル変数
detection_history = deque(maxlen=BUFFER_SIZE)
last_staff_detected_time = 0.0

def get_rtsp_url():
    """RTSP接続URLを組み立てる"""
    return f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM}"

def check_staff_presence():
    """
    指定されたスタッフ端末のIPに対してpingを打ち、接続を確認する。
    スマホのスリープ状態を考慮し、一度検出したら猶予時間(GRACE_PERIOD)が経過するまでは
    「スタッフ滞在中」と判定する。
    """
    global last_staff_detected_time
    current_time = time.time()
    
    # pingで現在ネットワーク上にいるか確認
    is_currently_connected = False
    for ip in STAFF_IPS:
        # pingを実行 (Windows/Mac/Linux共通で動くように -c/-n オプションを指定)
        # タイムアウト1秒、出力は捨てる
        param = "-n" if sys.platform.lower() == "win32" else "-c"
        response = os.system(f"ping {param} 1 -W 1 {ip} > /dev/null 2>&1")
        if response == 0:
            is_currently_connected = True
            break
            
    if is_currently_connected:
        last_staff_detected_time = current_time
        return True
        
    # 現在接続が切れていても、猶予時間内であれば「スタッフ滞表中」とみなす
    time_since_last_detect = current_time - last_staff_detected_time
    if time_since_last_detect < STAFF_GRACE_PERIOD_SEC:
        return True
        
    return False

def filter_boxes(boxes, img_height):
    """
    検出されたボックスのうち、Y軸制限フィルタ（画面下部のみを判定）を満たすものを抽出
    """
    valid_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cy = (y1 + y2) / 2
        if cy >= Y_MIN_LIMIT:
            valid_boxes.append(box)
    return valid_boxes

def init_log_file():
    """CSVログファイルを初期化（ヘッダー作成）"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "detected_objects", "raw_detect_count", 
                "camera_occupancy_raw", "camera_occupancy_smoothed", 
                "is_staff_present"
            ])

def write_log(timestamp_str, detected_labels, raw_count, raw_occupancy, smoothed_occupancy, is_staff):
    """CSVに判定結果を追記"""
    try:
        labels_str = ",".join(detected_labels) if detected_labels else "none"
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str, labels_str, raw_count, 
                1 if raw_occupancy else 0, 
                1 if smoothed_occupancy else 0, 
                1 if is_staff else 0
            ])
    except Exception as e:
        print(f"ログ書き込みエラー: {e}", file=sys.stderr)

def send_to_vps(data):
    """
    VPSへ判定データを送信する（送信先が設定されている場合のみ）
    """
    if not VPS_API_URL:
        return
        
    import requests
    try:
        response = requests.post(VPS_API_URL, json=data, timeout=5)
        if response.status_code == 200:
            print("🚀 VPSへのデータ送信に成功しました。")
        else:
            print(f"⚠️ VPSデータ送信失敗 (ステータスコード: {response.status_code})")
    except Exception as e:
        print(f"VPS送信エラー: {e}", file=sys.stderr)

def main():
    print("YOLO-Worldモデルを初期化しています...")
    try:
        model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(DETECT_CLASSES)
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}", file=sys.stderr)
        return

    init_log_file()
    rtsp_url = get_rtsp_url()
    
    print("\n" + "="*60)
    print("在室判定システムを起動しました。")
    print(f"・判定間隔: {CHECK_INTERVAL_SEC}秒おき")
    print(f"・時系列バッファ: 直近 {BUFFER_SIZE} 回中 {OCCUPANCY_RATIO_THRESHOLD*100:.0f}% 以上の検知で在室判定")
    print(f"・スタッフIP: {', '.join(STAFF_IPS)} (検知猶予: {STAFF_GRACE_PERIOD_SEC//60}分)")
    print(f"・ログ保存先: {LOG_FILE.resolve()}")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    last_vps_sent_status = None
    
    while True:
        if not cap.isOpened():
            print("⚠️ カメラへの接続がオフラインです。再接続を試みます...")
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(5)
            continue
            
        ret, frame = cap.read()
        if not ret:
            print("⚠️ フレームを取得できませんでした。ストリームを再起動します...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(2)
            continue
            
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. スタッフの滞在判定
        is_staff = check_staff_presence()
        
        # 2. YOLOでの物体検出実行
        res = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        all_boxes = res[0].boxes
        names_dict = res[0].names
        
        # Y軸制限による誤検知の除外
        valid_boxes = filter_boxes(all_boxes, frame.shape[0])
        
        # 検出された物体のラベル取得
        detected_labels = [names_dict[int(box.cls[0].cpu().item())] for box in valid_boxes]
        
        # 生の判定結果（靴、荷物、人が1つでもあればTrue）
        raw_occupancy = len(valid_boxes) > 0
        
        # 3. 時系列バッファへ追加
        detection_history.append(raw_occupancy)
        
        # 直近の履歴の中で「検出あり」の割合を計算
        history_detect_count = sum(1 for val in detection_history if val)
        detect_ratio = history_detect_count / len(detection_history) if len(detection_history) > 0 else 0.0
        
        # 時系列バッファを考慮した最終的なカメラ在室判定
        smoothed_occupancy = detect_ratio >= OCCUPANCY_RATIO_THRESHOLD
        
        # 4. ログ書き込み
        write_log(
            timestamp_str=current_time_str,
            detected_labels=detected_labels,
            raw_count=len(valid_boxes),
            raw_occupancy=raw_occupancy,
            smoothed_occupancy=smoothed_occupancy,
            is_staff=is_staff
        )
        
        # 画面への進捗出力
        staff_status = "👮 スタッフ滞在中" if is_staff else "👤 ゲスト判定モード"
        status_str = "【🟢 在室中】" if smoothed_occupancy else "【⚪ 不在】"
        print(f"[{current_time_str}] {status_str} (生判定:{1 if raw_occupancy else 0}, 履歴割合:{detect_ratio:.2f}) | {staff_status} | 検出物: {detected_labels}")
        
        # 5. VPSへのデータ送信 (状態変化、または最初の送信時に実行)
        # スタッフ滞在中であってもステータスを送信しますが、データに "is_staff" フラグを乗せます
        vps_payload = {
            "timestamp": datetime.now().isoformat(),
            "occupancy": 1 if (smoothed_occupancy and not is_staff) else 0, # スタッフ滞在中の場合は「不在(0)」として送る、もしくは別途ステータス管理
            "is_staff_present": is_staff,
            "raw_occupancy": raw_occupancy,
            "detected_objects": detected_labels
        }
        
        # 状態変化があった場合のみVPSに即時通知する (通信量の節約)
        current_vps_status = (vps_payload["occupancy"], is_staff)
        if current_vps_status != last_vps_sent_status:
            send_to_vps(vps_payload)
            last_vps_sent_status = current_vps_status
            
        # 次の判定までスリープ（Tapoカメラのストリームから最新フレームを得るため、バッファをクリア）
        # OpenCVのVideoCaptureはバックグラウンドでフレームをバッファするため、
        # 単純なsleepだと「過去のフレーム」を処理してしまう問題があります。
        # そのため、スリープ時間分フレームを空読みするか、接続を都度取り直すか、バッファサイズを1にする必要があります。
        # ここでは一番簡単な「都度最新フレームまで読み飛ばす」か「都度スリープ」を制御します。
        
        # 最新のフレームに追いつくために、バッファをフラッシュ
        for _ in range(5):
            cap.grab()
            
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
