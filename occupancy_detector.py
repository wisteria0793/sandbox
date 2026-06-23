import os
import sys
import time
import subprocess
import csv
import json
import threading
from datetime import datetime
from collections import deque
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
from ultralytics import YOLOWorld

# ================= 設定 =================
# Tapoカメラ接続情報
RTSP_USER = os.getenv("TAPO_USER", "admin")
RTSP_PASS = os.getenv("TAPO_PASS", "password123")
RTSP_IP = os.getenv("TAPO_IP", "192.168.1.100")
RTSP_PORT = os.getenv("TAPO_PORT", "554")
RTSP_STREAM = os.getenv("TAPO_STREAM", "stream1")

# スタッフ端末のMACアドレス一覧 (環境変数から取得、小文字に統一して保持)
_staff_macs_raw = os.getenv("STAFF_MACS", "")
STAFF_MACS = [mac.strip().lower() for mac in _staff_macs_raw.split(",") if mac.strip()]
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

# Pico W 物理センサー共有データとロック
sensor_lock = threading.Lock()
latest_sensors = {
    "mw_radar": 0,
    "pir": 0,
    "last_updated": 0.0
}

class PicoSensorHandler(BaseHTTPRequestHandler):
    """
    Pico W からのセンサーデータ HTTP POST を受け取るハンドラ
    """
    def do_POST(self):
        if self.path == '/sensor':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # スレッドセーフにグローバル変数を更新
                with sensor_lock:
                    latest_sensors["mw_radar"] = int(data.get("mw_radar", 0))
                    latest_sensors["pir"] = int(data.get("pir", 0))
                    latest_sensors["last_updated"] = time.time()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # 標準のアクセスログを標準出力に垂れ流さないようにミュートします
        pass

def start_sensor_server():
    """
    Pico W からのデータ受信サーバーを起動する (別スレッドで動作させます)
    """
    try:
        server_address = ('', 8080)
        httpd = HTTPServer(server_address, PicoSensorHandler)
        print("📡 Pico W センサー受信サーバーをポート 8080 で起動しました。")
        httpd.serve_forever()
    except Exception as e:
        print(f"❌ センサー受信サーバーの起動に失敗しました: {e}", file=sys.stderr)

class RTSPStreamReader:
    """
    RTSPストリームの受信遅延（OpenCVの内部バッファ）を防ぐため、
    バックグラウンドスレッドで最速で映像を読み込み続け、
    メイン処理には常に「最新の1フレーム」だけを提供するクラス。
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.ret = False
        self.frame = None
        self.is_running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue
                
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = True
            else:
                # 接続が切れた場合は少し待って再接続
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()

def get_rtsp_url():
    """RTSP接続URLを組み立てる"""
    return f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM}"

def check_staff_presence():
    """
    arp-scanコマンドを実行し、LAN内にスタッフのMACアドレスを持つ端末がいるか確認する。
    スマホのスリープ状態を考慮し、一度検出したら猶予時間(GRACE_PERIOD)が経過するまでは
    「スタッフ滞在中」と判定する。
    """
    global last_staff_detected_time
    current_time = time.time()
    
    if not STAFF_MACS:
        return False

    is_currently_connected = False
    
    try:
        # arp-scan を実行してローカルネットワークをスキャン
        # --localnet でホスト所属のサブネット全体をスキャン
        # --retry=1 --timeout=100 で高速スキャン
        result = subprocess.run(
            ["arp-scan", "--localnet", "--retry=1", "--timeout=100"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # arp-scanの出力結果から、大文字小文字を無視して登録されたMACアドレスを探索
        output_lower = result.stdout.lower()
        for mac in STAFF_MACS:
            if mac in output_lower:
                is_currently_connected = True
                break
    except FileNotFoundError:
        print("⚠️ 警告: arp-scan コマンドが見つかりません。システムに arp-scan をインストールしてください。", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ 警告: arp-scan の実行中にエラーが発生しました: {e}", file=sys.stderr)
        
    if is_currently_connected:
        last_staff_detected_time = current_time
        return True
        
    # 現在接続が切れていても、猶予時間内であれば「スタッフ滞在中」とみなす
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
                "mw_radar", "pir", "final_occupancy", "is_staff_present"
            ])

def write_log(timestamp_str, detected_labels, raw_count, raw_occupancy, smoothed_occupancy, radar_val, pir_val, final_occupancy, is_staff):
    """CSVに判定結果を追記"""
    try:
        labels_str = ",".join(detected_labels) if detected_labels else "none"
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str, labels_str, raw_count, 
                1 if raw_occupancy else 0, 
                1 if smoothed_occupancy else 0, 
                radar_val, pir_val,
                1 if final_occupancy else 0,
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

def load_model():
    """
    YOLO-Worldモデルをロードする。
    CPU推論の高速化・省電力化のため、初回起動時にOpenVINO形式へエクスポートし、
    2回目以降は最適化されたモデルを直接ロードします。
    """
    base_model_name = "yolov8s-worldv2.pt"
    openvino_model_path = Path("yolov8s-worldv2_openvino_model")
    
    # すでにOpenVINOモデルが存在すればそれをロード
    if openvino_model_path.exists():
        print("💡 最適化された OpenVINO モデルをロードしています...")
        model = YOLOWorld(str(openvino_model_path))
    else:
        print(f"🔄 初回起動：ベースモデル {base_model_name} をロード中...")
        model = YOLOWorld(base_model_name)
        
        print("⚡ CPU推論を高速化・省電力化するため、OpenVINO形式にエクスポート中 (数分かかります)...")
        try:
            # OpenVINO形式にエクスポート (CPU向け最適化)
            # ※初回のみ時間がかかりますが、次回以降は一瞬で起動します
            model.export(format="openvino")
            print("✅ OpenVINOへの変換が完了しました。")
            
            # 変換したモデルを再ロード
            print("💡 最適化されたモデルに切り替えています...")
            model = YOLOWorld(str(openvino_model_path))
        except Exception as e:
            print(f"⚠️ OpenVINOへの変換に失敗しました。通常のベースモデルで推論を継続します。エラー: {e}")
            
    # クラスの設定
    model.set_classes(DETECT_CLASSES)
    return model

def main():
    try:
        model = load_model()
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}", file=sys.stderr)
        return

    # Pico W 受信サーバーをバックグラウンドスレッドで起動します
    server_thread = threading.Thread(target=start_sensor_server, daemon=True)
    server_thread.start()

    init_log_file()
    rtsp_url = get_rtsp_url()
    
    print("\n" + "="*60)
    print("在室判定システムを起動しました。")
    print(f"・判定間隔: {CHECK_INTERVAL_SEC}秒おき")
    print(f"・時系列バッファ: 直近 {BUFFER_SIZE} 回中 {OCCUPANCY_RATIO_THRESHOLD*100:.0f}% 以上の検知で在室判定")
    print(f"・スタッフMACアドレス: {', '.join(STAFF_MACS)} (検知猶予: {STAFF_GRACE_PERIOD_SEC//60}分)")
    print(f"・ログ保存先: {LOG_FILE.resolve()}")
    print("="*60 + "\n")
    
    reader = RTSPStreamReader(rtsp_url)
    
    last_vps_sent_status = None
    
    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            # 映像がまだ受信できていない場合は少し待つ
            time.sleep(0.5)
            continue
            
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. スタッフの滞在判定
        is_staff = check_staff_presence()
        
        # 2. Pico W 物理センサーデータの取得 (スレッドセーフ)
        # 10秒以上更新がない場合は、センサーがオフラインとみなして0扱いにします
        with sensor_lock:
            sensor_time_diff = time.time() - latest_sensors["last_updated"]
            if latest_sensors["last_updated"] > 0 and sensor_time_diff < 10.0:
                radar_val = latest_sensors["mw_radar"]
                pir_val = latest_sensors["pir"]
                sensor_status_str = f"Radar:{radar_val}, PIR:{pir_val}"
            else:
                radar_val = 0
                pir_val = 0
                sensor_status_str = "Offline"
        
        # 3. YOLOでの物体検出実行
        res = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        all_boxes = res[0].boxes
        names_dict = res[0].names
        
        # Y軸制限による誤検知の除外
        valid_boxes = filter_boxes(all_boxes, frame.shape[0])
        
        # 検出された物体のラベル取得
        detected_labels = [names_dict[int(box.cls[0].cpu().item())] for box in valid_boxes]
        
        # 生の判定結果（靴、荷物、人が1つでもあればTrue）
        raw_occupancy = len(valid_boxes) > 0
        
        # 4. 時系列バッファへ追加
        detection_history.append(raw_occupancy)
        
        # 直近の履歴の中で「検出あり」の割合を計算
        history_detect_count = sum(1 for val in detection_history if val)
        detect_ratio = history_detect_count / len(detection_history) if len(detection_history) > 0 else 0.0
        
        # 時系列バッファを考慮した最終的なカメラ在室判定
        smoothed_occupancy = detect_ratio >= OCCUPANCY_RATIO_THRESHOLD
        
        # 5. 複合判定 (センサーフュージョン)
        # カメラ、ミリ波、赤外線のいずれか1つでも「あり」を示していれば【在室】と判定する
        final_occupancy = smoothed_occupancy or (radar_val == 1) or (pir_val == 1)
        
        # 6. ログ書き込み
        write_log(
            timestamp_str=current_time_str,
            detected_labels=detected_labels,
            raw_count=len(valid_boxes),
            raw_occupancy=raw_occupancy,
            smoothed_occupancy=smoothed_occupancy,
            radar_val=radar_val,
            pir_val=pir_val,
            final_occupancy=final_occupancy,
            is_staff=is_staff
        )
        
        # 画面への進捗出力
        staff_status = "👮 スタッフ滞在中" if is_staff else "👤 ゲスト判定モード"
        status_str = "【🟢 在室中】" if final_occupancy else "【⚪ 不在】"
        camera_status_str = "あり" if smoothed_occupancy else "なし"
        print(f"[{current_time_str}] {status_str} (カメラ:{camera_status_str}, センサー:{sensor_status_str}) | {staff_status} | 検出物: {detected_labels}")
        
        # 7. VPSへのデータ送信 (状態変化、または最初の送信時に実行)
        # スタッフ滞在中であってもステータスを送信しますが、データに "is_staff" フラグを乗せます
        vps_payload = {
            "timestamp": datetime.now().isoformat(),
            "occupancy": 1 if (final_occupancy and not is_staff) else 0, # スタッフ滞在中の場合は「不在(0)」として送る
            "is_staff_present": is_staff,
            "camera_occupancy": 1 if smoothed_occupancy else 0,
            "mw_radar": radar_val,
            "pir": pir_val,
            "detected_objects": detected_labels
        }
        
        # 状態変化があった場合のみVPSに即時通知する (通信量の節約)
        current_vps_status = (vps_payload["occupancy"], is_staff)
        if current_vps_status != last_vps_sent_status:
            send_to_vps(vps_payload)
            last_vps_sent_status = current_vps_status
            
        # バックグラウンドスレッド(RTSPStreamReader)が常に最新のフレームを取得しているため、
        # メインスレッド側でのバッファフラッシュ(cap.grab)は不要になります。
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
