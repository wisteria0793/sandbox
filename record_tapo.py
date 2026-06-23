import os
import sys
import subprocess
import time
import shutil
import signal
from datetime import datetime, timedelta
from pathlib import Path

# ================= 設定 =================
# 環境変数から読み込むか、直接書き換えてください。
RTSP_USER = os.getenv("TAPO_USER", "admin")
RTSP_PASS = os.getenv("TAPO_PASS", "password123")
RTSP_IP = os.getenv("TAPO_IP", "192.168.1.100")
RTSP_PORT = os.getenv("TAPO_PORT", "554")
RTSP_STREAM = os.getenv("TAPO_STREAM", "stream1")  # stream1: 1080p, stream2: 360p

# 保存先ディレクトリ
SAVE_DIR = Path("./data/recordings")

# 保存期間 (日)
KEEP_DAYS = 10

# ディスク残容量の閾値 (これ以下になったら古い動画を強制削除、単位: GB)
MIN_DISK_FREE_GB = 10

# 録画の分割単位 (秒)
SEGMENT_TIME_SEC = 3600  # 3600秒 = 1時間
# =======================================

ffmpeg_process = None

def get_rtsp_url():
    """RTSP接続URLを組み立てる"""
    return f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM}"

def check_disk_space():
    """
    ディスクの空き容量を確認し、閾値を下回っている場合は古い動画を削除する
    """
    try:
        total, used, free = shutil.disk_usage(SAVE_DIR)
        free_gb = free / (2**30)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ディスク空き容量: {free_gb:.2f} GB")
        
        while free_gb < MIN_DISK_FREE_GB:
            print(f"⚠️ ディスク空き容量が閾値 ({MIN_DISK_FREE_GB} GB) を下回っています。古い録画ファイルを削除します。")
            # 最も古いmp4ファイルを検索して削除
            mp4_files = sorted(SAVE_DIR.glob("*.mp4"))
            if not mp4_files:
                print("削除できる録画ファイルがありません。")
                break
            
            oldest_file = mp4_files[0]
            try:
                oldest_file.unlink()
                print(f"🗑️ 強制削除しました: {oldest_file.name}")
            except Exception as e:
                print(f"ファイル削除エラー: {oldest_file}, {e}")
                break
                
            # 容量の再計算
            total, used, free = shutil.disk_usage(SAVE_DIR)
            free_gb = free / (2**30)
    except Exception as e:
        print(f"ディスク容量の確認中にエラーが発生しました: {e}", file=sys.stderr)

def clean_old_files():
    """
    指定日数（KEEP_DAYS）以上前の古い録画ファイルを削除する
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 古いファイルのクリーンアップチェックを開始します...")
    threshold_time = datetime.now() - timedelta(days=KEEP_DAYS)
    
    try:
        mp4_files = SAVE_DIR.glob("*.mp4")
        deleted_count = 0
        
        for file_path in mp4_files:
            # ファイルの作成日時（または更新日時）を取得
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_mtime < threshold_time:
                try:
                    file_path.unlink()
                    print(f"🗑️ 保存期限切れのため削除しました: {file_path.name} (更新日時: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                    deleted_count += 1
                except Exception as e:
                    print(f"ファイル削除エラー: {file_path}, {e}", file=sys.stderr)
        
        print(f"クリーンアップチェック完了。削除したファイル数: {deleted_count}")
    except Exception as e:
        print(f"クリーンアップ中にエラーが発生しました: {e}", file=sys.stderr)

def start_recording():
    """
    FFmpegを用いてRTSPストリームの録画を開始する
    """
    global ffmpeg_process
    
    rtsp_url = get_rtsp_url()
    
    # FFmpegコマンドの組み立て
    # -y                  : 上書き許可
    # -rtsp_transport tcp : パケットロスを防ぐためTCPで接続
    # -i [URL]            : 入力RTSPストリーム
    # -c:v copy           : 映像は再エンコードせず、そのまま保存 (CPU負荷がほぼゼロになります)
    # -an                 : 音声は完全除外 (プライバシー保護)
    # -f segment          : セグメント分割機能を使用
    # -segment_time [秒]  : 分割時間
    # -segment_format mp4 : 保存フォーマット
    # -reset_timestamps 1 : 分割時にタイムスタンプを0からリセット
    # -strftime 1         : ファイル名に日時表記を使用
    # 保存ファイル名のフォーマット: 年月日_時分秒.mp4 (例: 20260623_150000.mp4)
    output_pattern = str(SAVE_DIR / "%Y%m%d_%H%M%S.mp4")
    
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c:v", "copy",
        "-an",
        "-f", "segment",
        "-segment_time", str(SEGMENT_TIME_SEC),
        "-segment_format", "mp4",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_pattern
    ]
    
    print("="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 録画を開始します。")
    print(f"ストリーム源: rtsp://{RTSP_USER}:***@{RTSP_IP}:{RTSP_PORT}/{RTSP_STREAM}")
    print(f"保存先: {SAVE_DIR.resolve()}")
    print(f"設定: {SEGMENT_TIME_SEC // 60}分ごとに分割保存 / 音声なし / {KEEP_DAYS}日間保存")
    print("="*60)
    
    # 標準エラー（FFmpegのログ）を取得しつつバックグラウンド起動
    ffmpeg_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    return ffmpeg_process

def signal_handler(sig, frame):
    """プログラム終了シグナルを受け取った際のクリーンアップ処理"""
    global ffmpeg_process
    print("\n🛑 終了シグナルを受信しました。プロセスを終了します...")
    if ffmpeg_process and ffmpeg_process.poll() is None:
        ffmpeg_process.terminate()
        try:
            ffmpeg_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
        print("FFmpegプロセスを停止しました。")
    sys.exit(0)

# シグナルハンドラーの登録 (Ctrl+Cなどで綺麗にFFmpegも終了させる)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global ffmpeg_process
    
    # 保存先ディレクトリの作成
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 起動時に一度クリーンアップとディスクチェックを行う
    clean_old_files()
    check_disk_space()
    
    last_cleanup_time = time.time()
    
    while True:
        # 録画プロセスの開始
        process = start_recording()
        
        # FFmpegのログ（エラー）を監視するループ
        # 接続が切れた場合は、readline()が空を返すか、プロセスが終了する
        try:
            while process.poll() is None:
                # 定期的なディスク・期限切れチェック (1時間に1回実行)
                current_time = time.time()
                if current_time - last_cleanup_time > 3600:
                    clean_old_files()
                    check_disk_space()
                    last_cleanup_time = current_time
                
                # ログの読み込み（接続維持確認も兼ねる）
                # stderrをノンブロッキング的に読むか、単純にプロセス生存確認のためにスリープする
                # ここでは10秒ごとに生存確認を行う
                time.sleep(10)
                
            # プロセスが終了してしまった場合（ネットワーク切断等）
            ret_code = process.returncode
            print(f"⚠️ FFmpegプロセスが終了しました。リターンコード: {ret_code}")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}", file=sys.stderr)
            if process.poll() is None:
                process.terminate()
                
        # 接続切断などによる再起動前の待機
        print("5秒後に再起動を試みます...")
        time.sleep(5)
        
        # 再起動前にディスクの確認
        clean_old_files()
        check_disk_space()

if __name__ == "__main__":
    main()
