import os
import sys
from pathlib import Path
import cv2

def load_env_file():
    """
    .envファイルを手動で簡易的にパースしてos.environにロードします。
    (python-dotenvなどの外部ライブラリを不要にするため)
    """
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # 前後のクォーテーションを削除
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val
        print("💡 .env ファイルの設定をロードしました。")
    else:
        print("⚠️ .env ファイルが見つかりません。デフォルト値または環境変数を使用します。")

def main():
    load_env_file()
    
    # 環境変数から設定を取得
    user = os.getenv("TAPO_USER", "admin")
    password = os.getenv("TAPO_PASS", "password123")
    ip = os.getenv("TAPO_IP", "192.168.1.100")
    port = os.getenv("TAPO_PORT", "554")
    stream = os.getenv("TAPO_STREAM", "stream1")
    
    rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/{stream}"
    
    print("="*60)
    print("Tapo C200 接続テストスクリプト")
    print(f"接続先URL: rtsp://{user}:******@{ip}:{port}/{stream}")
    print("="*60)
    
    print("🎥 カメラに接続を試みています (最大で10秒ほどかかる場合があります)...")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("\n❌ エラー: カメラへの接続に失敗しました。")
        print("\n【確認チェックリスト】")
        print(f"1. IPアドレス ( {ip} ) は正しいですか？")
        print("   -> スマホのTapoアプリの「カメラ設定（歯車マーク） ➔ カメラ情報 ➔ IPアドレス」で確認できます。")
        print(f"2. ユーザー名 ( {user} ) と パスワード は正しいですか？")
        print("   -> Tapoアプリの「カメラ設定 ➔ 高度な設定 ➔ カメラのアカウント」で作成した専用アカウントである必要があります。(TapoのログインID/パスとは異なります)")
        print("3. ネットワーク接続")
        print("   -> テストを実行しているPCとTapoカメラは、同じルーター（Wi-Fi）に接続されていますか？")
        return
        
    print("\n✅ カメラへの接続に成功しました！")
    
    # バッファに溜まった古いフレームを飛ばし、最新の映像を得るために空読みします
    print("📸 テスト画像をキャプチャ中...")
    for i in range(10):
        cap.grab()
        
    ret, frame = cap.read()
    if ret:
        output_file = "tapo_test_capture.jpg"
        cv2.imwrite(output_file, frame)
        print(f"\n🎉 キャプチャ成功！画像ファイルを保存しました: {output_file}")
        print("保存された画像を開いて、カメラの映像が正しく表示されているか確認してください。")
    else:
        print("\n❌ エラー: カメラへの接続はできましたが、映像フレームの読み取りに失敗しました。")
        
    cap.release()

if __name__ == "__main__":
    main()
