import cv2
import ollama
import os

# 設定
video_path = "data/videos_split/20251027_024644_tp00027_clip_001.mp4"  # 解析したい動画のパス
save_dir = "data/extracted_frames" # 一時保存するフォルダ
os.makedirs(save_dir, exist_ok=True)

# 1. 動画ファイルの読み込み
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)      # 動画のフレームレート（1秒あたりのフレーム数）
frame_interval = int(fps * 1.0)      # 1.0秒に1回のペースで間引いて解析（ここを0.5にすれば0.5秒おき）

frame_count = 0
saved_count = 0

print("動画の解析を開始します...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 動画の終わりに達したら終了
    
    # 指定したインターバル（間隔）のフレームだけを処理
    if frame_count % frame_interval == 0:
        # 一時的に画像を保存
        img_path = os.path.join(save_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(img_path, frame)
        
        # タイムスタンプの計算（秒）
        timestamp_sec = frame_count / fps
        
        # 2. Ollama (MiniCPM) に画像を投げて判定させる
        prompt = """
        これは防犯カメラの映像のワンシーンです。
        中央の出入り口に対して、映っている人間の状態を以下の3つのいずれかで判定してください。
        理由を述べる必要はありません。必ず指定の形式のみで答えてください。

        【判定ラベル】
        - 入室
        - 退出
        - その他（誰もいない、またはただ横切っただけなど）

        【出力形式】
        ラベル: [ここに判定を記入]
        """
        
        try:
            response = ollama.chat(
                model='openbmb/minicpm-o2.6:8b',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_path]
                }]
            )
            
            result = response['message']['content'].strip()
            print(f"[{timestamp_sec:.1f}秒時点] -> {result}")
            
        except Exception as e:
            print(f"[{timestamp_sec:.1f}秒時点] エラーが発生しました: {e}")
            
        saved_count += 1

    frame_count += 1

cap.release()
print("すべてのフレームの解析が完了しました。")
