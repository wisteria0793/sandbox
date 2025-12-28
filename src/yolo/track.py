from ultralytics import YOLO
import cv2
import numpy as np

from collections import defaultdict

# YOLOモデルをロード
model = YOLO("yolo11n.pt")

# ビデオファイルのパスを指定
video_path = "./data/videos/20251027_024644_tp00027.mp4"
# ビデオをキャプチャ
cap = cv2.VideoCapture(video_path)

# トラック履歴を保存するための辞書
track_history = defaultdict(lambda: [])

# ビデオが開いている間ループ
while cap.isOpened():
    # フレームを1枚読み込む
    success, frame = cap.read()

    if success:
        # フレームに対してトラッキングを実行
        result = model.track(frame, persist=True)[0]

        # 結果をフレームに描画
        annotated_frame = result.plot()

        # 結果フレームを表示
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終端に達したらループを抜ける
        break

# ビデオキャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()