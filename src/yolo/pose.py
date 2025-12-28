from ultralytics import YOLO
import cv2

# ポーズ推定用のYOLOモデルをロード
model = YOLO("yolo11n-pose.pt")

# COCOモデルのキーポイント名とインデックスのマッピング
#
# 人体のキーポイント（関節）は17個あります。
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
keypoint_names = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
keypoint_indices = {name: i for i, name in enumerate(keypoint_names)}

# 座標を取得したい関節を指定
# 例えば、両肩と両肘を指定
target_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']

# ビデオファイルのパスを指定
video_path = "./data/videos/20251027_024644_tp00027.mp4"
# ビデオをキャプチャ
cap = cv2.VideoCapture(video_path)

# ビデオが開いている間ループ
while cap.isOpened():
    # フレームを1枚読み込む
    success, frame = cap.read()

    if success:
        # フレームに対してポーズ推定を実行
        results = model(frame)

        # 結果をフレームに描画
        annotated_frame = results[0].plot()

        # 検出された各人物のキーポイントを処理
        # results[0].keypoints.xy は、検出された各人物のキーポイント座標（x, y）を格納したテンソル
        for person_keypoints in results[0].keypoints.xy:
            for joint_name in target_joints:
                # 指定した関節のインデックスを取得
                if joint_name in keypoint_indices:
                    joint_index = keypoint_indices[joint_name]
                    # 関節の座標を取得 (x, y)
                    x, y = person_keypoints[joint_index]

                    # 座標が検出されている場合（0より大きい値）
                    if x > 0 and y > 0:
                        # コンソールに関節名と座標を出力
                        print(f"Joint: {joint_name}, Coordinates: ({int(x)}, {int(y)})")
                        # 指定した関節をハイライト表示（緑色の円）
                        cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # 結果フレームを表示
        cv2.imshow("YOLOv11 Pose Estimation", annotated_frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終端に達したらループを抜ける
        break

# ビデオキャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
