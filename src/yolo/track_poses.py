import cv2
from ultralytics import YOLO
from pathlib import Path
import logging
import csv
from tqdm import tqdm

# --- 設定 ---
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 入力ディレクトリ: 処理対象の動画クリップが保存されているフォルダ
INPUT_DIR = Path("./data/videos_split")
# 出力ディレクトリ: 骨格追跡後の動画を保存するフォルダ
VIDEO_OUTPUT_DIR = Path("./data/poses_tracked")
# 出力ディレクトリ: キーポイント座標を保存するCSVフォルダ
CSV_OUTPUT_DIR = Path("./data/keypoints_csv")

# 使用するポーズ推定モデル
MODEL_NAME = "yolov8n-pose.pt" # yolo11n-pose.pt もしくは yolov8n-pose.pt

# 描画色の設定 (BGR形式)
BOX_COLOR = (255, 255, 0)   # シアン (バウンディングボックス)
ID_COLOR = (255, 255, 255) # 白 (追跡ID)
KPT_COLOR = (0, 0, 255)    # 赤 (キーポイント)
LINE_COLOR = (0, 255, 0)   # 緑 (骨格線)

# COCOのキーポイント名（yolov8-poseモデルの出力順）
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
# --- 設定ここまで ---


def draw_custom_annotations(frame, result):
    """手動でカスタマイズした色で描画を行う (修正版)"""
    # このスケルトン定義はモデルのメタデータから取得するのがより堅牢
    # ここでは可視化のために一般的なものを定義 (COCOフォーマットの17点)
    # ultralyticsのplot()が使用するボーンの定義に準拠
    skeleton_lines = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),  # 足
        (6, 12), (7, 13), (6, 7),  # 体幹
        (6, 8), (7, 9), (8, 10), (9, 11),  # 腕
        (2, 3), (1, 2), (1, 3),  # 顔 (目)
        (2, 4), (3, 5),  # 顔 (耳)
        (4, 6), (5, 7) # 耳から肩
    ]

    # トラッキング結果がない、またはIDがない場合は描画しない
    if result.boxes is None or result.boxes.id is None or result.keypoints is None:
        return frame.copy()

    # numpy配列としてデータを取得
    track_ids = result.boxes.id.int().cpu().tolist()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    keypoints_xy = result.keypoints.xy.cpu().numpy() # 各人物のキーポイント座標 (N, K, 2)

    current_frame = frame.copy()

    # 各検出人物に対して描画
    for i in range(len(track_ids)):
        track_id = track_ids[i]
        box = boxes_xyxy[i]
        kpts = keypoints_xy[i] # この人物のキーポイント (K, 2)

        x1, y1, x2, y2 = map(int, box)

        # バウンディングボックスを描画
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # 追跡IDを描画
        cv2.putText(current_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ID_COLOR, 2, cv2.LINE_AA)
        
        # キーポイント（関節）を描画
        for kpt in kpts:
            x, y = map(int, kpt)
            if x > 0 and y > 0: # 有効な座標のみ描画
                cv2.circle(current_frame, (x, y), 3, KPT_COLOR, -1) # 半径3の塗りつぶし円

        # スケルトン（骨格線）を描画
        for bone in skeleton_lines:
            kp1_idx, kp2_idx = bone[0]-1, bone[1]-1 # インデックスは0始まりに調整
            if kp1_idx < len(kpts) and kp2_idx < len(kpts):
                pt1_x, pt1_y = map(int, kpts[kp1_idx])
                pt2_x, pt2_y = map(int, kpts[kp2_idx])
                
                # 両方のキーポイントが有効な場合のみ描画
                if pt1_x > 0 and pt1_y > 0 and pt2_x > 0 and pt2_y > 0:
                    cv2.line(current_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), LINE_COLOR, 2, cv2.LINE_AA)
                    
    return current_frame


def process_one_video(input_path: Path, video_output_path: Path, csv_output_path: Path, model: YOLO):
    """
    単一の動画ファイルを処理し、追跡ビデオとキーポイントCSVを同時に生成する。
    """
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logging.error(f"動画ファイルを開けませんでした: {input_path.name}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # VideoWriterの初期化をframe_width, frame_heightが確実に整数であることを確認してから行う
        if frame_width == 0 or frame_height == 0 or fps == 0:
            logging.error(f"動画 {input_path.name} のプロパティ取得に失敗したか、不正な値です (W:{frame_width}, H:{frame_height}, FPS:{fps})")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            logging.error(f"出力ビデオファイル {video_output_path.name} を開けませんでした。コーデックまたはパスを確認してください。")
            return

        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['frame_id', 'track_id', 'keypoint_id', 'keypoint_name', 'x', 'y'])

            frame_pbar = tqdm(range(total_frames), desc=f"処理中: {input_path.name}", unit="フレーム", leave=False)
            
            for frame_idx in frame_pbar:
                ret, frame = cap.read()
                if not ret:
                    break

                # フレームに対して骨格のトラッキングを実行
                # persist=True でトラッキングを継続
                # verbose=False でultralyticsのログを抑制
                results = model.track(frame, persist=True, verbose=False)
                
                # CSVデータ書き込み
                # 結果が存在し、かつトラッキングIDが割り当てられている場合のみ処理
                if results and results[0].keypoints and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    keypoints_xy = results[0].keypoints.xy.cpu().numpy() # N人のK個のキーポイント (x,y)
                    
                    for i, track_id in enumerate(track_ids):
                        person_keypoints = keypoints_xy[i] # この人物のK個のキーポイント
                        for j, kpt_coords in enumerate(person_keypoints):
                            # x, y はfloatなので、必要に応じて丸める
                            x, y = kpt_coords
                            row = [frame_idx, track_id, j, KEYPOINT_NAMES[j], float(x), float(y)]
                            csv_writer.writerow(row)

                # カスタム描画
                # results[0]が存在する場合のみ描画を試みる
                if results and len(results[0]) > 0: # Ensure there's at least one detection
                    annotated_frame = draw_custom_annotations(frame.copy(), results[0])
                else:
                    annotated_frame = frame.copy() # 検出がない場合はそのままのフレーム
                
                video_writer.write(annotated_frame)

        cap.release()
        video_writer.release()

    except Exception as e:
        logging.error(f"動画 {input_path.name} の処理中にエラーが発生しました: {e}", exc_info=True)


def main():
    """メイン関数"""
    if not INPUT_DIR.exists():
        logging.error(f"入力ディレクトリが見つかりません: {INPUT_DIR}")
        logging.error("先に 'split_videos_by_activity.py' を実行して、動画クリップを生成してください。")
        return

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"モデルをロードしています: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        logging.error("モデル名を確認するか、インターネット接続を確認してください。")
        return

    video_files = sorted(list(INPUT_DIR.glob("*.mp4")))
    if not video_files:
        logging.warning(f"入力ディレクトリに処理対象の動画が見つかりません: {INPUT_DIR}")
        return

    logging.info(f"合計 {len(video_files)} 個の動画クリップを処理します。")

    # tqdmで動画ファイル全体の進捗バーを表示
    for video_path in tqdm(video_files, desc="全体の進捗", unit="動画"):
        video_output_path = VIDEO_OUTPUT_DIR / video_path.name
        csv_output_path = CSV_OUTPUT_DIR / (video_path.stem + ".csv")
        # log.infoはtqdmのプログレスバーを壊す可能性があるので、デバッグレベルで出力
        logging.debug(f"動画 {video_path.name} の処理を開始します。") 
        process_one_video(video_path, video_output_path, csv_output_path, model)
        logging.debug(f"動画 {video_path.name} の処理が完了しました。")

    logging.info("="*50)
    logging.info(f"すべての処理が完了しました。")
    logging.info(f"ビデオ出力先: {VIDEO_OUTPUT_DIR}")
    logging.info(f"CSV出力先: {CSV_OUTPUT_DIR}")
    logging.info("="*50)


if __name__ == "__main__":
    main()