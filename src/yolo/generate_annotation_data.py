import cv2
from ultralytics import YOLO
import os
from pathlib import Path
import logging
import csv
from tqdm import tqdm

# --- 設定 ---
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 入力ディレクトリ: 分割されたクリップが保存されているフォルダ
INPUT_CLIP_DIR = Path("./data/videos_split")
# 出力ディレクトリ: アノテーションデータを保存するCSVファイルを個別に生成するフォルダ
ANNOTATIONS_DIR = Path("./data/annotations")

# 使用するモデル
MODEL_NAME = "yolo11n.pt"

# 検出設定
PERSON_CLASS_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
# --- 設定ここまで ---


def main():
    """メイン関数"""
    if not INPUT_CLIP_DIR.exists():
        logging.error(f"入力ディレクトリが見つかりません: {INPUT_CLIP_DIR}")
        logging.error("先に 'split_videos_by_activity.py' を実行して、動画クリップを生成してください。")
        return

    # アノテーション出力ディレクトリを作成
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # モデルをロード
    logging.info(f"モデルをロードしています: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        return

    # 入力ディレクトリ内の動画クリップを取得
    clip_files = sorted(list(INPUT_CLIP_DIR.glob("*.mp4")))
    if not clip_files:
        logging.warning(f"入力ディレクトリに動画クリップが見つかりません: {INPUT_CLIP_DIR}")
        return
            
    logging.info(f"合計{len(clip_files)}個のクリップを処理し、アノテーションデータを生成します。")

    # 各クリップを処理し、個別のCSVを生成
    for clip_path in tqdm(clip_files, desc="クリップを処理中", unit="クリップ"):
        output_csv_path = ANNOTATIONS_DIR / (clip_path.stem + ".csv")
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['clip_name', 'frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id']
            writer.writerow(header)

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                logging.error(f"クリップを開けませんでした: {clip_path.name}")
                continue
            
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # フレームに対してトラッキングを実行
                results = model.track(frame, persist=True, classes=[PERSON_CLASS_INDEX], conf=CONFIDENCE_THRESHOLD, verbose=False)

                # トラッキングIDが割り当てられている結果のみを処理
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    cls_ids = results[0].boxes.cls.cpu().numpy()

                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = boxes[i]
                        row = [
                            clip_path.name, # クリップ名も保持
                            frame_index,
                            int(track_id),
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            confs[i],
                            int(cls_ids[i])
                        ]
                        writer.writerow(row)
                
                frame_index += 1
            
            cap.release()

    logging.info("すべてのクリップのアノテーションデータ生成が完了しました。")

if __name__ == "__main__":
    main()