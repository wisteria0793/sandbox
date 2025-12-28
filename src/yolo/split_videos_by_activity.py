import cv2
from ultralytics import YOLO
from pathlib import Path
import logging
from tqdm import tqdm # Added tqdm

# --- 設定 ---
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 入力と出力のディレクトリ
# 入力: オリジナルの動画が保存されているフォルダ
INPUT_VIDEO_DIR = Path("./data/videos")
# 出力: 切り出されたクリップを保存するフォルダ
OUTPUT_CLIP_DIR = Path("./data/videos_split2")

# 使用するモデル
MODEL_NAME = "yolo11n.pt"

# 検出設定
# 何フレームごとに人物検出を実行するか（1に近いほど高精度だが遅い）
DETECTION_INTERVAL = 15  # 何フレームごとに人物検出を実行するか（整数値）
# 人物クラスのインデックス（YOLOのCOCOデータセットでは0がperson）
PERSON_CLASS_INDEX = 0
# 検出の信頼度の閾値
CONFIDENCE_THRESHOLD = 0.5

# クリップ設定
# 人物が検出されなくなってから、何秒間録画を続けるか
POST_ACTIVITY_BUFFER_SECONDS = 5
# 保存するクリップの最小の長さ（秒）これより短いクリップは無視する
MIN_CLIP_DURATION_SECONDS = 3
# --- 設定ここまで ---


def process_video(video_path, model, output_dir):
    """単一の動画を処理してクリップを抽出する（メモリ効率化版）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"動画ファイルを開けませんでした: {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logging.warning(f"動画 {video_path.name} のFPSが不正です ({fps})。デフォルトの30fpsを使用します。")
        fps = 30
    
    post_activity_buffer_frames = int(POST_ACTIVITY_BUFFER_SECONDS * fps)
    
    video_writer = None
    temp_clip_path = None
    frames_since_last_detection = 0
    clip_count = 0
    frame_index = 0
    current_clip_start_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        person_detected_in_interval = False
        if frame_index % DETECTION_INTERVAL == 0:
            results = model(frame, classes=[PERSON_CLASS_INDEX], conf=CONFIDENCE_THRESHOLD, verbose=False)
            if len(results[0].boxes) > 0:
                person_detected_in_interval = True

        if person_detected_in_interval:
            frames_since_last_detection = 0
            if video_writer is None:
                # 録画開始
                current_clip_start_frame = frame_index
                clip_count += 1
                # ログレベルをDEBUGにすることでデフォルトでは表示されないように変更
                logging.debug(f"アクティビティを検出、クリップの記録を開始します。 (フレーム: {frame_index})")
                
                temp_filename = f"{video_path.stem}_clip_{clip_count:03d}.tmp.mp4"
                temp_clip_path = output_dir / temp_filename
                
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(temp_clip_path), fourcc, fps, (width, height))
        else:
            if video_writer is not None:
                frames_since_last_detection += 1

        if video_writer is not None:
            video_writer.write(frame)

        if video_writer is not None and frames_since_last_detection > post_activity_buffer_frames:
            # 録画終了とクリップの最終処理
            video_writer.release()
            video_writer = None
            
            clip_duration_frames = frame_index - current_clip_start_frame
            clip_duration_seconds = clip_duration_frames / fps if fps > 0 else 0

            if clip_duration_seconds < MIN_CLIP_DURATION_SECONDS:
                # ログレベルをDEBUGにすることでデフォルトでは表示されないように変更
                logging.debug(f"クリップが短すぎるため削除します（{clip_duration_seconds:.2f}秒）。")
                if temp_clip_path.exists():
                    temp_clip_path.unlink()
            else:
                final_filename = f"{video_path.stem}_clip_{clip_count:03d}.mp4"
                final_clip_path = output_dir / final_filename
                temp_clip_path.rename(final_clip_path)
                # ログレベルをDEBUGにすることでデフォルトでは表示されないように変更
                logging.debug(f"クリップを保存しました: {final_clip_path} ({clip_duration_seconds:.2f}秒)")
            
            temp_clip_path = None
            frames_since_last_detection = 0

        frame_index += 1

    # 動画の終端でまだ録画中だった場合のクリップを保存
    if video_writer is not None:
        video_writer.release()
        clip_duration_frames = frame_index - current_clip_start_frame
        clip_duration_seconds = clip_duration_frames / fps if fps > 0 else 0
        
        if clip_duration_seconds < MIN_CLIP_DURATION_SECONDS:
            # ログレベルをDEBUGにすることでデフォルトでは表示されないように変更
            logging.debug(f"動画終端のクリップが短すぎるため削除します（{clip_duration_seconds:.2f}秒）。")
            if temp_clip_path and temp_clip_path.exists():
                temp_clip_path.unlink()
        else:
            final_filename = f"{video_path.stem}_clip_{clip_count:03d}.mp4"
            final_clip_path = output_dir / final_filename
            if temp_clip_path:
                temp_clip_path.rename(final_clip_path)
                # ログレベルをDEBUGにすることでデフォルトでは表示されないように変更
                logging.debug(f"動画終端のクリップを保存しました: {final_clip_path} ({clip_duration_seconds:.2f}秒)")

    cap.release()


def main():
    """メイン関数"""
    OUTPUT_CLIP_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"モデルをロードしています: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        return

    video_files = sorted(list(INPUT_VIDEO_DIR.glob("*.mp4")))
    if not video_files:
        logging.warning(f"入力ディレクトリに動画ファイルが見つかりません: {INPUT_VIDEO_DIR}")
        return
        
    total_videos = len(video_files)
    logging.info(f"合計{total_videos}個の動画ファイルを処理します。")

    # tqdmで進捗バーを表示
    for video_path in tqdm(video_files, desc="動画を処理中", unit="動画"):
        # 個別の動画開始/完了ログは、tqdmと重複するため、ここでは出力しない。
        # 代わりに、tqdmのpostfixに動画名を設定するなどの工夫も可能。
        process_video(video_path, model, OUTPUT_CLIP_DIR)
        
    logging.info("すべての動画ファイルの処理が完了しました。")

if __name__ == "__main__":
    main()