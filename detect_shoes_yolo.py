import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLOWorld

# ================= 設定 =================
# 検出する物体 (YOLO-Worldにセットするテキスト)
# 靴だけでなく、旅行用スーツケース、バッグ、傘、人そのものを検出対象に広げます
DETECT_CLASSES = ["shoes", "footwear", "suitcase", "bag", "backpack", "umbrella", "person"]
CONFIDENCE_THRESHOLD = 0.25  # 検出の閾値 (0.0〜1.0)
Y_MIN_LIMIT = 250            # 判定対象とする最小のY座標 (これより上側はノイズとして除外)
# =======================================

def filter_boxes(boxes, img_height):
    """
    検出されたボックスのうち、画面の上側（棚やスリッパラックの誤検知エリア）にあるものを除外する。
    カメラの移動に対応するため、Y座標の閾値（画面の下側のみを判定対象とする）で判定します。
    """
    valid_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        
        # ボックスの中心Y座標
        cy = (y1 + y2) / 2
        
        # 設定されたY_MIN_LIMITより下（大きい値）にある場合のみ有効とする
        if cy >= Y_MIN_LIMIT:
            valid_boxes.append(box)
            
    return valid_boxes

def draw_annotations(frame, boxes, names_dict):
    """
    有効と判定されたボックスのみを画像に描画する
    """
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = box.conf[0].cpu().item()
        cls_id = int(box.cls[0].cpu().item())
        label = names_dict.get(cls_id, "object")
        
        # 緑色の枠線を描画
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, 
            f"{label} {conf:.2f}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )
    return annotated

def detect_shoes_in_image(image_path, model, output_dir):
    """
    静止画からYOLO-Worldを使って靴や荷物を検出し、検出位置を囲んだ画像を保存する
    """
    print(f"YOLO-Worldで画像を解析中 (Y軸制限 >= {Y_MIN_LIMIT}): {image_path}")
    
    results = model.predict(
        source=str(image_path), 
        conf=CONFIDENCE_THRESHOLD, 
        verbose=False
    )
    
    img_height = results[0].orig_shape[0]
    names_dict = results[0].names
    
    # 座標による緩やかなフィルタリング
    all_boxes = results[0].boxes
    valid_boxes = filter_boxes(all_boxes, img_height)
    has_objects = len(valid_boxes) > 0
    
    if has_objects:
        print(f"-> {len(valid_boxes)}個の対象物を検出しました (除外されたノイズ: {len(all_boxes) - len(valid_boxes)}個)。")
        for box in valid_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls_id = int(box.cls[0].cpu().item())
            label = names_dict.get(cls_id, "object")
            print(f"   [{label} - 位置: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}), 確信度: {conf:.2f}]")
            
        # 枠線を描画して保存
        img = cv2.imread(str(image_path))
        annotated_frame = draw_annotations(img, valid_boxes, names_dict)
        output_path = output_dir / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"👉 検出位置を囲んだ画像を保存しました: {output_path}")
    else:
        print(f"-> 玄関に靴や荷物は検出されませんでした (除外されたノイズ: {len(all_boxes)}個)。")
        
    return has_objects

def detect_shoes_in_video(video_path, model, output_dir, interval_sec=2.0, save_viz=True):
    """
    動画ファイルから指定秒おきにフレームを切り出し、在室状況の変遷を記録する
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    frame_interval = int(fps * interval_sec) if fps > 0 else 30
    if frame_interval == 0:
        frame_interval = 1

    video_writer = None
    if save_viz:
        viz_path = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = 1.0 / interval_sec if interval_sec > 0 else 1.0
        video_writer = cv2.VideoWriter(str(viz_path), fourcc, output_fps, (width, height))
        print(f"判定結果動画の保存先: {viz_path}")

    print(f"YOLO-Worldで動画解析を開始します (Y軸制限 >= {Y_MIN_LIMIT}): {video_path.name}")
    print(f"動画の長さ: {duration_sec:.1f}秒 (総フレーム数: {total_frames}, FPS: {fps:.2f})")
    print(f"解析間隔: {interval_sec}秒おき (約 {frame_interval} フレームごと)")
    print("-" * 50)

    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps if fps > 0 else 0
            
            res = model.predict(
                source=frame, 
                conf=CONFIDENCE_THRESHOLD, 
                verbose=False
            )
            
            all_boxes = res[0].boxes
            valid_boxes = filter_boxes(all_boxes, height)
            names_dict = res[0].names
            has_objects = len(valid_boxes) > 0
            
            # 検出された物体のラベル一覧を作成
            detected_labels = [names_dict.get(int(box.cls[0].cpu().item()), "object") for box in valid_boxes]
            labels_str = ", ".join(detected_labels) if detected_labels else "なし"
            
            status_str = f"【在室中 (検出: {labels_str})】" if has_objects else "【外出中 / 不在】"
            print(f"[{timestamp_sec:.1f}秒時点] -> {status_str} (ノイズ除外: {len(all_boxes) - len(valid_boxes)}個)")
            results.append((timestamp_sec, has_objects))
            
            if video_writer is not None:
                annotated_frame = draw_annotations(frame, valid_boxes, names_dict)
                video_writer.write(annotated_frame)
            
        frame_count += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\n👉 判定箇所の囲み枠付き動画を保存しました: {viz_path}")
        
    print("-" * 50)
    print("動画の解析が完了しました。")

    # サマリー出力
    if len(results) >= 2:
        start_status = results[0][1]
        end_status = results[-1][1]
        
        print("\n【状態変化のサマリー】")
        print(f"動画開始時 (0.0秒): {'在室 (靴・荷物あり)' if start_status else '不在 (靴・荷物なし)'}")
        print(f"動画終了時 ({results[-1][0]:.1f}秒): {'在室 (靴・荷物あり)' if end_status else '不在 (靴・荷物なし)'}")
        
        changes = []
        for i in range(1, len(results)):
            prev_time, prev_status = results[i-1]
            curr_time, curr_status = results[i]
            if prev_status != curr_status:
                changes.append((curr_time, prev_status, curr_status))
                
        if changes:
            print("\n🚨 在室状況の変化を検出しました:")
            for time_sec, before, after in changes:
                before_str = "在室 ➡ 不在 (外出/チェックアウト)" if before else "不在 ➡ 在室 (帰宅/チェックイン)"
                print(f" - {time_sec:.1f}秒付近: {before_str}")
        else:
            print("\n🔄 動画中に在室状況の変化は検出されませんでした。")

def main():
    parser = argparse.ArgumentParser(description="YOLO-Worldを利用して、玄関の靴や荷物、人を検知し在室状況を判定します。")
    parser.add_argument("--image", type=str, help="解析したい画像ファイルのパス")
    parser.add_argument("--video", type=str, help="解析したい動画ファイルのパス")
    parser.add_argument("--interval", type=float, default=2.0, help="動画判定を行うフレーム間隔（秒）")
    parser.add_argument("--conf", type=float, default=0.25, help="検出の閾値 (デフォルト: 0.25)")
    parser.add_argument("--y-min", type=int, default=250, help="判定対象とする最小のY座標 (デフォルト: 250。これより上の検出は無視)")
    parser.add_argument("--no-viz", action="store_true", help="動画出力時に枠線付き動画の保存を無効化する")
    args = parser.parse_args()

    if not args.image and not args.video:
        print("エラー: --image または --video のどちらか一方を指定してください。")
        parser.print_help()
        return

    output_dir = Path("./data/shoes_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    global Y_MIN_LIMIT, CONFIDENCE_THRESHOLD
    Y_MIN_LIMIT = args.y_min
    CONFIDENCE_THRESHOLD = args.conf

    print("YOLO-Worldモデルを初期化しています...")
    try:
        model = YOLOWorld("yolov8s-worldv2.pt")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    model.set_classes(DETECT_CLASSES)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"エラー: 画像が見つかりません: {image_path}")
            return
        
        has_objects = detect_shoes_in_image(image_path, model, output_dir)
        print("\n" + "="*40)
        if has_objects:
            print("【判定結果】: ゲストは【在室中】の可能性が高いです。")
        else:
            print("【判定結果】: 玄関に靴や荷物は置かれていません (不在)。")
        print("="*40)

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"エラー: 動画が見つかりません: {video_path}")
            return
            
        detect_shoes_in_video(video_path, model, output_dir, args.interval, not args.no_viz)

if __name__ == "__main__":
    main()
