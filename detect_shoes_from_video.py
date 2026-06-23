import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# ================= 設定 =================
SAVE_DIR = Path("./data/shoes_detection")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# =======================================

def analyze_frame_with_gemini(frame, frame_idx, timestamp_sec):
    """
    OpenCVのフレーム画像をメモリ上でJPEGに変換し、Gemini APIに送信して靴の有無を判定する。
    """
    import google.generativeai as genai
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[エラー] Gemini API キーが設定されていません。")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # メモリ上で OpenCV (BGR) を JPEG にエンコード
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        print(f"[{timestamp_sec:.1f}秒] フレームの画像エンコードに失敗しました。")
        return None

    try:
        image_data = [{
            'mime_type': 'image/jpeg',
            'data': encoded_image.tobytes()
        }]
        
        prompt = """
        これは民泊の玄関を映した防犯カメラのフレーム画像です。
        玄関のたたき（靴を脱ぐ床のスペース）に、ゲストの「靴（スニーカー、サンダル、革靴など何でも）」が置いてありますか？
        置いてある場合は 'True'、1足も置いてない場合は 'False' とだけ答えてください。
        理由を述べる必要はありません。必ず 'True' または 'False' のどちらか一言のみで回答してください。
        """
        
        response = model.generate_content([prompt, image_data[0]])
        result = response.text.strip()
        return result
    except Exception as e:
        print(f"[{timestamp_sec:.1f}秒] Gemini API エラー: {e}")
        return None

def analyze_frame_with_ollama(frame, frame_idx, timestamp_sec):
    """
    フレームを一時ファイルに保存し、Ollama (MiniCPM) を用いて靴の有無を判定する。
    """
    try:
        import ollama
    except ImportError:
        print("[エラー] ollama パッケージがインストールされていません。")
        return None

    # 一時ファイルとして保存
    temp_img_path = SAVE_DIR / f"temp_frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(temp_img_path), frame)

    prompt = """
    これは民泊の玄関を映した写真です。
    玄関の床（靴を脱ぐスペース）に、靴が置いてありますか？
    置いてある場合は 'True'、置いてない場合は 'False' と答えてください。
    理由を述べる必要はありません。必ず 'True' または 'False' のどちらかのみで答えてください。
    """
    
    try:
        response = ollama.chat(
            model='openbmb/minicpm-o2.6:8b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(temp_img_path)]
            }]
        )
        result = response['message']['content'].strip()
        # 一時ファイルの削除
        if temp_img_path.exists():
            temp_img_path.unlink()
        return result
    except Exception as e:
        print(f"[{timestamp_sec:.1f}秒] Ollama エラー: {e}")
        if temp_img_path.exists():
            temp_img_path.unlink()
        return None

def main():
    parser = argparse.ArgumentParser(description="動画データから、各フレームにおける玄関の靴の有無を判定します。")
    parser.add_argument("video_path", type=str, help="解析したい動画ファイル（.mp4など）のパス")
    parser.add_argument("--interval", type=float, default=2.0, help="判定を行うフレーム間隔（秒）。デフォルトは2.0秒ごと")
    parser.add_argument("--ollama", action="store_true", help="Gemini APIの代わりにローカルのOllamaを使用する")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"エラー: 指定された動画ファイルが見つかりません: {video_path}")
        return

    # 動画の読み込み
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    # 判定を行うフレーム間隔の計算
    frame_interval = int(fps * args.interval) if fps > 0 else 30
    if frame_interval == 0:
        frame_interval = 1

    print(f"動画解析を開始します: {video_path.name}")
    print(f"動画の長さ: {duration_sec:.1f}秒 (総フレーム数: {total_frames}, FPS: {fps:.2f})")
    print(f"解析間隔: {args.interval}秒おき (約 {frame_interval} フレームごと)")
    print("-" * 50)

    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 指定したインターバルでのみ処理
        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps if fps > 0 else 0
            
            if args.ollama:
                res = analyze_frame_with_ollama(frame, frame_count, timestamp_sec)
            else:
                res = analyze_frame_with_gemini(frame, frame_count, timestamp_sec)
                
            if res is not None:
                has_shoes = "True" in res
                status_str = "【靴あり (在室中)】" if has_shoes else "【靴なし (外出中)】"
                print(f"[{timestamp_sec:.1f}秒時点] -> {status_str} (判定: {res})")
                results.append((timestamp_sec, has_shoes))
                
        frame_count += 1

    cap.release()
    print("-" * 50)
    print("動画の解析が完了しました。")

    # 全体を通した状態変化の要約
    if len(results) >= 2:
        start_status = results[0][1]
        end_status = results[-1][1]
        
        print("\n【状態変化のサマリー】")
        print(f"動画開始時 (0.0秒): {'靴あり (在室中)' if start_status else '靴なし (外出中)'}")
        print(f"動画終了時 ({results[-1][0]:.1f}秒): {'靴あり (在室中)' if end_status else '靴なし (外出中)'}")
        
        # 変化があったタイミングの検出
        changes = []
        for i in range(1, len(results)):
            prev_time, prev_status = results[i-1]
            curr_time, curr_status = results[i]
            if prev_status != curr_status:
                changes.append((curr_time, prev_status, curr_status))
                
        if changes:
            print("\n🚨 状態の変化を検出しました:")
            for time_sec, before, after in changes:
                before_str = "靴あり ➡ 靴なし (外出/退出)" if before else "靴なし ➡ 靴あり (帰宅/入室)"
                print(f" - {time_sec:.1f}秒付近: {before_str}")
        else:
            print("\n🔄 動画中に状態の変化（入退室）は検出されませんでした。")

if __name__ == "__main__":
    main()
