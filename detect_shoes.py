import os
import argparse
import requests
from requests.auth import HTTPDigestAuth
from pathlib import Path

# ================= 設定 =================
# Tapo C200 カメラの接続情報 (ローカルIPアドレスを設定してください)
TAPO_IP = "192.168.1.100"
TAPO_USER = "your_tapo_user"       # Tapoアプリの「カメラのアカウント」で設定したもの
TAPO_PASSWORD = "your_tapo_password"

# 保存用一時ディレクトリ
SAVE_DIR = Path("./data/shoes_detection")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# =======================================

def get_tapo_snapshot(ip, user, password, output_path):
    """
    Tapo C200 から ONVIF Snapshot 機能を利用して現在の静止画を取得・保存する。
    TapoカメラはDigest認証が必要なため、HTTPDigestAuthを使用します。
    """
    snapshot_url = f"http://{ip}/onvif-service/image"
    print(f"Tapoカメラから静止画を取得中: {snapshot_url}")
    
    try:
        response = requests.get(
            snapshot_url, 
            auth=HTTPDigestAuth(user, password), 
            timeout=10
        )
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"静止画を保存しました: {output_path}")
            return True
        else:
            print(f"エラー: 画像の取得に失敗しました (ステータスコード: {response.status_code})")
            print("ユーザー名やパスワード、カメラのIPアドレスが正しいか確認してください。")
            return False
    except Exception as e:
        print(f"接続エラー: {e}")
        return False

def detect_shoes_with_gemini(image_path):
    """
    Gemini 1.5 Flash API を利用して、画像内に靴があるか判定する。
    実行には環境変数 GEMINI_API_KEY の設定が必要です。
    """
    import google.generativeai as genai
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[エラー] Gemini API キーが設定されていません。")
        print("環境変数 GEMINI_API_KEY を設定するか、--ollama オプションを使用してください。")
        print("設定例 (ターミナル): export GEMINI_API_KEY='your_api_key_here'")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    print("Gemini API で画像を解析中...")
    try:
        # 画像ファイルを読み込む
        image_data = [{
            'mime_type': 'image/jpeg',
            'data': Path(image_path).read_bytes()
        }]
        
        # 靴の有無を判定するプロンプト
        prompt = """
        これは民泊の玄関（靴を脱ぐたたきのスペース）を撮影した写真です。
        現在、玄関にゲストの「靴（サンダル、スニーカー、ブーツなど何でも）」が置いてありますか？
        置いてある場合は 'True'、置いてない（1足もない）場合は 'False' とだけ答えてください。
        理由を述べる必要はありません。必ず 'True' または 'False' のどちらか一言のみで回答してください。
        """
        
        response = model.generate_content([prompt, image_data[0]])
        result = response.text.strip()
        return result
    except Exception as e:
        print(f"Gemini API エラー: {e}")
        return None

def detect_shoes_with_ollama(image_path):
    """
    Ollama (MiniCPM) を利用して、画像内に靴があるか判定する。
    ローカルPCのスペックが必要ですが、APIキーは不要です。
    """
    try:
        import ollama
    except ImportError:
        print("[エラー] ollama パッケージがインストールされていません。")
        return None

    print("Ollama (MiniCPM) で画像を解析中...")
    prompt = """
    これは民泊の玄関を撮影した写真です。
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
                'images': [image_path]
            }]
        )
        result = response['message']['content'].strip()
        return result
    except Exception as e:
        print(f"Ollama エラー: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Tapoカメラから写真を取得し、玄関に靴があるかを判定（在室状況の確認）します。")
    parser.add_argument("--image", type=str, help="テスト用の既存の画像パス（カメラから取得しない場合）")
    parser.add_argument("--ollama", action="store_true", help="Gemini APIではなくローカルのOllamaを使用する")
    args = parser.parse_args()

    image_path = SAVE_DIR / "current_status.jpg"

    if args.image:
        # 指定されたテスト画像を使用
        if not Path(args.image).exists():
            print(f"エラー: 指定された画像 {args.image} が見つかりません。")
            return
        image_path = Path(args.image)
    else:
        # Tapoカメラから最新の写真を取得
        success = get_tapo_snapshot(TAPO_IP, TAPO_USER, TAPO_PASSWORD, image_path)
        if not success:
            print("カメラからの画像取得に失敗したため、処理を中断します。")
            return

    # 解析を実行
    if args.ollama:
        result = detect_shoes_with_ollama(str(image_path))
    else:
        result = detect_shoes_with_gemini(str(image_path))

    if result is not None:
        print("\n" + "="*40)
        print(f"【判定結果】")
        print(f"靴の有無: {result}")
        if "True" in result:
            print("👉 判定: ゲストは【在室中】の可能性が高いです。")
        elif "False" in result:
            print("👉 判定: 玄関に靴はありません（外出中 または 未チェックイン/チェックアウト済）。")
        else:
            print(f"👉 判定結果を解釈できませんでした: {result}")
        print("="*40)

if __name__ == "__main__":
    main()
