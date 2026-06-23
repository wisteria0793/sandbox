import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# --- 設定 ---
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 入力ディレクトリ: 個別のCSVが保存されているフォルダ
INPUT_ANNOTATIONS_DIR = Path("./data/annotations")
# 出力ファイル: ラベル付けを行うためのサマリーファイル
OUTPUT_SUMMARY_FILE = Path("./data/annotation_summary.csv")
# --- 設定ここまで ---


def main():
    """メイン関数"""
    if not INPUT_ANNOTATIONS_DIR.exists():
        logging.error(f"入力ディレクトリが見つかりません: {INPUT_ANNOTATIONS_DIR}")
        logging.error("先に 'generate_annotation_data.py' を実行して、個別のCSVファイルを生成してください。")
        return

    csv_files = sorted(list(INPUT_ANNOTATIONS_DIR.glob("*.csv")))
    if not csv_files:
        logging.warning(f"入力ディレクトリにCSVファイルが見つかりません: {INPUT_ANNOTATIONS_DIR}")
        return

    logging.info(f"合計{len(csv_files)}個のCSVファイルを読み込んでいます...")

    # 全てのCSVを一つのDataFrameに結合
    df_list = [pd.read_csv(f) for f in tqdm(csv_files, desc="CSVファイルを処理中")]
    full_df = pd.concat(df_list, ignore_index=True)

    logging.info("ユニークなトラックを抽出しています...")

    # 'clip_name'と'track_id'のユニークな組み合わせを抽出
    summary_df = full_df[['clip_name', 'track_id']].drop_duplicates().sort_values(by=['clip_name', 'track_id'])

    # ラベルを書き込むための空の'label'列を追加
    summary_df['label'] = ""

    # サマリーファイルをCSVとして保存（インデックスは不要）
    summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False, encoding='utf-8')

    logging.info("="*50)
    logging.info(f"アノテーションサマリーファイルの生成が完了しました: {OUTPUT_SUMMARY_FILE}")
    logging.info(f"このファイルを開き、{len(summary_df)}個の各トラックに対して'label'列を記入してください。")
    logging.info("="*50)


if __name__ == "__main__":
    main()
