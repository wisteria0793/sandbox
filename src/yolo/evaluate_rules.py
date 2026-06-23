import pandas as pd
from pathlib import Path
import numpy as np
from rule_detector import RuleDetector

def evaluate():
    summary_path = Path("./data/annotation_summary.csv")
    annotations_dir = Path("./data/annotations")
    
    if not summary_path.exists():
        print(f"Error: {summary_path} が存在しません。")
        return
        
    summary_df = pd.read_csv(summary_path)
    
    # ラベルが設定されているデータのみ抽出 (entry, exit, passing_by のみ対象)
    test_df = summary_df[summary_df['label'].notna() & (summary_df['label'] != "")].copy()
    test_df = test_df[test_df['label'].isin(['entry', 'exit', 'passing_by'])]
    
    if test_df.empty:
        print("評価用のアノテーション済みデータが見つかりません。")
        return
        
    print(f"評価用アノテーション数: {len(test_df)} 件")
    print(test_df['label'].value_counts())
    print("-" * 50)
    
    # 2Dゲート判定ルール (ベストパラメータを指定、トラック結合はオフ)
    detector = RuleDetector(x_line=1500, y_door_max=350, max_frame_diff=0, max_pixel_dist=50)
    
    y_true = []
    y_pred = []
    errors = []
    
    grouped = test_df.groupby('clip_name')
    
    for clip_name, group in grouped:
        csv_path = annotations_dir / f"{Path(clip_name).stem}.csv"
        if not csv_path.exists():
            continue
            
        predictions = detector.detect_clip_labels(csv_path)
        
        for idx, row in group.iterrows():
            track_id = int(row['track_id'])
            true_label = row['label']
            pred_label = predictions.get(track_id, 'passing_by')
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if true_label != pred_label:
                track_df = pd.read_csv(csv_path)
                person_df = track_df[track_df['track_id'] == track_id].sort_values('frame_id')
                if not person_df.empty:
                    cxs = ((person_df['x1'] + person_df['x2']) / 2).values
                    cys = ((person_df['y1'] + person_df['y2']) / 2).values
                    first_x = cxs[0]
                    last_x = cxs[-1]
                    min_x = np.min(cxs)
                    max_x = np.max(cxs)
                    min_y = np.min(cys)
                    max_y = np.max(cys)
                else:
                    first_x, last_x, min_x, max_x, min_y, max_y = 0, 0, 0, 0, 0, 0
                
                errors.append({
                    'clip_name': clip_name,
                    'track_id': track_id,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'first_x': first_x,
                    'last_x': last_x,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                })

    # 評価指標の計算と表示
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    print(f"全体正解率 (Accuracy): {accuracy:.4f}\n")
    
    classes = ['entry', 'exit', 'passing_by']
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
    print("-" * 50)
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{c:<12} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
        
    if errors:
        print("\n" + "="*20 + " 誤判定事例の詳細 (Top 15) " + "="*20)
        errors_df = pd.DataFrame(errors)
        print(errors_df.head(15).to_string())
        
        print("\n誤判定の組み合わせ:")
        print(errors_df.groupby(['true_label', 'pred_label']).size().reset_index(name='count'))

if __name__ == "__main__":
    evaluate()
