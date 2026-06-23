# 実装タスクリスト

- `[x]` 判定ロジック of コアモジュール実装 (`src/yolo/rule_detector.py`)
  - `[x]` 同一人物とみなせる断片化トラックの結合ロジックの作成 (結合なしが最適であると判明)
  - `[x]` 境界線 (X=1500, Y=350) と移動方向に基づく入退室 (entry / exit) 判定ルールの実装
- `[x]` 評価用スクリプト実装 (`src/yolo/evaluate_rules.py`)
  - `[x]` `annotation_summary.csv` のアノテーション（entry, exit, passing_by）と予測値の比較
  - `[x]` 精度（Accuracy, Precision, Recall, F1）の計算と表示
  - `[x]` 誤判定したデータの詳細ログ出力（デバッグ用）
- `[x]` 評価とチューニングの実行
  - `[x]` 評価スクリプトを実行し、初期精度を測定する
  - `[x]` 誤判定されたデータを分析し、閾値や結合パラメータを微調整する
- `[x]` 最終結果のまとめと Walkthrough の作成
