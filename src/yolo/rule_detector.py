import pandas as pd
import numpy as np
from pathlib import Path

class RuleDetector:
    def __init__(self, x_line=1500, y_door_max=350, max_frame_diff=45, max_pixel_dist=200):
        """
        Args:
            x_line (int): 玄関扉の境界線とするX座標 (0〜1920)
            y_door_max (int): 玄関扉の境界線とする最大Y座標。これより上（小さい値）がドアゾーン。
            max_frame_diff (int): トラック結合時に許容する最大フレーム間隔 (15fps動画なので30で2秒、45で3秒)
            max_pixel_dist (float): トラック結合時に許容する最大移動ピクセル距離
        """
        self.x_line = x_line
        self.y_door_max = y_door_max
        self.max_frame_diff = max_frame_diff
        self.max_pixel_dist = max_pixel_dist
        self.room_zone_x = 1400  # これより左側は室内

    def detect_clip_labels(self, csv_path_or_df):
        """
        特定のクリップのCSVデータを読み込み、トラック結合を行ったうえで、
        アノテーション対象の各 track_id (オリジナル) に対する判定を辞書形式で返す。
        
        Args:
            csv_path_or_df (Path/str/DataFrame): クリップのトラッキングデータ
            
        Returns:
            dict: { original_track_id: predicted_label }
        """
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df
        else:
            df = pd.read_csv(csv_path_or_df)
            
        if df.empty or 'track_id' not in df.columns:
            return {}

        # 1. 各トラックの情報を整理
        tracks = {}
        unique_ids = df['track_id'].unique()
        
        for tid in unique_ids:
            tdf = df[df['track_id'] == tid].sort_values('frame_id')
            if len(tdf) < 2:
                continue
            
            cxs = ((tdf['x1'] + tdf['x2']) / 2).values
            cys = ((tdf['y1'] + tdf['y2']) / 2).values
            frames = tdf['frame_id'].values
            
            tracks[int(tid)] = {
                'id': int(tid),
                'start_f': int(frames[0]),
                'end_f': int(frames[-1]),
                'start_x': cxs[0],
                'start_y': cys[0],
                'end_x': cxs[-1],
                'end_y': cys[-1],
                'cxs': cxs,
                'cys': cys,
                'frames': frames
            }

        if not tracks:
            return {}

        # 2. トラックの結合 (Stitching)
        if self.max_frame_diff > 0:
            merged_groups = {tid: [tid] for tid in tracks}
            sorted_tids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
            
            for i in range(len(sorted_tids)):
                tid_a = sorted_tids[i]
                track_a = tracks[tid_a]
                
                for j in range(i + 1, len(sorted_tids)):
                    tid_b = sorted_tids[j]
                    track_b = tracks[tid_b]
                    
                    frame_diff = track_b['start_f'] - track_a['end_f']
                    if frame_diff > self.max_frame_diff:
                        break
                    
                    dist = np.sqrt((track_b['start_x'] - track_a['end_x'])**2 + 
                                   (track_b['start_y'] - track_a['end_y'])**2)
                    
                    if frame_diff > 0 and dist <= self.max_pixel_dist:
                        group_a = merged_groups[tid_a]
                        group_b = merged_groups[tid_b]
                        
                        if group_a is not group_b:
                            new_group = list(set(group_a + group_b))
                            for member in new_group:
                                merged_groups[member] = new_group

            unique_groups = []
            for g in merged_groups.values():
                if g not in unique_groups:
                    unique_groups.append(g)
        else:
            # 結合なし
            unique_groups = [[tid] for tid in tracks.keys()]

        # マージされた各グループについて判定
        group_predictions = {}
        
        for g in unique_groups:
            g_tracks = [tracks[tid] for tid in g]
            g_tracks = sorted(g_tracks, key=lambda x: x['start_f'])
            
            all_cxs = np.concatenate([t['cxs'] for t in g_tracks])
            all_cys = np.concatenate([t['cys'] for t in g_tracks])
            
            pred = self._classify_trajectory(all_cxs, all_cys)
            
            for tid in g:
                group_predictions[tid] = pred

        predictions = {}
        for tid in unique_ids:
            predictions[int(tid)] = group_predictions.get(int(tid), 'passing_by')

        return predictions

    def _classify_trajectory(self, cxs, cys):
        """
        2Dゲート（XとY座標の条件）を用いた入退室判定ルール
        """
        # ドアゾーンの定義: X >= x_line かつ Y <= y_door_max
        in_door_zone = (cxs >= self.x_line) & (cys <= self.y_door_max)
        has_door = np.any(in_door_zone)
        
        # 部屋ゾーンの定義: X < room_zone_x
        in_room_zone = (cxs < self.room_zone_x)
        has_room = np.any(in_room_zone)
        
        # ドアゾーンと部屋ゾーンの両方に存在した軌跡のみが入退室候補
        if has_door and has_room:
            # 移動方向 (全体のX座標の変位)
            # 最初と最後の平均値を使うことでノイズ対策
            k = min(5, len(cxs))
            avg_start_x = np.mean(cxs[:k])
            avg_end_x = np.mean(cxs[-k:])
            dx = avg_end_x - avg_start_x
            
            if dx < -30:
                return 'entry' # 室内へ移動
            elif dx > 30:
                return 'exit'  # ドアへ移動
                
        return 'passing_by'
