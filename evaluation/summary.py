import os
import json
import numpy as np
from glob import glob


base_path = "output"

# Các loại cần lấy dữ liệu
categories = [
    "original",
    "target",
    "NoTE/beam_uni",
    "NoTE/beam_de",
    "NoTE/beam_in",
    "NoTE/popop",
    "TE/beam_uni",
    "TE/beam_de",
    "TE/beam_in",
    "TE/popop",
]

# Các loại metric cần xử lý
metrics = [
    "clip_score.json",
    "success_rate_oo.json",
    "success_rate_to.json",
    "success_rate_both.json"
]

results = {}

for category in categories:
    for metric in metrics:
        all_scores = []

        for i in range(1, 21):  # sentence1 -> sentence20
            sentence_folder = os.path.join(base_path, f"sentence{i}")

            if "popop" in category:
                # Lặp qua tất cả các subfolder 22xxxxx
                popop_base = os.path.join(sentence_folder, category)
                if os.path.exists(popop_base):
                    subfolders = glob(os.path.join(popop_base, "*"))
                    for sub in subfolders:
                        metric_path = os.path.join(sub, metric)
                        if os.path.exists(metric_path):
                            with open(metric_path, "r") as f:
                                data = json.load(f)
                                values = list(data.values())
                                all_scores.extend(values)
            else:
                metric_path = os.path.join(sentence_folder, category, metric)
                if os.path.exists(metric_path):
                    with open(metric_path, "r") as f:
                        data = json.load(f)
                        values = list(data.values())
                        all_scores.extend(values)

        # Tính toán mean và std
        mean_val = np.mean(all_scores) if all_scores else None
        std_val = np.std(all_scores) if all_scores else None

        key = f"{category.replace('/', '_')}::{metric.replace('.json','')}"
        results[key] = {
            "mean": mean_val,
            "std": std_val
        }

# In ra kết quả
for k, v in results.items():
    print(f"{k} -> Mean: {v['mean']:.4f}, Std: {v['std']:.4f}")
