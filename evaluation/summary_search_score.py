import os
import json
import numpy as np

# Đường dẫn tới folder chứa sentence1 -> sentence20
root_folder = "output"

# Các loại cần xử lý
types = [
    "NoTE/beam_uni",
    "NoTE/beam_de",
    "NoTE/beam_in",
    "NoTE/popop",
    "TE/beam_uni",
    "TE/beam_de",
    "TE/beam_in",
    "TE/popop"
]

# Dict để lưu các giá trị đã xử lý
scores = {t: [] for t in types}

# Lặp qua từng câu
for i in range(1, 21):
    sentence_folder = os.path.join(root_folder, f"sentence{i}")
    for t in types:
        t_path = os.path.join(sentence_folder, t)
        if "popop" in t:
            # Với popop: tính mean từng subfolder
            if os.path.exists(t_path):
                for sub in os.listdir(t_path):
                    sub_path = os.path.join(t_path, sub, "search_score.json")
                    if os.path.exists(sub_path):
                        with open(sub_path) as f:
                            data = json.load(f)
                            if data:
                                sub_mean = np.mean(data)
                                scores[t].append(sub_mean)
        else:
            # Với các loại còn lại: lấy tất cả score
            score_path = os.path.join(t_path, "search_score.json")
            if os.path.exists(score_path):
                with open(score_path) as f:
                    data = json.load(f)
                    scores[t].extend(data)

# Tính toán mean và std
for t in types:
    arr = np.array(scores[t])
    print(f"{t}:")
    print(f"  Mean: {arr.mean():.6f}")
    print(f"  Std:  {arr.std():.6f}")
