import os
import json
import numpy as np

# Thư mục log, lấy từ script đang chạy
current_dir = os.path.dirname(__file__)
log_folder = os.path.abspath(os.path.join(current_dir, '..', 'log'))

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

# Dict lưu kết quả
score_means = {t: [] for t in types}
query_times = {t: [] for t in types}

for i in range(1, 21):
    sentence_folder = os.path.join(log_folder, f"sentence{i}")
    for t in types:
        t_path = os.path.join(sentence_folder, t)

        if "popop" in t:
            # Duyệt tất cả subfolder trong popop
            if os.path.exists(t_path):
                for sub in os.listdir(t_path):
                    sub_file = os.path.join(t_path, sub, "score_dict.json")
                    if os.path.exists(sub_file):
                        with open(sub_file) as f:
                            d = json.load(f)
                            values = list(d.values())
                            if values:
                                score_means[t].append(np.mean(values))  # mean của subfolder
                                query_times[t].append(len(values))      # số lượng entry
        else:
            # Với các loại khác: chỉ 1 file
            file_path = os.path.join(t_path, "score_dict.json")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    d = json.load(f)
                    values = list(d.values())
                    if values:
                        score_means[t].extend(values)          # gom hết để tính chung
                        query_times[t].append(len(values))     # mỗi file có 1 query time

# Xuất kết quả
print("\n=== Mean & Std Score ===")
for t in types:
    arr = np.array(score_means[t])
    print(f"{t}: Mean={arr.mean():.6f}, Std={arr.std():.6f}")

print("\n=== Mean & Std Query Time ===")
for t in types:
    qt = np.array(query_times[t])
    print(f"{t}: Mean={qt.mean():.2f}, Std={qt.std():.2f}")
