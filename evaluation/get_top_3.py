import os
import shutil
import json

def create_top3_log_folder():
    # Tạo folder "top3_log" trong cùng thư mục với file get_top_3.py
    folder_path = os.path.join(os.path.dirname(__file__), 'top3_log')
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def copy_log_to_top3(log_folder, top3_folder):
    # Copy toàn bộ nội dung từ log vào top3_log
    for item in os.listdir(log_folder):
        s = os.path.join(log_folder, item)
        d = os.path.join(top3_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def process_score_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Lấy top 3 entry có điểm cao nhất
        top_3 = dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:3])

        # Ghi đè lại file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(top_3, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}")

def clean_top3_log(top3_log_path):
    for root, dirs, files in os.walk(top3_log_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file == 'pool_score_log.json':
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Lỗi xóa file {file_path}: {e}")
            elif file == 'score_dict.json':
                process_score_dict(file_path)

if __name__ == "__main__":
    # Thiết lập đường dẫn
    current_dir = os.path.dirname(__file__)
    top3_log_dir = create_top3_log_folder()
    log_folder = os.path.abspath(os.path.join(current_dir, '..', 'log'))

    # Copy và xử lý dữ liệu
    copy_log_to_top3(log_folder, top3_log_dir)
    clean_top3_log(top3_log_dir)
