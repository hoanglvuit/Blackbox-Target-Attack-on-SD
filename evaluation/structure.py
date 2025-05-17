import os

def print_directory_structure(path, depth, max_depth, indent=0):
    # Nếu độ sâu hiện tại vượt quá max_depth, dừng lại
    if depth > max_depth:
        return
    
    # In tên thư mục hoặc tệp
    print('  ' * indent + os.path.basename(path) + ('/' if os.path.isdir(path) else ''))
    
    # Nếu còn độ sâu để duyệt, tiếp tục duyệt các thư mục con
    if os.path.isdir(path) and depth < max_depth:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            print_directory_structure(item_path, depth + 1, max_depth, indent + 1)

# Ví dụ sử dụng
root_directory = r'top3_log\sentence1'
max_depth = 4  # Độ sâu tối đa để in ra
print_directory_structure(root_directory, 0, max_depth)
