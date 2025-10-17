import os
import shutil

# 文件路径
match_file_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_match.txt"
input_folder = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_original"
output_folder = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_rename_original"
error_log_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_log_error_rename.txt"

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

def load_match_file(file_path):
    """加载 match.txt，返回匹配信息"""
    matches = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                error_name, dbpedia_url, wikidata_url = parts
                qid = wikidata_url.split('/')[-1]  # 提取 QID
                matches.append((error_name, qid))
    return matches

def process_folders(matches, input_folder, output_folder, error_log_path):
    """根据 match.txt 复制和重命名图片"""
    error_entries = []

    for error_name, qid in matches:
        folder_path = os.path.join(input_folder, error_name)
        if not os.path.isdir(folder_path):
            error_entries.append(error_name)  # 未找到匹配文件夹
            continue

        try:
            # 遍历文件夹中的图片
            for idx, image_name in enumerate(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)
                if not os.path.isfile(image_path):
                    continue
                # 重命名图片
                new_image_name = f"{qid}_{idx}.jpg"
                new_image_path = os.path.join(output_folder, new_image_name)
                shutil.copy2(image_path, new_image_path)
        except Exception as e:
            error_entries.append(error_name)
            print(f"处理文件夹 {error_name} 时出错: {e}")

    # 保存错误日志
    with open(error_log_path, 'w', encoding='utf-8') as error_log:
        for error in error_entries:
            error_log.write(f"{error}\n")

def main():
    # 加载匹配文件
    matches = load_match_file(match_file_path)
    # 处理文件夹和图片
    process_folders(matches, input_folder, output_folder, error_log_path)
    print(f"处理完成。错误日志已保存到 {error_log_path}")

if __name__ == "__main__":
    main()
