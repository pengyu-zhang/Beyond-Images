# import os
# import logging
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor

# def setup_logging():
#     """设置日志记录配置"""
#     logging.basicConfig(
#         level=logging.ERROR,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         filename='error.log',
#         filemode='a',
#         encoding='utf-8'
#     )

# def count_files_and_images(folder_path):
#     """
#     统计指定文件夹中的文件总数和图片文件数量

#     参数：
#         folder_path (str): 文件夹路径

#     返回：
#         dict: 包含文件总数和图片文件数量的字典
#     """
#     image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
#     total_files = 0
#     image_files = 0

#     try:
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 total_files += 1
#                 if os.path.splitext(file)[1].lower() in image_extensions:
#                     image_files += 1
#     except Exception as e:
#         logging.error(f"Error processing folder {folder_path}: {e}")

#     return {
#         "total_files": total_files,
#         "image_files": image_files
#     }

# def read_large_file(file_path):
#     """
#     使用生成器逐行读取大文件

#     参数：
#         file_path (str): 文件路径

#     返回：
#         generator: 文件内容行的生成器
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 yield line
#     except Exception as e:
#         logging.error(f"Error reading file {file_path}: {e}")
#         return

# def process_data_in_chunks(file_path, chunk_size, output_path):
#     """
#     分块处理大数据并写入中间文件

#     参数：
#         file_path (str): 输入文件路径
#         chunk_size (int): 每次处理的数据块大小
#         output_path (str): 输出文件路径
#     """
#     try:
#         data_iter = pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8')
#         for i, chunk in enumerate(data_iter):
#             output_chunk_path = f"{output_path}_part_{i}.csv"
#             chunk.to_csv(output_chunk_path, index=False, encoding='utf-8')
#     except Exception as e:
#         logging.error(f"Error processing chunks from file {file_path}: {e}")

# def copy_files_multithreaded(source_folder, target_folder):
#     """
#     使用多线程加速文件复制

#     参数：
#         source_folder (str): 源文件夹路径
#         target_folder (str): 目标文件夹路径
#     """
#     def copy_file(file_name):
#         try:
#             source_path = os.path.join(source_folder, file_name)
#             target_path = os.path.join(target_folder, file_name)
#             with open(source_path, 'rb') as src, open(target_path, 'wb') as tgt:
#                 tgt.write(src.read())
#         except Exception as e:
#             logging.error(f"Error copying file {file_name}: {e}")

#     try:
#         os.makedirs(target_folder, exist_ok=True)
#         files = os.listdir(source_folder)
#         with ThreadPoolExecutor() as executor:
#             executor.map(copy_file, files)
#     except Exception as e:
#         logging.error(f"Error setting up file copy: {e}")

# if __name__ == "__main__":
#     setup_logging()

#     # 提前定义需要处理的路径
#     folder_paths = [
#         "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new",
#         "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_original_rename",
#         "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_new",
#         "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_original_rename",
#         "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_new",
#         "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_original"
#     ]
#     large_file_path = "path/to/large_file.csv"
#     output_path = "path/to/output"
#     chunk_size = 1000  # 每块处理的行数
#     source_folder = "path/to/source_folder"
#     target_folder = "path/to/target_folder"

#     for folder_path in folder_paths:
#         if os.path.isdir(folder_path):
#             counts = count_files_and_images(folder_path)
#             print(f"路径: {folder_path}")
#             print(f"  总文件数: {counts['total_files']}")
#             print(f"  图片文件数: {counts['image_files']}")
#         else:
#             print(f"路径无效或不存在: {folder_path}")

#     process_data_in_chunks(large_file_path, chunk_size, output_path)
#     copy_files_multithreaded(source_folder, target_folder)



import os
import logging
from collections import defaultdict

def setup_logging():
    """设置日志记录配置"""
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='error.log',
        filemode='a',
        encoding='utf-8'
    )

def analyze_folder_ids(folder_path):
    """
    分析文件夹中的 ID 数据

    参数：
        folder_path (str): 文件夹路径

    返回：
        dict: 包含独有 ID 数量、平均每个 ID 的图片数量、最大和最小图片数量的统计
    """
    id_to_images = defaultdict(list)

    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if "_" in file:  # 判断文件名是否包含下划线
                    id_part = file.split("_")[0]  # 提取 ID 部分
                    id_to_images[id_part].append(file)
    except Exception as e:
        logging.error(f"Error processing folder {folder_path}: {e}")
        return {}

    # 统计信息
    unique_ids = len(id_to_images)
    image_counts = [len(images) for images in id_to_images.values()]
    avg_images_per_id = sum(image_counts) / unique_ids if unique_ids > 0 else 0
    max_images_per_id = max(image_counts) if image_counts else 0
    min_images_per_id = min(image_counts) if image_counts else 0

    return {
        "unique_ids": unique_ids,
        "avg_images_per_id": avg_images_per_id,
        "max_images_per_id": max_images_per_id,
        "min_images_per_id": min_images_per_id
    }

if __name__ == "__main__":
    setup_logging()

    # 定义需要处理的文件夹路径
    folder_paths = [
        "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new",
        "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_original_rename",
        "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_new",
        "/gpfs/home5/pzhang/beyond_images/datasets/MKG-Y_img_original_rename",
        "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_new",
        "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_original"
    ]

    # 对每个文件夹进行统计分析
    for folder_path in folder_paths:
        if os.path.isdir(folder_path):
            stats = analyze_folder_ids(folder_path)
            print(f"路径: {folder_path}")
            print(f"  独有的 ID 数量: {stats['unique_ids']}")
            print(f"  平均每个 ID 的图片数量: {stats['avg_images_per_id']:.2f}")
            print(f"  最大图片数量的 ID: {stats['max_images_per_id']}")
            print(f"  最小图片数量的 ID: {stats['min_images_per_id']}")
        else:
            print(f"路径无效或不存在: {folder_path}")
