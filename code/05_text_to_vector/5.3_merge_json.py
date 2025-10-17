import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("merge_json.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# 输入与配置
FILE1_PATH = "/gpfs/work5/0/prjs1145/code/mmkg-mygo/tokens/MKG-W-textual.json"  # 文件1路径/gpfs/work5/0/prjs1145/code/mmkg-mygo/tokens/DB15K-textual-v2.json
FILE2_PATH = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_vector_qid_based/MKG-W_orig_and_new_img_blip.json"  # 文件2路径
OUTPUT_PATH = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_vector_model_based/MKG-W_orig_and_new_img_blip.json"  # 输出文件路径
CHUNK_SIZE = 100  # 数据分块大小
THREADS = 4  # 多线程数量

def load_json_file(filepath):
    """加载JSON文件为字典"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logging.info(f"成功加载文件: {filepath}")
        return data
    except Exception as e:
        logging.error(f"加载文件失败: {filepath}, 错误信息: {e}")
        return None

def save_json_file(data, filepath):
    """保存字典为JSON文件，符合文件1的格式（带空格的紧凑单行格式）"""
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, separators=(', ', ': '))
        logging.info(f"成功保存文件: {filepath}")
    except Exception as e:
        logging.error(f"保存文件失败: {filepath}, 错误信息: {e}")

def process_tokens(file1_tokens, file2_tokens):
    """处理token序列，去掉file2中开头的101和结尾的102，并合并"""
    filtered_file2_tokens = [t for t in file2_tokens if t not in (101, 102)]
    result_tokens = file1_tokens[:-1] + filtered_file2_tokens + [102]
    return result_tokens

def merge_data(file1, file2):
    """合并文件1和文件2的内容"""
    result = {}
    for key, file1_tokens in file1.items():
        if key in file2:
            result[key] = process_tokens(file1_tokens, file2[key])
        else:
            result[key] = file1_tokens
    for key, file2_tokens in file2.items():
        if key not in file1:
            result[key] = [101] + [t for t in file2_tokens if t not in (101, 102)] + [102]
    return result

def process_large_json(file1_path, file2_path, output_path, chunk_size):
    """分块处理大型JSON文件"""
    try:
        file1 = load_json_file(file1_path)
        file2 = load_json_file(file2_path)

        if file1 is None or file2 is None:
            logging.error("文件加载失败，终止处理。")
            return

        keys = list(file1.keys()) + list(set(file2.keys()) - set(file1.keys()))
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]

        final_result = {}
        for chunk in chunks:
            partial_file1 = {k: file1[k] for k in chunk if k in file1}
            partial_file2 = {k: file2[k] for k in chunk if k in file2}
            merged_chunk = merge_data(partial_file1, partial_file2)
            final_result.update(merged_chunk)

        save_json_file(final_result, output_path)
        logging.info(f"合并完成，结果保存在: {output_path}")

    except Exception as e:
        logging.error(f"处理JSON文件时出错: {e}")

def multi_thread_file_copy(file_paths, output_dir):
    """使用多线程加速文件复制"""
    os.makedirs(output_dir, exist_ok=True)

    def copy_file(file_path):
        try:
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(file_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            logging.info(f"成功复制文件: {file_path}")
        except Exception as e:
            logging.error(f"复制文件失败: {file_path}, 错误信息: {e}")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        executor.map(copy_file, file_paths)

if __name__ == "__main__":
    # 合并文件
    process_large_json(FILE1_PATH, FILE2_PATH, OUTPUT_PATH, CHUNK_SIZE)

    # 示例文件复制操作
    file_paths = [FILE1_PATH, FILE2_PATH]
    output_dir = "backup_files"
    multi_thread_file_copy(file_paths, output_dir)
