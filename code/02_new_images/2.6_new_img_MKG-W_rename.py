import os
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rename_images.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# 参数配置
INPUT_FOLDER = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new"
INPUT_JSON = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new_summary.json"
OUTPUT_FOLDER = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new_rename"
LOG_LEVEL = logging.INFO

# 设置日志级别
logging.getLogger().setLevel(LOG_LEVEL)

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载JSON文件失败: {file_path}, 错误信息: {e}")
        return None

def rename_image(file_name, mapping, output_folder):
    """重命名单个图片文件"""
    try:
        # 提取图片ID部分
        image_id = os.path.splitext(file_name)[0]  # 去掉扩展名
        if image_id not in mapping:
            logging.warning(f"文件{file_name}未找到对应的映射关系，跳过处理。")
            return

        # 获取新的文件名
        qid = mapping[image_id]
        new_file_name = f"{qid}_{image_id.split('_')[1]}.jpg"

        # 复制文件到输出文件夹
        src_path = os.path.join(INPUT_FOLDER, file_name)
        dest_path = os.path.join(output_folder, new_file_name)
        shutil.copy(src_path, dest_path)
        logging.info(f"文件 {file_name} 成功重命名为 {new_file_name}")
    except Exception as e:
        logging.error(f"重命名文件 {file_name} 失败: {e}")

def main():
    """主函数"""
    logging.info("=== 开始处理图片重命名任务 ===")

    # 检查输入和输出文件夹
    if not os.path.exists(INPUT_FOLDER):
        logging.error(f"输入文件夹不存在: {INPUT_FOLDER}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        logging.info(f"输出文件夹已创建: {OUTPUT_FOLDER}")

    # 加载输入JSON文件
    data = load_json(INPUT_JSON)
    if data is None:
        logging.error("加载输入JSON文件失败，退出程序。")
        return

    # 构建映射关系
    mapping = {}
    for entry in data:
        try:
            image_id = entry["id"]
            qid = entry["wikidata_url"].split("/")[-1]
            mapping[image_id] = qid
        except KeyError as e:
            logging.warning(f"跳过无效条目: {entry}, 错误信息: {e}")

    logging.info(f"已加载映射关系，共计 {len(mapping)} 条。")

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')]
    logging.info(f"找到 {len(image_files)} 张图片需要处理。")

    # 使用多线程重命名图片文件
    with ThreadPoolExecutor(max_workers=8) as executor:
        for file_name in image_files:
            executor.submit(rename_image, file_name, mapping, OUTPUT_FOLDER)

    logging.info("=== 图片重命名任务完成 ===")

if __name__ == "__main__":
    main()
