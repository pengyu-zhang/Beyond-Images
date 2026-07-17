import os
import logging
import shutil

# 配置参数
input_file1 = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_ent_links_wikidata.txt"
input_file2 = "/gpfs/scratch1/nodespecific/int5/pzhang/beyond_images/datasets/DB15K_img_original"
output_file3 = "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_original"
output_file4 = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_log_error.txt"  # 未匹配日志
output_file5 = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_log_success.txt"  # 成功匹配日志
output_file6 = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_log_fail.txt"  # 未匹配实体日志
log_level = "INFO"

# 配置日志
def configure_logging():
    """配置日志"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 日志文件处理器（简化错误日志）
    file_handler = logging.FileHandler(output_file4, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # 简化日志格式

    # 移除所有现有处理器并添加文件处理器
    logger.handlers = []
    logger.addHandler(file_handler)

configure_logging()

# 创建输出目录
os.makedirs(output_file3, exist_ok=True)

def parse_ent_links(file_path):
    """解析输入文件1，返回dbpedia_name到wikidata_qid的映射"""
    dbpedia_to_qid = {}
    ent_links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    dbpedia_url, dbpedia_name, wikidata_url = parts
                    wikidata_qid = wikidata_url.split('/')[-1]
                    dbpedia_to_qid[dbpedia_name] = wikidata_qid  # 使用dbpedia_name作为键
                    ent_links.append((dbpedia_name, dbpedia_url, wikidata_url))
        print(f"Parsed dbpedia_to_qid: {dbpedia_to_qid}")
    except Exception as e:
        logging.error(f"读取文件 {file_path} 失败: {e}")
    return dbpedia_to_qid, ent_links

def save_logs(success_log, fail_log):
    """保存成功匹配和未匹配实体日志"""
    # 保存成功日志
    with open(output_file5, 'w', encoding='utf-8') as success_file:
        for dbpedia_suffix, dbpedia_url, wikidata_url in success_log:
            success_file.write(f"{dbpedia_url}\t{wikidata_url}\n")  # 使用 \t 分隔

    # 保存未匹配日志
    with open(output_file6, 'w', encoding='utf-8') as fail_file:
        for dbpedia_suffix, dbpedia_url, wikidata_url in fail_log:
            fail_file.write(f"{dbpedia_url}\t{wikidata_url}\n")  # 使用 \t 分隔

def rename_and_copy_images(input_folder, dbpedia_to_qid, ent_links, output_folder):
    """遍历文件夹2，重命名图片并复制到输出文件夹"""
    total_folders = 0
    matched_folders = []
    unmatched_folders = []

    success_log = []  # 存储成功匹配的完整实体信息
    fail_log = ent_links.copy()  # 初始为所有实体，匹配成功后从中移除

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        total_folders += 1
        matched = False

        # 直接匹配文件夹名
        if folder_name in dbpedia_to_qid:
            matched = True
            qid = dbpedia_to_qid[folder_name]
            matched_entity = next((entry for entry in ent_links if entry[0] == folder_name), None)
            if matched_entity:
                success_log.append(matched_entity)  # 添加到成功日志
                fail_log.remove(matched_entity)  # 从未匹配中移除
        else:
            # 如果包含 __ 尝试替换第一个 _ 为 :
            if "__" in folder_name:
                modified_name = folder_name.replace("__", ":_", 1)
                if modified_name in dbpedia_to_qid:
                    matched = True
                    qid = dbpedia_to_qid[modified_name]
                    matched_entity = next((entry for entry in ent_links if entry[0] == modified_name), None)
                    if matched_entity:
                        success_log.append(matched_entity)  # 添加到成功日志
                        fail_log.remove(matched_entity)  # 从未匹配中移除

        if matched:
            matched_folders.append(folder_name)
            try:
                for idx, image_name in enumerate(os.listdir(folder_path)):
                    image_path = os.path.join(folder_path, image_name)
                    if not os.path.isfile(image_path):
                        continue
                    new_image_name = f"{qid}_{idx}.jpg"
                    new_image_path = os.path.join(output_folder, new_image_name)
                    shutil.copy2(image_path, new_image_path)
            except Exception as e:
                logging.error(f"处理文件夹 {folder_name} 时出错: {e}")
        else:
            unmatched_folders.append(folder_name)

    # 保存简化的错误日志
    with open(output_file4, 'w', encoding='utf-8') as error_file:
        error_file.write(f"共有 {total_folders} 个子文件夹，成功识别 {len(matched_folders)} 个文件夹，不成功识别 {len(unmatched_folders)} 个文件夹\n")
        for folder in unmatched_folders:
            error_file.write(folder + '\n')

    # 保存成功匹配和未匹配日志
    save_logs(success_log, fail_log)

def main():
    # 解析输入文件1
    dbpedia_to_qid, ent_links = parse_ent_links(input_file1)
    if not dbpedia_to_qid:
        logging.error("无法解析输入文件1，退出程序。")
        return

    # 处理文件夹和图片
    rename_and_copy_images(input_file2, dbpedia_to_qid, ent_links, output_file3)
    logging.info("图片重命名和复制完成。")

if __name__ == "__main__":
    main()
