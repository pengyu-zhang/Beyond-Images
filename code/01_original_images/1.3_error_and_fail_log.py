import re
from difflib import SequenceMatcher

# 文件路径
error_log_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_log_error.txt"
fail_log_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_log_fail.txt"
output_match_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_match.txt"

def load_error_log(file_path):
    """加载 error_log 文件，返回文件夹名字列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def load_fail_log(file_path):
    """加载 fail_log 文件，返回 dbpedia_url 和 wikidata_url 的列表"""
    fail_entries = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                fail_entries.append((parts[0], parts[1]))
    return fail_entries

def clean_name(name):
    """清理名字，去掉特殊符号"""
    # 替换常见符号
    name = re.sub(r"[._,()]", " ", name)  # 替换 .,_,(,) 为空格
    name = re.sub(r"\s+", " ", name).strip()  # 去除多余空格
    return name.lower()  # 返回小写以便比较

def find_best_match(error_name, fail_entries):
    """使用模糊匹配找到最佳匹配的 dbpedia_url 和 wikidata_url"""
    best_match = None
    highest_ratio = 0.0

    for dbpedia_url, wikidata_url in fail_entries:
        # 提取 dbpedia_url 的最后部分
        dbpedia_name = dbpedia_url.split("/")[-1]
        # 清理名字用于比较
        cleaned_error_name = clean_name(error_name)
        cleaned_dbpedia_name = clean_name(dbpedia_name)

        # 计算相似度
        similarity_ratio = SequenceMatcher(None, cleaned_error_name, cleaned_dbpedia_name).ratio()
        if similarity_ratio > highest_ratio and similarity_ratio > 0.8:  # 阈值为 0.8
            highest_ratio = similarity_ratio
            best_match = (dbpedia_url, wikidata_url)

    return best_match

def match_error_to_fail(error_log, fail_log, output_path):
    """根据 error_log 和 fail_log 进行模糊匹配，保存匹配结果"""
    fail_entries = load_fail_log(fail_log)
    error_entries = load_error_log(error_log)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for error_name in error_entries:
            match = find_best_match(error_name, fail_entries)
            if match:
                dbpedia_url, wikidata_url = match
                output_file.write(f"{error_name}\t{dbpedia_url}\t{wikidata_url}\n")
            else:
                output_file.write(f"{error_name}\t\n")

def main():
    match_error_to_fail(error_log_path, fail_log_path, output_match_path)
    print(f"匹配结果已保存到 {output_match_path}")

if __name__ == "__main__":
    main()
