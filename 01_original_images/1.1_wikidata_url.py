import os
import requests
from bs4 import BeautifulSoup
import time

# Only for MKG-Y dataset!!!
# 输入文件和输出文件路径
input_file = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_ent_links.txt"
output_file = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_ent_links_wikidata.txt"
error_file = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-Y_ent_links_wikidata_errors.txt"

# HTTP 请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def fetch_wikidata_url(dbpedia_url, retries=3, delay=2):
    """从 dbpedia_url 的 HTML 页面中提取 wikidata_url，支持重试"""
    for attempt in range(retries):
        try:
            response = requests.get(dbpedia_url, headers=headers, timeout=10)
            response.raise_for_status()  # 检查 HTTP 请求是否成功
            soup = BeautifulSoup(response.text, "html.parser")

            # 查找 rel="owl:sameAs" 的 <a> 标签
            links = soup.find_all("a", {"rel": "owl:sameAs"})
            for link in links:
                href = link.get("href")
                if "wikidata.org" in href:  # 检查链接是否包含 wikidata
                    return href
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(delay)  # 延迟后重试
                delay *= 2  # 每次失败后加倍延迟
    return None


def process_line(line, retries=3, delay=2):
    """处理单行数据并返回结果"""
    parts = line.strip().split("\t")  # 获取原始行的各部分
    dbpedia_url = parts[0]  # 第一个部分是 dbpedia_url
    dbpedia_id = parts[1] if len(parts) > 1 else ""  # 第二个部分是 dbpedia_id
    wikidata_url = fetch_wikidata_url(dbpedia_url, retries, delay)  # 获取 wikidata_url
    return dbpedia_url, dbpedia_id, wikidata_url


def process_ent_links(input_file, output_file, error_file, log_frequency=50):
    """读取 ent_links.txt，处理并输出带有 wikidata_url 的文件，同时记录错误"""
    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile, \
            open(error_file, "w", encoding="utf-8") as errfile:
        lines = infile.readlines()
        total = len(lines)
        successes = 0
        failures = 0

        for idx, line in enumerate(lines, start=1):
            dbpedia_url, dbpedia_id, wikidata_url = process_line(line)

            # 写入输出文件
            if wikidata_url:
                outfile.write(f"{dbpedia_url}\t{dbpedia_id}\t{wikidata_url}\n")
                successes += 1
            else:
                outfile.write(f"{dbpedia_url}\t{dbpedia_id}\t\n")  # 未找到时写入空值
                errfile.write(f"{dbpedia_url}\n")  # 将失败的 URL 写入错误文件
                failures += 1

            # 输出进度信息
            if idx % log_frequency == 0:
                print(f"已处理 {idx}/{total} 个样本，成功 {successes} 个，失败 {failures} 个")


def main():
    process_ent_links(input_file, output_file, error_file)
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"失败记录已保存到 {error_file}")


if __name__ == "__main__":
    main()
