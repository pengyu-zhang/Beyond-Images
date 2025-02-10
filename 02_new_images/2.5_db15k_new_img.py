import requests
from bs4 import BeautifulSoup
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from typing import List, Tuple, Dict, Any, Optional

def log_error(message: str):
    """
    记录错误日志
    """
    logging.error(message)

def log_failed(base_id: str, image_url: str, failed_images_log_file: str):
    """
    记录失败的图片下载
    """
    with open(failed_images_log_file, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{base_id} {image_url}\n")

def read_input_file(file_path: str) -> List[Tuple[str, str]]:
    """
    读取输入文件，返回 [(Wikidata URL, QID), ...] 的列表
    说明：
    1. 如果一行少于三列，则跳过（因为没有 Wikidata URL）。
    2. 取第三列作为 Wikidata URL（形如 http://www.wikidata.org/entity/Qxxxxxx）。
    3. 从该 URL 中提取 QID（Qxxxxxx）当做 base_id。
    """
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                # 如果一行少于 3 列，跳过
                if len(parts) < 3:
                    continue
                # 假设前两列可能是 dbpedia_url, label; 第三列才是 wikidata_url
                dbpedia_url, some_label, wikidata_url = parts[0], parts[1], parts[2]
                # 从 wikidata_url 提取 QID
                # 比如 wikidata_url = "http://www.wikidata.org/entity/Q6412036"
                # 那么 qid = "Q6412036"
                qid = wikidata_url.split('/')[-1].strip()
                # 只要 qid 不是空，说明是合法的
                if qid:
                    pairs.append((wikidata_url, qid))
    except Exception as e:
        log_error(f"Failed to read input file: {e}")

    return pairs

def clean_text(text: str) -> str:
    """
    清理文本，去除多余的空格和换行
    """
    return ' '.join(text.split())

def extract_table_data(soup: BeautifulSoup, base_id: str, page_url: str) -> Dict[str, str]:
    """
    提取页面中 Summary 表格的内容
    """
    table_data = {}
    tables = soup.find_all('table')

    for table in tables:
        for row in table.find_all('tr'):
            header_cell = row.find('th') or row.find('td', {'class': 'fileinfo-paramfield'})
            # 注意这里：row.find_all('td') 需要防止越界（不一定所有行都俩td）
            tds = row.find_all('td')
            if header_cell and len(tds) > 0:
                # 当 row.find('th') 存在时，一般 value_cell 就在后面的 td
                # 当 row.find('th') 不存在，但存在 fileinfo-paramfield，也可以在 tds[1]
                # 为了稳妥，这里做个判断
                if len(tds) > 1:
                    value_cell = tds[1]
                else:
                    value_cell = tds[0]

                header = clean_text(header_cell.text.strip())
                value = clean_text(value_cell.text.strip())
                table_data[header] = value

    if not table_data:
        log_error(f"No suitable table found for {base_id} {page_url}")

    return table_data

def extract_image_url(soup: BeautifulSoup, base_id: str, page_url: str) -> Optional[str]:
    """
    提取页面中的图片 URL
    """
    full_image_link_div = soup.find('div', {'class': 'fullImageLink'})
    if full_image_link_div:
        main_image_tag = full_image_link_div.find('img')
        if main_image_tag and 'src' in main_image_tag.attrs:
            src = main_image_tag['src']
            # 避免以 // 开头的情况，补全协议
            if src.startswith('//'):
                src = 'https:' + src
            return src
    log_error(f"No fullImageLink div or no valid img src found for {base_id} {page_url}")
    return None

def extract_image_url_and_table(page_url: str, base_id: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    提取图片 URL 和页面中的 Summary 表格内容
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        image_url = extract_image_url(soup, base_id, page_url)
        table_data = extract_table_data(soup, base_id, page_url)

        return image_url, table_data
    except requests.RequestException as e:
        log_error(f"Failed to extract image URL and table for {base_id} {page_url}: {e}")
        return None, {}

def download_image(image_url: str, save_path: str, image_name: str) -> Optional[str]:
    """
    下载图片并保存到指定路径
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        image_path = os.path.join(save_path, image_name)
        with open(image_path, 'wb') as file:
            file.write(response.content)
        return image_path
    except Exception as e:
        log_error(f"Failed to download image {image_url}: {e}")
        return None

def find_image_details_pages(wikipedia_url: str) -> List[str]:
    """
    查找英语 Wikipedia 页面中的图片详情页面链接
    """
    try:
        response = requests.get(wikipedia_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        image_links = []
        for a_tag in soup.find_all('a', {'class': 'mw-file-description'}):
            if 'href' in a_tag.attrs:
                img_page_url = urljoin(wikipedia_url, a_tag['href'])
                image_links.append(img_page_url)

        return image_links
    except Exception as e:
        log_error(f"Failed to find image details pages for URL {wikipedia_url}: {e}")
        return []

def get_english_wikipedia_url(qid: str) -> Optional[str]:
    """
    根据 QID 获取英语 Wikipedia 的 URL
    """
    try:
        wikidata_url = f"http://www.wikidata.org/wiki/{qid}"
        response = requests.get(wikidata_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        sitelinks_div = soup.find('div', {'class': 'wikibase-sitelinklistview'})
        if sitelinks_div:
            enwiki_link = sitelinks_div.find('li', {'class': 'wikibase-sitelinkview-enwiki'})
            if enwiki_link:
                return enwiki_link.find('a')['href']
        return None
    except Exception as e:
        log_error(f"Failed to get English Wikipedia URL for QID {qid}: {e}")
        return None

def process_url_pair(url_pair: Tuple[str, str], save_path: str, intermediate_data: List[Dict[str, Any]], failed_images_log_file: str):
    """
    处理单个 (wikidata_url, qid) 对，提取图片和表格信息并保存
    """
    base_url, base_id = url_pair  # base_url = http://www.wikidata.org/entity/Qxxxxx, base_id = Qxxxxx

    # 如果需要在这里再一次 split() 也行，但实际上 base_id 已经是 Qxxxxx
    # qid = base_url.split('/')[-1]
    qid = base_id  # 直接用即可

    wikipedia_url = get_english_wikipedia_url(qid)
    if not wikipedia_url:
        log_error(f"Failed to get English Wikipedia URL for {qid}")
        return

    image_pages = find_image_details_pages(wikipedia_url)
    for page_index, page_url in enumerate(image_pages):
        image_url, table_data = extract_image_url_and_table(page_url, base_id)
        if image_url:
            # 文件名使用 base_id 再加索引
            image_name = f"{base_id}_{page_index+1}.jpg"
            downloaded_path = download_image(image_url, save_path, image_name)
            if downloaded_path is None:
                # 如果下载失败，记录一下
                log_failed(base_id, image_url, failed_images_log_file)

        entry = {
            "id": f"{base_id}_{page_index+1}",
            "wikidata_url": base_url,
            "page_url": page_url,
            "image_url": image_url,
            "summary": table_data
        }
        intermediate_data.append(entry)

def save_table_data(json_output_path: str, all_data: List[Dict[str, Any]]):
    """
    保存数据到 JSON 文件
    原始代码的排序方式是按“数字_数字”排序，但现在 base_id 是 Qxxxxx，
    如果你想保留就得改排序方式。这里先改成简单的字符串排序即可。
    """
    try:
        # 简单按字符串排序
        all_data.sort(key=lambda entry: entry['id'])
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(all_data, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        log_error(f"Failed to save JSON data: {e}")

def main(input_file: str, save_path: str, json_output_path: str, start_index: int, end_index: int, failed_images_log_file: str):
    """
    主函数，协调各模块功能
    """
    url_id_pairs = read_input_file(input_file)  # 新的读取逻辑
    selected_pairs = url_id_pairs[start_index:end_index]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    intermediate_data = []
    # 保持线程并发逻辑不变
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_url_pair, pair, save_path, intermediate_data, failed_images_log_file)
            for pair in selected_pairs
        ]
        for future in as_completed(futures):
            future.result()

    save_table_data(json_output_path, intermediate_data)

if __name__ == "__main__":
    # 路径配置（示例）
    input_file = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_ent_links_wikidata.txt"
    save_path = "/gpfs/scratch1/nodespecific/int5/pzhang/beyond_images/datasets/DB15K_img_new"
    json_output_path = "/gpfs/home5/pzhang/beyond_images/datasets/DB15K_img_new_summary.json"
    ERROR_LOG_FILE = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_img_new_error_log.txt"
    FAILED_IMAGES_LOG_FILE = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_img_new_failed_images_log.txt"

    # 日志配置
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=ERROR_LOG_FILE,
        filemode='a'
    )

    # 程序运行参数（示例）
    start_index = 0
    end_index = 12845

    # 主程序运行
    main(input_file, save_path, json_output_path, start_index, end_index, FAILED_IMAGES_LOG_FILE)
