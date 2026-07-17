import os
import json
import logging
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def read_file(file_path, separator, columns, skip_header=False):
    """
    通用文件读取函数，返回 DataFrame。
    """
    try:
        data = pd.read_csv(
            file_path,
            sep=separator,
            names=columns,
            header=0 if skip_header else None,
            encoding="utf-8",
            engine="python"  # 使用 Python 引擎，避免复杂分隔符导致警告
        )
        logging.info(f"成功读取文件: {file_path}")
        return data
    except Exception as e:
        logging.error(f"读取文件失败: {file_path}, 错误信息: {e}")
        return pd.DataFrame()

def merge_data(ent_links, img_details):
    """
    根据要求整合数据。
    """
    result = {}

    # 去掉 ent_links 的前缀
    ent_links["entity_name"] = ent_links["dbpedia_url"].str.replace("http://dbpedia.org/resource/", "", regex=False)
    ent_links["entity_qid"] = ent_links["wikidata_url"].str.replace("http://www.wikidata.org/entity/", "", regex=False)

    # 遍历每个实体
    for _, row in ent_links.iterrows():
        entity_name = row["entity_name"]
        entity_qid = row["entity_qid"]
        dbpedia_url = row["dbpedia_url"]
        wikidata_url = row["wikidata_url"]

        # 筛选图片详情
        img_det = img_details[img_details["image_name"].str.startswith(entity_qid)]

        images = {}
        merged_descriptions = []

        for _, det_row in img_det.iterrows():
            image_name = det_row["image_name"].rsplit(".", 1)[0]  # 去掉.jpg
            images[image_name] = {
                "image_description_detail": det_row["image_description_detail"]
            }
            merged_descriptions.append(det_row["image_description_detail"])

        # 添加合并后的描述，过滤 None 并转换为字符串
        merged_descriptions = [desc for desc in merged_descriptions if desc is not None]
        merged_descriptions = [str(desc) for desc in merged_descriptions]
        images["merged_descriptions"] = " ".join(merged_descriptions)

        # 整合数据
        result[entity_name] = {
            "entity_name": entity_name,
            "entity_qid": entity_qid,
            "dbpedia_url": dbpedia_url,
            "wikidata_url": wikidata_url,
            "images": images
        }

    return result

def main():
    # 输入文件路径
    file1 = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-W_ent_links.txt"
    file2 = "/gpfs/home5/pzhang/beyond_images/datasets/log/MKG-W_img_new_descriptions.txt"
    output_file = "/gpfs/home5/pzhang/beyond_images/datasets/MKG-W_img_new_summary_blip.json"

    # 读取文件
    ent_links = read_file(file1, "\t", ["dbpedia_url", "wikidata_url"])
    img_details = read_file(file2, ": ", ["image_name", "image_description_detail"], skip_header=True)

    # 数据整合
    result = merge_data(ent_links, img_details)

    # 写入输出文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)  # 输出为格式化的 JSON 文件
        logging.info(f"成功写入输出文件: {output_file}")
    except Exception as e:
        logging.error(f"写入输出文件失败: {output_file}, 错误信息: {e}")

if __name__ == "__main__":
    main()
