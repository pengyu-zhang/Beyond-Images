import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("generate_files_log.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置类
class Config:
    MODEL_NAME = "bert-base-uncased"
    JSON_INPUT_PATH = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_img_new_summary_blip.json"
    OUTPUT_FOLDER_PATH = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_vector_qid_based"
    JSON_OUTPUT_FILE = "DB15K_new_img_blip.json"

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModel.from_pretrained(Config.MODEL_NAME)

def tokenize_text(text):
    """
    使用 BERT tokenizer 对文本进行分词，返回 token ID 列表
    """
    return tokenizer.encode(text, truncation=True, padding=False, max_length=512)

def generate_json_file(config):
    """
    根据输入 JSON 文件生成 JSON 文件
    """
    try:
        # 创建输出文件夹
        output_folder = Path(config.OUTPUT_FOLDER_PATH)
        output_folder.mkdir(parents=True, exist_ok=True)

        json_output_path = output_folder / config.JSON_OUTPUT_FILE

        logging.info(f"Loading JSON file: {config.JSON_INPUT_PATH}")
        with open(config.JSON_INPUT_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        json_output_data = {}

        logging.info("Processing entities...")

        for entity, details in json_data.items():
            qid = details.get("entity_qid")
            images = details.get("images", {})
            merged_descriptions = images.get("merged_descriptions", "")

            # 跳过描述为空的实体
            if not merged_descriptions:
                continue

            # JSON 数据（使用 BERT tokenizer）
            tokenized_output = tokenize_text(merged_descriptions)
            json_output_data[f"http://www.wikidata.org/entity/{qid}"] = tokenized_output

        # 写入 JSON 文件
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_output_data, json_file, ensure_ascii=False, indent=None, separators=(',', ':'))
        logging.info(f"JSON file saved to: {json_output_path}")

        logging.info("JSON file generated successfully.")
    except Exception as e:
        logging.error(f"Error generating JSON file: {e}")

def main():
    """
    主程序入口
    """
    generate_json_file(Config)

if __name__ == "__main__":
    main()
