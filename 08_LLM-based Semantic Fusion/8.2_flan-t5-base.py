import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os
import argparse # <--- 新增部分

# --- 1. 全局配置 ---
INPUT_FILES = ["/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_img_new_summary_blip.json", "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_original_img_summary_blip.json"]
OUTPUT_FILE = "/gpfs/work5/0/prjs1145/code/beyond_images/08_t5/DB15K_t5-base_summaries.json"
MODEL_NAME = "google/flan-t5-base"  #/gpfs/work5/0/prjs1145/code/beyond_images/google_flan-t5-base/"  # "google/flan-t5-large"  或者 "google/flan-t5-xl"  "google/flan-t5-base"

# 设定描述数量的上限
MAX_DESCRIPTIONS_PER_ENTITY = 500
# 设定哪个文件中的描述具有优先权
PRIORITY_FILE = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_original_img_summary_blip.json"

def load_and_merge_data(file_paths: list) -> dict:
    """
    加载所有输入JSON文件，并根据来源文件对描述进行分组。
    """
    print("Step 1: Loading and merging data from input files...")
    merged_data = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: Input file not found at '{file_path}'. Skipping.")
            continue
            
        print(f"-> Reading '{file_path}'")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entity_name, entity_data in data.items():
            if entity_name not in merged_data:
                merged_data[entity_name] = {
                    key: value for key, value in entity_data.items() if key != 'images'
                }
                # <--- 修改部分：初始化数据结构以按来源存储描述 --->
                merged_data[entity_name]['_descriptions_by_source'] = {}

            # <--- 修改部分：将描述添加到对应来源文件的列表中 --->
            images_obj = entity_data.get("images")
            if isinstance(images_obj, dict):
                # 确保当前文件的描述列表已初始化
                if file_path not in merged_data[entity_name]['_descriptions_by_source']:
                    merged_data[entity_name]['_descriptions_by_source'][file_path] = []
                
                for key, value in images_obj.items():
                    if isinstance(value, dict) and "image_description_detail" in value:
                        desc = value["image_description_detail"]
                        if desc:
                            merged_data[entity_name]['_descriptions_by_source'][file_path].append(desc)
    
    print(f"Data merging complete. Found {len(merged_data)} unique entities.")
    return merged_data


def preprocess_descriptions(descriptions: list) -> list:
    """
    对描述列表进行去重和简单清洗。
    """
    unique_descs = list(dict.fromkeys(descriptions))
    cleaned_descs = []
    for desc in unique_descs:
        if not desc or desc.isspace():
            continue
        if len(desc.split()) < 3:
            continue
        if len(set(desc.split())) < len(desc.split()) / 2 and len(desc.split()) > 5:
            continue
        cleaned_descs.append(desc)
    return cleaned_descs


def get_prioritized_descriptions(
    descriptions_by_source: dict, 
    priority_file: str, 
    all_input_files: list, 
    max_limit: int
) -> list:
    """
    根据文件优先级选择描述，直到达到上限。
    """
    final_descriptions = []

    # 1. 首先处理优先文件
    priority_descs = descriptions_by_source.get(priority_file, [])
    cleaned_priority_descs = preprocess_descriptions(priority_descs)
    final_descriptions.extend(cleaned_priority_descs)

    # 如果已经超出或达到上限，直接返回
    if len(final_descriptions) >= max_limit:
        return final_descriptions[:max_limit]

    # 2. 接着处理其他文件
    other_files = [f for f in all_input_files if f != priority_file]
    for file in other_files:
        other_descs = descriptions_by_source.get(file, [])
        cleaned_other_descs = preprocess_descriptions(other_descs)
        
        needed = max_limit - len(final_descriptions)
        final_descriptions.extend(cleaned_other_descs[:needed])
        
        if len(final_descriptions) >= max_limit:
            break # 达到上限，停止处理

    return final_descriptions


class T5Fuser:
    """
    封装T5模型，用于生成融合描述。（最终优化版）
    """
    def __init__(self, model_name: str):
        print("\nStep 2: Initializing T5 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # <--- 修改部分 1: 采用一个更注重细节和融合的Prompt模板 --->
        self.prompt_template = """Your task is to integrate the following list of visual descriptions for the entity '{entity_name}' into a rich, detailed, and coherent summary paragraph. Capture as many key details as possible, such as objects, colors, actions, and settings. Your final output must be a single paragraph, not a list.

List of descriptions to summarize:
{descriptions}

Detailed Summary Paragraph:"""
        
        print("Model initialized successfully.")

    def generate(self, entity_name: str, descriptions: list) -> str:
        """
        根据实体名和描述列表生成融合文本。
        """
        # <--- 修改部分 2: 使用换行符分隔描述，清晰且不会误导模型 --->
        formatted_descriptions = "\n".join(descriptions)
        
        prompt = self.prompt_template.format(
            entity_name=entity_name, 
            descriptions=formatted_descriptions
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # <--- 修改部分 3: 移除了过于严格的 no_repeat_ngram_size 参数 --->
        outputs = self.model.generate(
            input_ids,
            max_length=384,  # 稍微增加最大长度以容纳更多细节
            num_beams=4,
            early_stopping=True
        )
        
        fused_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fused_text


def main():
    """
    主执行函数
    """
    # <--- 新增/修改 开始 --->
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="Fuse entity descriptions from JSON files using a T5 model."
    )
    parser.add_argument(
        "samples",
        nargs="?",  # 使参数变为可选
        default="all",
        help="Number of entities to process (e.g., 10, 100). Use 'all' to process every entity. Defaults to 'all'."
    )
    args = parser.parse_args()
    num_to_process_str = args.samples
    # <--- 新增/修改 结束 --->

    # 步骤 1: 加载并合并数据
    merged_data = load_and_merge_data(INPUT_FILES)
    
    # <--- 新增/修改 开始 --->
    # 根据命令行参数决定要处理的数据子集
    items_to_process = list(merged_data.items())
    
    if num_to_process_str.lower() != 'all':
        try:
            limit = int(num_to_process_str)
            items_to_process = items_to_process[:limit]
            print(f"\n-> Running in test mode. Processing a subset of {len(items_to_process)} entities.")
        except ValueError:
            print(f"\nError: Invalid sample count '{num_to_process_str}'. Please provide an integer or 'all'.")
            return # 无效输入则退出
    else:
        print("\n-> Running in full mode. Processing all entities.")
    # <--- 新增/修改 结束 --->

    # 步骤 2: 初始化模型
    fuser = T5Fuser(MODEL_NAME)
    
    # 步骤 3: 迭代处理每个实体并构建最终输出
    print("\nStep 3: Processing entities and generating descriptions...")
    final_output = {}

    # 使用 tqdm 创建进度条，并对筛选后的数据进行处理
    # <--- 修改部分 --->
    for entity_name, entity_data in tqdm(items_to_process, desc="Fusing Descriptions"):
        final_output[entity_name] = {
            key: value for key, value in entity_data.items() if not key.startswith('_')
        }
        final_output[entity_name]['images'] = {}

        # <--- 修改部分：调用新的优先级筛选函数 --->
        descriptions_by_source = entity_data.get('_descriptions_by_source', {})
        
        descriptions_for_prompt = get_prioritized_descriptions(
            descriptions_by_source,
            PRIORITY_FILE,
            INPUT_FILES,
            MAX_DESCRIPTIONS_PER_ENTITY
        )
        # <--- 修改结束 --->

        if descriptions_for_prompt:
            try:
                fused_text = fuser.generate(entity_name, descriptions_for_prompt)
                final_output[entity_name]['images']['images_t5_descriptions'] = fused_text
            except Exception as e:
                print(f"\nError processing entity '{entity_name}': {e}")
    
    # 步骤 4: 保存结果
    output_filename = OUTPUT_FILE
    # 如果是测试模式，可以保存为不同的文件名以防覆盖
    if num_to_process_str.lower() != 'all':
        base, ext = os.path.splitext(OUTPUT_FILE)
        output_filename = f"{base}_sample_{len(items_to_process)}{ext}"
    
    print(f"\nStep 4: Saving fused data to '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print("✅ All tasks completed successfully!")


if __name__ == "__main__":
    main()