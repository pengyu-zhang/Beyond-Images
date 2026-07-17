import json
import torch
# <--- 1. 修改导入的类，因为Llama/Mistral是Causal LM模型 --->
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import argparse

# --- 1. 全局配置 ---
INPUT_FILES = ["/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_img_new_summary_blip.json", "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_original_img_summary_blip.json"]
OUTPUT_FILE = "/gpfs/work5/0/prjs1145/code/beyond_images/08_t5/DB15K_Llama-3.1-8B_summaries.json" # 建议修改输出文件名

# <--- 2. 更换为新的模型名称 --->
# 您可以在这里切换 Llama-3 或 Mistral
# MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"  /gpfs/work5/0/prjs1145/code/beyond_images/meta-llama_Llama-3.1-8B-Instruct/
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  "/gpfs/work5/0/prjs1145/code/beyond_images/mistralai_Mistral-7B-Instruct-v0.3/"
MODEL_NAME = "/gpfs/work5/0/prjs1145/code/beyond_images/meta-llama_Llama-3.1-8B-Instruct/"

# 设定描述数量的上限
MAX_DESCRIPTIONS_PER_ENTITY = 500 # 这个上限对于Llama/Mistral的超长上下文来说依然有用，可以防止极端情况
# 设定哪个文件中的描述具有优先权
PRIORITY_FILE = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/DB15K_original_img_summary_blip.json"


# ... load_and_merge_data, preprocess_descriptions, get_prioritized_descriptions ...
# ... 这三个函数完全不需要任何改动，保持原样即可 ...
def load_and_merge_data(file_paths: list) -> dict:
    """加载所有输入JSON文件，并根据来源文件对描述进行分组。"""
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
                merged_data[entity_name] = {key: value for key, value in entity_data.items() if key != 'images'}
                merged_data[entity_name]['_descriptions_by_source'] = {}
            images_obj = entity_data.get("images")
            if isinstance(images_obj, dict):
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
    """对描述列表进行去重和简单清洗。"""
    unique_descs = list(dict.fromkeys(descriptions))
    cleaned_descs = []
    for desc in unique_descs:
        if not desc or desc.isspace(): continue
        if len(desc.split()) < 3: continue
        if len(set(desc.split())) < len(desc.split()) / 2 and len(desc.split()) > 5: continue
        cleaned_descs.append(desc)
    return cleaned_descs

def get_prioritized_descriptions(descriptions_by_source: dict, priority_file: str, all_input_files: list, max_limit: int) -> list:
    """根据文件优先级选择描述，直到达到上限。"""
    final_descriptions = []
    priority_descs = descriptions_by_source.get(priority_file, [])
    cleaned_priority_descs = preprocess_descriptions(priority_descs)
    final_descriptions.extend(cleaned_priority_descs)
    if len(final_descriptions) >= max_limit:
        return final_descriptions[:max_limit]
    other_files = [f for f in all_input_files if f != priority_file]
    for file in other_files:
        other_descs = descriptions_by_source.get(file, [])
        cleaned_other_descs = preprocess_descriptions(other_descs)
        needed = max_limit - len(final_descriptions)
        final_descriptions.extend(cleaned_other_descs[:needed])
        if len(final_descriptions) >= max_limit:
            break
    return final_descriptions

# <--- 3. 重写模型封装类，以适应新的模型架构 --->
# 请用这个新版本替换您代码中现有的整个 ModelFuser 类

class ModelFuser:
    """
    封装Causal LM模型（如Llama, Mistral），用于生成融合描述。（最终优化版 v2）
    """
    def __init__(self, model_name: str):
        print("\nStep 2: Initializing Causal LM model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # <--- 修复警告问题 1: 为tokenizer设置pad_token --->
        # 如果tokenizer没有默认的pad_token，就使用eos_token（句子结尾符号）作为替代
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model initialized successfully.")

    def generate(self, entity_name: str, descriptions: list) -> str:
        """
        根据实体名和描述列表生成融合文本。
        """
        descriptions_text = "\n".join(descriptions)
        
        system_prompt = "You are an expert in summarizing and synthesizing information. Your task is to integrate a list of visual descriptions about an entity into a single, rich, and detailed paragraph. Capture as many key details as possible and ensure the final output is a coherent paragraph, not a list."
        user_prompt = f"Please synthesize the following descriptions for the entity '{entity_name}':\n\n{descriptions_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # <--- 修复警告问题 2: 让tokenizer同时返回input_ids和attention_mask --->
        # 将其打包成一个字典，然后移动到GPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # <--- 修复潜在问题: 智能判断使用哪个停止符 --->
        # 检查当前加载的模型名称是否包含'Llama-3'
        if 'Llama-3' in self.model.config.name_or_path:
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            # 对于Mistral和其他模型，只使用标准的eos_token_id
            terminators = self.tokenizer.eos_token_id

        # <--- 修复警告问题 3: 在generate函数中传入attention_mask --->
        # 使用 **inputs 将字典解包成 `input_ids=...` 和 `attention_mask=...`
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id, # 明确指定pad_token_id
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        # 从总输出中只解码新生成的部分
        response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        fused_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return fused_text


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="Fuse entity descriptions from JSON files using a Causal LM model.")
    parser.add_argument("samples", nargs="?", default="all", help="Number of entities to process (e.g., 10, 100). Use 'all' to process every entity. Defaults to 'all'.")
    args = parser.parse_args()
    num_to_process_str = args.samples

    merged_data = load_and_merge_data(INPUT_FILES)
    
    items_to_process = list(merged_data.items())
    if num_to_process_str.lower() != 'all':
        try:
            limit = int(num_to_process_str)
            items_to_process = items_to_process[:limit]
            print(f"\n-> Running in test mode. Processing a subset of {len(items_to_process)} entities.")
        except ValueError:
            print(f"\nError: Invalid sample count '{num_to_process_str}'. Please provide an integer or 'all'.")
            return
    else:
        print("\n-> Running in full mode. Processing all entities.")

    # <--- 4. 初始化新的模型封装类 --->
    fuser = ModelFuser(MODEL_NAME)
    
    print("\nStep 3: Processing entities and generating descriptions...")
    final_output = {}

    for entity_name, entity_data in tqdm(items_to_process, desc="Fusing Descriptions"):
        final_output[entity_name] = {key: value for key, value in entity_data.items() if not key.startswith('_')}
        final_output[entity_name]['images'] = {}

        descriptions_by_source = entity_data.get('_descriptions_by_source', {})
        descriptions_for_prompt = get_prioritized_descriptions(
            descriptions_by_source,
            PRIORITY_FILE,
            INPUT_FILES,
            MAX_DESCRIPTIONS_PER_ENTITY
        )
        
        if descriptions_for_prompt:
            try:
                fused_text = fuser.generate(entity_name, descriptions_for_prompt)
                final_output[entity_name]['images']['images_t5_descriptions'] = fused_text
            except Exception as e:
                print(f"\nError processing entity '{entity_name}': {e}")
    
    output_filename = OUTPUT_FILE
    if num_to_process_str.lower() != 'all':
        base, ext = os.path.splitext(OUTPUT_FILE)
        output_filename = f"{base}_sample_{len(items_to_process)}{ext}"
    
    print(f"\nStep 4: Saving fused data to '{output_filename}'...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print("✅ All tasks completed successfully!")

if __name__ == "__main__":
    main()