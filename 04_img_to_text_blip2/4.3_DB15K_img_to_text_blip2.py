from PIL import Image, ImageFile
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken
import torch
import os
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16).to(device)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto").to(device)

# 添加图像 token 以兼容新版本
processor.num_query_tokens = model.config.num_query_tokens
image_token = AddedToken("<image>", normalized=False, special=True)
processor.tokenizer.add_tokens([image_token], special_tokens=True)

# 调整模型的嵌入层大小
model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
model.config.image_token_index = len(processor.tokenizer) - 1

# 允许加载被截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义生成函数
def generate_batch_descriptions(image_paths, prompt="Describe the scene, objects, colors, and other details in detail.", max_new_tokens=100):
    images = []
    for image_path in image_paths:
        try:
            # 打开图像并验证其有效性
            img = Image.open(image_path)
            img.verify()  # 验证图像是否有效
            img = Image.open(image_path).convert("RGB")  # 如果验证通过，重新打开并转换为 RGB 模式
            images.append(img)
        except (OSError, KeyError, TypeError, ValueError, Image.DecompressionBombError) as e:
            print(f"跳过无法处理的图像 {image_path}: {e}")
            continue

    if not images:
        return []  # 如果没有有效图像，则返回空列表
    
    try:
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        descriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [desc.strip() for desc in descriptions]  # 添加 .strip() 去除空白
    except Exception as e:
        print(f"生成描述时发生错误: {e}")
        return []


# 图片文件夹路径
image_folder = "/gpfs/scratch1/nodespecific/int5/pzhang/beyond_images/datasets/DB15K_img_new"
output_file = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_img_new_descriptions.txt"
batch_size = 100  # 设置批次大小

with open(output_file, "a") as f:
    all_images = os.listdir(image_folder)
    for i in range(0, len(all_images), batch_size):
        batch_files = all_images[i:i + batch_size]
        batch_paths = [os.path.join(image_folder, image_file) for image_file in batch_files]
        descriptions = generate_batch_descriptions(batch_paths)

        # 写入并立即刷新
        for image_file, description in zip(batch_files, descriptions):
            f.write(f"{image_file}: {description}\n")
        f.flush()  # 每批次刷新文件，避免积压内存

        print(f"已处理 {i + len(batch_files)} 张图片")

