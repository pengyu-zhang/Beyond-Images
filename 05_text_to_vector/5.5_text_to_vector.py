#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_json_to_mkg.py

从一个 JSON 文件中读取实体信息，每个实体含有 "merged_descriptions" 的字段。
对该文本进行向量化(默认使用 384 维 sentence-transformers)，
然后写入与原始 MKG 数据类似的 H5 和 PTH 文件结构。

使用前请先安装:
    pip install sentence-transformers h5py torch

运行示例:
    python convert_json_to_mkg.py

完成后，将会在当前目录下生成:
    - my_new_dataset.h5
    - my_new_dataset.pth

你可用之前的 inspect 脚本验证二者结构是否与原始数据一致。
"""

import os
import sys
import json
import logging
import datetime
import h5py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

LOG_FILENAME = 'convert_json_to_mkg.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 输出文件路径，可根据需要修改
OUTPUT_H5_PATH = "/gpfs/work5/0/prjs1145/code/mmkg-mmrns/data/MKG_W_description_sentences.h5"
OUTPUT_PTH_PATH = "/gpfs/work5/0/prjs1145/code/mmkg-adamf/embeddings/MKG-W-textual.pth"

# 这里是你的 JSON 文件路径，请修改为实际路径
INPUT_JSON_PATH = "/gpfs/home5/pzhang/beyond_images/datasets/img_text_summary/MKG-W_original_img_summary_blip.json"

# 如果你希望使用其他模型或非 384 维度，请在此处修改
# 例如 "sentence-transformers/all-mpnet-base-v2" 返回768维sentence-transformers/multi-qa-MiniLM-L6-cos-v1
EMBEDDING_MODEL_NAME = "bert-base-uncased"  
# 该模型默认输出 384 维。如果你想要768维，可以尝试 'all-mpnet-base-v2' 等

def load_json(json_path):
    """
    从给定的 JSON 文件中读取数据并返回 dict 对象。
    """
    logger.info("开始读取 JSON 文件: %s", json_path)
    if not os.path.exists(json_path):
        logger.error("JSON 文件不存在: %s", json_path)
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("JSON 文件读取完毕，共包含 %d 个实体(顶层键)。", len(data))
        return data
    except Exception as e:
        logger.error("读取 JSON 文件时发生错误: %s", e, exc_info=True)
        return {}

def embed_texts(text_list, model):
    """
    使用指定的 sentence-transformers 模型对一批文本进行向量化。
    返回一个 numpy array, shape=(len(text_list), embedding_dim).
    """
    try:
        embeddings = model.encode(text_list, convert_to_numpy=True)
        # embeddings 形状: [batch_size, embedding_dim]
        return embeddings
    except Exception as e:
        logger.error("文本向量化时发生错误: %s", e, exc_info=True)
        return np.zeros((len(text_list), 384), dtype=np.float32)  # 兜底处理

def convert_to_data_dict(json_data, model):
    """
    将 JSON 数据转为 {entity_name: [向量1, 向量2, ...], ...} 的结构，以写入 H5/PTH.

    这里默认:
    - 每个实体仅使用 "merged_descriptions" 一条文本生成一条向量
    - 如果你的 "merged_descriptions" 需要拆分成多行，可自行修改
    """
    data_dict = {}

    for entity_key, entity_info in json_data.items():
        # entity_key 类似 "National_Congress_(Sudan)"
        # entity_info 是一个 dict，包含 "merged_descriptions" 在 entity_info["images"]["merged_descriptions"]
        try:
            images_info = entity_info.get("images", {})
            merged_text = images_info.get("merged_descriptions", "").strip()
            if not merged_text:
                logger.warning("实体 %s 没有 merged_descriptions，跳过。", entity_key)
                continue

            # 对 merged_descriptions 做向量化
            embeddings = embed_texts([merged_text], model)  # 返回 shape=(1, embedding_dim)
            # 转成 float32
            embeddings = embeddings.astype(np.float32)

            # data_dict[entity_key] 的值应该是一个 list[np.array(384,)]
            # 如果后续想要多行，可以把 merged_text 拆成多句，然后再 encode
            data_dict[entity_key] = [embeddings[0]]

        except Exception as e:
            logger.error("处理实体 %s 时出现错误: %s", entity_key, e, exc_info=True)

    return data_dict

def write_h5_file(data_dict, h5_path):
    """
    将 data_dict 写入 H5 文件: {entity: [vec1, vec2, ...]}
    每个 entity 在 H5 中创建一个 dataset，shape = (num_sentences, embedding_dim).
    """
    logger.info("开始写入 H5 文件: %s", h5_path)
    if os.path.exists(h5_path):
        logger.warning("目标 H5 文件已存在，将被覆盖: %s", h5_path)
        os.remove(h5_path)

    try:
        with h5py.File(h5_path, 'w') as h5_file:
            for entity, vectors in data_dict.items():
                # vectors 形如 list[np.array(384, ), np.array(384, ), ...]
                # 先把它们堆叠起来
                vectors_np = np.stack(vectors, axis=0)  # shape=(num_sentences, 384)
                h5_file.create_dataset(entity, data=vectors_np)
        logger.info("H5 文件写入完成: %s", h5_path)
    except Exception as e:
        logger.error("写入 H5 文件时出错: %s", e, exc_info=True)

def write_pth_file(data_dict, pth_path):
    """
    将 data_dict 的所有向量拼接成一个 (total_sentences, embedding_dim) 的 tensor，并保存为 .pth
    """
    logger.info("开始写入 PTH 文件: %s", pth_path)
    if os.path.exists(pth_path):
        logger.warning("目标 PTH 文件已存在，将被覆盖: %s", pth_path)
        os.remove(pth_path)

    all_vectors = []
    for entity, vectors in data_dict.items():
        # vectors 可能是 list[np.array(384, ), ...]
        # 堆叠后再收集
        vectors_np = np.stack(vectors, axis=0)
        all_vectors.append(vectors_np)

    if len(all_vectors) == 0:
        logger.warning("data_dict 为空，无法写入 PTH 文件。")
        return

    all_vectors_np = np.concatenate(all_vectors, axis=0)  # shape=(total_sentences, 384)
    all_vectors_tensor = torch.from_numpy(all_vectors_np)

    try:
        torch.save(all_vectors_tensor, pth_path)
        logger.info("PTH 文件写入完成: %s", pth_path)
    except Exception as e:
        logger.error("写入 PTH 文件时出错: %s", e, exc_info=True)

def main():
    logger.info("脚本开始执行。")

    # 1. 加载 JSON
    json_data = load_json(INPUT_JSON_PATH)
    if not json_data:
        logger.warning("JSON 数据为空或加载失败，脚本结束。")
        return

    # 2. 初始化 SentenceTransformer 模型
    #    如果你有更适合的模型，可以替换 EMBEDDING_MODEL_NAME
    try:
        logger.info("加载模型: %s", EMBEDDING_MODEL_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error("加载模型失败: %s", e, exc_info=True)
        return

    # 3. 生成 {entity -> 向量列表} 字典
    data_dict = convert_to_data_dict(json_data, model)
    logger.info("共生成 %d 个实体的向量。", len(data_dict))

    # 4. 写入 H5 文件
    write_h5_file(data_dict, OUTPUT_H5_PATH)

    # 5. 写入 PTH 文件
    write_pth_file(data_dict, OUTPUT_PTH_PATH)

    logger.info("脚本执行结束。")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("脚本执行过程中出现未捕获的异常: %s", e, exc_info=True)
        sys.exit(1)
