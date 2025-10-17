#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本脚本用于读取并分析 .h5 和 .pth 文件的结构与基本信息。
无需命令行参数，直接在代码中指定文件路径后运行即可。

运行方式:
    python inspect_original_files_no_args.py

输出:
    在脚本同目录下生成 inspection.log 文件，记录读取到的文件结构和信息。
"""

import os
import sys
import logging
import datetime
import pandas as pd
import h5py  # 用于读取 .h5 文件
import torch  # 用于读取 .pth 文件

# 在此处写死路径，无需在命令行输入
h5_path = "/gpfs/work5/0/prjs1145/code/mmkg-mmrns/data/MKG_W_description_sentences.h5"
pth_path = "/gpfs/work5/0/prjs1145/code/mmkg-adamf/embeddings/MKG-W-textual.pth"

# 配置日志
LOG_FILENAME = 'inspection.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

def read_h5_file(path):
    """
    读取 .h5 文件并输出其内部数据集的结构和基本信息。

    :param path: .h5 文件的路径
    """
    logger.info("开始读取 H5 文件: %s", path)
    if not os.path.exists(path):
        logger.error("H5 文件不存在: %s", path)
        return

    try:
        with h5py.File(path, 'r') as h5_file:
            keys = list(h5_file.keys())
            logger.info("H5 文件顶层数据集/组: %s", keys)

            for key in keys:
                data = h5_file[key]
                if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                    # 如果是 dataset
                    logger.info("数据集名称: %s, shape: %s, dtype: %s",
                                key, data.shape, data.dtype)
                else:
                    # 如果是 group，列出其内部
                    logger.info("组名称: %s", key)
                    sub_keys = list(data.keys())
                    logger.info("组 %s 中包含的键: %s", key, sub_keys)

    except Exception as e:
        logger.error("读取 H5 文件时发生错误: %s", e, exc_info=True)

def read_pth_file(path):
    """
    读取 .pth 文件（通常为 PyTorch 保存的模型或数据字典），并输出其内部键和基本信息。

    :param path: .pth 文件的路径
    """
    logger.info("开始读取 PTH 文件: %s", path)
    if not os.path.exists(path):
        logger.error("PTH 文件不存在: %s", path)
        return

    try:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            logger.info("PTH 文件加载后得到的数据类型: dict")
            for key, value in data.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    logger.info("键: %s, 类型: %s, shape: %s, dtype: %s", 
                                key, type(value), value.shape, value.dtype)
                else:
                    logger.info("键: %s, 类型: %s, 内容示例(前50字符): %s", 
                                key, type(value), str(value)[:50])
        else:
            logger.info("PTH 文件加载后得到的数据类型: %s", type(data))
            if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                logger.info("数据 shape: %s, dtype: %s", data.shape, data.dtype)
            else:
                logger.info("数据内容示例(前50字符): %s", str(data)[:50])
    except Exception as e:
        logger.error("读取 PTH 文件时发生错误: %s", e, exc_info=True)

def main():
    """
    主函数，无需命令行参数，直接读取脚本中指定的文件路径。
    """
    logger.info("脚本开始执行。")

    # 读取 .h5 文件
    read_h5_file(h5_path)

    # 读取 .pth 文件
    read_pth_file(pth_path)

    logger.info("脚本执行结束。")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("脚本执行过程中出现未捕获的异常: %s", e, exc_info=True)
        sys.exit(1)
