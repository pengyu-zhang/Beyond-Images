def transform_file(input_file, output_file):
    """
    从 input_file 中读取行，提取其中的 mid 和 dbpedia URI，
    以 "dbpediaURI    mid" 的格式写入 output_file
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            # 如果是空行则跳过
            if not line:
                continue
            
            # 假设每行的格式固定为：
            # /m/xxx <SameAs> <http://dbpedia.org/resource/yyy> .
            parts = line.split()
            # parts[0] -> /m/xxx
            # parts[1] -> <SameAs>
            # parts[2] -> <http://dbpedia.org/resource/yyy>
            # parts[3] -> .
            
            if len(parts) != 4:
                # 如果格式不符合预期，可根据需要选择报错或跳过
                continue
            
            mid = parts[0]  # /m/xxx
            dbpedia_uri = parts[2].strip('<>')  # 去除尖括号 <http://...>
            
            # 输出格式： http://dbpedia.org/resource/xxx    /m/xxx
            fout.write(f"{dbpedia_uri}\t{mid}\n")


if __name__ == "__main__":
    # 示例：假设原始文件为 input.txt，输出文件为 output.txt
    input_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_SameAsLink.txt"
    output_path = "/gpfs/home5/pzhang/beyond_images/datasets/log/DB15K_ent_links.txt"
    
    transform_file(input_path, output_path)
    print("转换完成！")
