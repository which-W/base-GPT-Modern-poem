import os
import json
from pathlib import Path
import re

def extract_author_name(folder_name):
    """
    从文件夹名提取作者名（取第一个下划线前的部分）
    如果没有下划线，返回原名称
    """
    # 按第一个下划线分割，取第一部分
    author = folder_name.split('_', 1)[0]
    return author

def parse_poem_file(file_path, author):
    """解析单个 .pt 诗歌文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    title = ""
    content_lines = []
    in_content = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('title:'):
            title = stripped[6:].strip()  # 去掉 "title:"
        elif stripped == 'date:':
            continue
        elif stripped == '':  # 空行后开始正文
            in_content = True
        else:
            if in_content:
                content_lines.append(stripped)
    
    # 按空行分段（更通用的分段方式）
    paragraphs = []
    current_para = []
    
    for line in content_lines:
        if line == "":
            if current_para:
                paragraphs.append("\n".join(current_para))
                current_para = []
        else:
            current_para.append(line)
    
    # 添加最后一段
    if current_para:
        paragraphs.append("\n".join(current_para))
    
    # 如果没有检测到段落，整个内容作为一段
    if not paragraphs and content_lines:
        paragraphs = ["\n".join(content_lines)]
    
    return {
        "title": title,
        "para": paragraphs,
        "author": author
    }

def main(root_dir, output_json):
    """主函数：遍历目录并生成 JSON"""
    all_poems = []
    
    # 遍历所有子目录（作者文件夹）
    for author_dir in Path(root_dir).iterdir():
        if author_dir.is_dir():
            # 提取作者名（下划线前的部分）
            clean_author = extract_author_name(author_dir.name)
            print(f"处理作者: {clean_author} (原文件夹: {author_dir.name})")
            
            # 遍历该作者的所有 .pt 文件
            for poem_file in author_dir.glob("*.pt"):
                try:
                    poem_data = parse_poem_file(poem_file, clean_author)
                    all_poems.append(poem_data)
                    print(f"  - 已添加: {poem_data['title']}")
                except Exception as e:
                    print(f"  - 处理失败 {poem_file}: {e}")
    
    # 保存为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_poems, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！共处理 {len(all_poems)} 首诗歌，保存至 {output_json}")

if __name__ == "__main__":
    #配置文件
    SCRIPT_DIR = Path(__file__).parent  
    POETRY_DATA_DIR = SCRIPT_DIR / "poetry" / "data"  
    
    ROOT_DIR = POETRY_DATA_DIR
    OUTPUT_JSON = SCRIPT_DIR / "all_poems.json"  # 输出到项目根目录
    
    main(ROOT_DIR, OUTPUT_JSON)