#!/usr/bin/env python3
import argparse
import os
import re

def split_markdown(input_file, output_dir, level=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    heading_prefix = '#' * level + ' '
    current_section = []
    section_title = "00_Введение"
    part_counter = 1

    def save_section():
        nonlocal part_counter
        if not current_section:
            return
        
        # Clean title for filename
        clean_title = re.sub(r'[\\/*?:"<>|]', "", section_title).strip()
        clean_title = clean_title.replace(" ", "_")
        if not clean_title:
            clean_title = f"part_{part_counter}"
            
        filename = f"{part_counter:02d}_{clean_title}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.writelines(current_section)
        
        part_counter += 1

    for line in lines:
        if line.startswith(heading_prefix):
            # Save the previous section
            if current_section:
                save_section()
            
            # Start a new section
            section_title = line[len(heading_prefix):].strip()
            current_section = [line]
        else:
            current_section.append(line)

    # Save the last section
    if current_section:
        save_section()
        
    print(f"Разделение завершено. Файлы сохранены в папке: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split markdown file by headings")
    parser.add_argument("input_file", help="Path to the input .md file")
    parser.add_argument("-o", "--output_dir", help="Directory to save split files", default="split_book")
    parser.add_argument("-l", "--level", type=int, help="Heading level to split by (1 for #, 2 for ##, etc.)", default=1)
    
    args = parser.parse_args()
    split_markdown(args.input_file, args.output_dir, args.level)
