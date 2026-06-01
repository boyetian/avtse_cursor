#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate ref.txt, hyp.txt, hyp_xf.txt from individual .txt files."""

import os
import glob

def generate_index(data_dir, output_path):
    """Read all .txt files from data_dir and write ID<TAB>content to output_path."""
    files = glob.glob(os.path.join(data_dir, '*.txt'))
    entries = []
    for fpath in files:
        fname = os.path.basename(fpath)
        file_id = os.path.splitext(fname)[0]
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        entries.append((int(file_id), file_id, content))

    # Sort by numeric ID
    entries.sort(key=lambda x: x[0])

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        for _, file_id, content in entries:
            f.write(f'{file_id}\t{content}\r\n')

    print(f'Generated {output_path}: {len(entries)} entries')

if __name__ == '__main__':
    base = r'D:\GitDocs\AV_TSE_Cursor'
    generate_index(
        os.path.join(base, '测试结果', 'groundtruth'),
        os.path.join(base, 'scripts', 'ref.txt')
    )
    generate_index(
        os.path.join(base, '测试结果', '测试用例asr'),
        os.path.join(base, 'scripts', 'hyp.txt')
    )
    generate_index(
        os.path.join(base, '测试结果', '讯飞', 'audio_asr'),
        os.path.join(base, 'scripts', 'hyp_xf.txt')
    )
    print('Done.')
