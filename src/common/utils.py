from typing import List
import os

import csv


def read_csv_file(file_path: str, skip_lines: int = 0) -> List[list]:
    result = []
    with open(file_path, mode='r') as csv_file:
        csv_file_reader = csv.reader(csv_file)
        for row in csv_file_reader:
            result.append(row)
    return result[skip_lines:]


def dump_content_to_csv(file_path: str, content: List[list]) -> None:
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(content)


def extract_file_name(file_path: str) -> str:
    file_base_name = os.path.basename(file_path)
    last_dot_pos = file_base_name.rfind('.')
    if last_dot_pos < 0:
        last_dot_pos = len(file_path)
    file_base_name = file_base_name[:last_dot_pos]
    return file_base_name


def create_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, 0o755, exist_ok=True)
