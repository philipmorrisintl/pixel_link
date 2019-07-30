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


def extract_file_name(file_path: str) -> str:
    file_base_name = os.path.basename(file_path)
    file_base_name = file_base_name[:file_base_name.rfind('.')]
    return file_base_name
