import os

import unittest
from parameterized import parameterized

from src.common.utils import read_csv_file, extract_file_name
from src.tests.utils import get_test_data_root


class CommonUtilsTests(unittest.TestCase):

    def __init__(self, methodName: str):
        super().__init__(methodName)
        test_data_root = get_test_data_root()
        self.__common_test_data_root = os.path.join(test_data_root, 'common')

    @parameterized.expand([
        ['test_1.csv', 0, 3],
        ['test_1.csv', 1, 2],
        ['test_1.csv', 3, 0],
        ['test_1.csv', 10, 0],
        ['test_2.csv', 0, 3],
        ['test_2.csv', 1, 2],
        ['test_2.csv', 3, 0],
        ['test_2.csv', 10, 0],
    ])
    def test_read_lines_number(self,
                               file_path: str,
                               lines_to_skip: int,
                               expected_lines_read: int) -> None:
        file_path = self.__get_test_file_path(file_path)
        file_content = read_csv_file(file_path, lines_to_skip)
        self.assertEqual(expected_lines_read, len(file_content))

    @parameterized.expand([
        ['test_1.csv', 3],
        ['test_2.csv', 4]
    ])
    def test_read_columns_number(self,
                                 file_path: str,
                                 expected_columns_in_each_row: int) -> None:
        file_path = self.__get_test_file_path(file_path)
        file_content = read_csv_file(file_path)
        for file_row in file_content:
            self.assertEqual(expected_columns_in_each_row, len(file_row))

    @parameterized.expand([
        ['test_1.csv', 0, 0, 'col_1'],
        ['test_1.csv', 0, 2, 'col_3'],
        ['test_1.csv', 1, 1, '2'],
        ['test_1.csv', 2, 0, '4'],
        ['test_1.csv', 2, 2, '6'],
        ['test_2.csv', 0, 0, '1.0'],
        ['test_2.csv', 0, 2, '3.0'],
        ['test_2.csv', 0, 3, 'str1'],
        ['test_2.csv', 1, 1, '5.0'],
        ['test_2.csv', 2, 0, '7.0'],
        ['test_2.csv', 2, 3, 'str3']
    ])
    def test_read_content(self,
                          file_path: str,
                          target_row: int,
                          target_column: int,
                          expected_content: str) -> None:
        file_path = self.__get_test_file_path(file_path)
        file_content = read_csv_file(file_path)
        read_content = file_content[target_row][target_column]
        self.assertEqual(expected_content, read_content)

    @parameterized.expand([
        ['file', 'file'],
        ['file.ext', 'file'],
        ['file.long_extension', 'file'],
        ['file.long_extension.ext', 'file.long_extension'],
        ['file_123', 'file_123'],
        ['file_123.ext', 'file_123'],
        ['file_123.ling_extension', 'file_123'],
        ['dupa/file', 'file'],
        ['dupa/file.ext', 'file'],
        ['/задница/file', 'file'],
        ['/尻/file.ext', 'file'],
        ['/cul/file.ext', 'file'],
        ['/røv/đít/file.ext', 'file']
    ])
    def test_file_name_extraction(self,
                                  file_path: str,
                                  expected_output: str) -> None:
        result = extract_file_name(file_path)
        self.assertEqual(expected_output, result)

    def __get_test_file_path(self, file_name: str) -> str:
        return os.path.join(self.__common_test_data_root, file_name)
