import os
from functools import partial
from typing import List

import shutil
import unittest
from parameterized import parameterized
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.python_io import TFRecordWriter

from src.common.utils import create_dir
from src.datasets.conversion_tools.utils import ImageSizeFormat, ImageSize
from src.datasets.utils import convert_to_example, parse_example
from src.tests.utils import get_test_data_root


class DataSetsUtilsTests(unittest.TestCase):

    def __init__(self, methodName: str):
        super().__init__(methodName)
        test_data_root = get_test_data_root()
        self.__test_data_root = os.path.join(
            test_data_root,
            'datasets',
            'tfrecords_conversion_tests')

    def setUp(self) -> None:
        create_dir(self.__test_data_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.__test_data_root)

    @parameterized.expand([
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])],
            5, (120, 80)
        ],
        [
            33, np.zeros((120, 80, 3)), 'dumb_file',
            [np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])],
            5, (120, 80)
        ],
        [
            0, np.zeros((240, 120, 3)), 'dumb_file',
            [np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])],
            5, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            5, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            5, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            5, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.9, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 1.0]),
                np.asarray([0.0, 0.7, 1.0, 0.2, 1.0, 1.0, 0.4, 1.0]),
                np.asarray([0.1, 0.3, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
            ],
            5, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            1, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            1, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
            ],
            1, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [
                np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.9, 0.0, 1.0, 0.0, 1.0, 1.0, 0.9, 1.0]),
                np.asarray([0.0, 0.7, 1.0, 0.2, 1.0, 1.0, 0.4, 1.0]),
                np.asarray([0.1, 0.3, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
                np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
            ],
            1, (120, 80)
        ],
        [
            0, np.zeros((120, 80, 3)), 'dumb_file',
            [np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])],
            5, (1024, 1024)
        ]
    ])
    def test_conversion(self,
                        source_example_id: int,
                        image: np.ndarray,
                        file_name: str,
                        oriented_bboxes: List[np.ndarray],
                        max_bounding_boxes: int,
                        target_size: ImageSize,
                        size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> None:
        tfrecords_path = os.path.join(self.__test_data_root, 'test.tfrecords')
        with TFRecordWriter(tfrecords_path) as tfrecord_writer:
            example = convert_to_example(
                example_id=source_example_id,
                image=image,
                file_name=file_name,
                oriented_bboxes=oriented_bboxes,
                max_bounding_boxes=max_bounding_boxes,
                target_size=target_size,
                size_format=size_format
            )
            tfrecord_writer.write(example.SerializeToString())
        oriented_bboxes = self.__prapare_target_bboxes(
            oriented_bboxes,
            max_bounding_boxes
        )
        if size_format is ImageSizeFormat.HEIGHT_WIDTH:
            target_size = target_size[1], target_size[0]
        image = cv.resize(image, target_size)
        image = image.astype(dtype=np.float32)
        dataset = tf.data.TFRecordDataset([tfrecords_path])
        parse = partial(parse_example, decode_example_id=True)
        dataset = dataset.batch(1, drop_remainder=True)
        dataset = dataset.map(parse)
        iterator = dataset.make_one_shot_iterator()
        example_id, x, y = iterator.get_next()
        with tf.Session() as session:
            example_id_eval, x_eval, y_eval = session.run([example_id, x, y])
            self.assertEqual(example_id_eval.shape[0], 1)
            print(x_eval.shape)
            example_id_eval = example_id_eval[0]
            x1, y1, x2, y2, x3, y3, x4, y4 = \
                y_eval[0][0], y_eval[1][0], y_eval[2][0], y_eval[3][0],\
                y_eval[4][0], y_eval[5][0], y_eval[6][0], y_eval[7][0]
            self.assertEqual(example_id_eval, source_example_id)
            self.assertTrue(np.array_equal(image, x_eval))
            y_eval = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)
            self.assertEqual(y_eval.shape[0], max_bounding_boxes)
            self.assertTrue(np.array_equal(oriented_bboxes, y_eval))
            try:
                session.run([example_id, x, y])
            except tf.errors.OutOfRangeError:
                return None
            raise RuntimeError('Successfull test should not reach this point')

    def __prapare_target_bboxes(self,
                                oriented_bboxes: List[np.ndarray],
                                max_bounding_boxes: int) -> np.ndarray:
        bboxes_array = np.asarray(oriented_bboxes, dtype=np.float32)
        if bboxes_array.shape[0] > max_bounding_boxes:
            bboxes_array = bboxes_array[:max_bounding_boxes, :]
        else:
            to_pad = max_bounding_boxes - bboxes_array.shape[0]
            pad = np.zeros((to_pad, 8), dtype=np.float32)
            bboxes_array = np.concatenate((bboxes_array, pad), axis=0)
        return bboxes_array
