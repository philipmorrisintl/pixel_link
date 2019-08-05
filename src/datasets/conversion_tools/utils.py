from enum import Enum
from typing import Union, Tuple, List
from functools import reduce

import argparse
import numpy as np

from src.datasets.conversion_tools.converters import DataConversionError

Number = Union[int, float]
Point = Tuple[Number, Number]


INVALID_ORIENTED_BOX_MSG = 'Oriented box should be list of number in format: ' \
                           '[left_top_x, left_top_y, right_top_x, ' \
                           'right_top_y, ..., left_bottom_x, left_bottom_y]'
INVALID_IMAGE_SIZE_MSG = 'Imag size is invalid, both factors must be ' \
                         'greater than 0.'


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Dataset converter to *.tfrecords format'
    )
    parser.add_argument(
        '--training_samples',
        dest='training_list_path',
        help='Path to *.csv file with training samples',
        type=str
    )
    parser.add_argument(
        '--test_samples',
        dest='test_list_path',
        help='Path to *.csv file with test samples',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Folder to save converted TFRecords',
        type=str
    )
    args = parser.parse_args()
    return args


class ImageSizeFormat(Enum):

    ImageSize = Tuple[int, int]

    HEIGHT_WIDTH = 1
    WIDTH_HEIGHT = 2

    @staticmethod
    def convert_to_height_width(image_size: ImageSize,
                                source_format: 'ImageSizeFormat') -> ImageSize:
        if source_format is ImageSizeFormat.HEIGHT_WIDTH:
            return image_size
        else:
            return image_size[1], image_size[0]

    @staticmethod
    def convert_to_width_height(image_size: ImageSize,
                                source_format: 'ImageSizeFormat') -> ImageSize:
        if source_format is ImageSizeFormat.WIDTH_HEIGHT:
            return image_size
        else:
            return image_size[1], image_size[0]

    @staticmethod
    def size_is_valid(image_size: ImageSize) -> bool:
        return image_size[0] > 0 and image_size[1] > 0


class OrientedBoundingBox:

    @staticmethod
    def assembly(points: List[Number],
                 image_size: ImageSizeFormat.ImageSize = None,
                 size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> 'OrientedBoundingBox':
        if len(points) is not 8:
            raise DataConversionError(INVALID_ORIENTED_BOX_MSG)
        if image_size is not None:
            points = OrientedBoundingBox.__standardize_box(
                points=points,
                image_size=image_size,
                size_format=size_format
            )
        return OrientedBoundingBox(
            left_top=(points[0], points[1]),
            right_top=(points[2], points[3]),
            right_bottom=(points[4], points[5]),
            left_bottom=(points[6], points[7])
        )

    @staticmethod
    def __standardize_box(points: List[Number],
                          image_size: ImageSizeFormat.ImageSize,
                          size_format: ImageSizeFormat) -> List[Number]:
        if not ImageSizeFormat.size_is_valid(image_size):
            raise DataConversionError(INVALID_IMAGE_SIZE_MSG)
        image_size = ImageSizeFormat.convert_to_width_height(
            image_size=image_size,
            source_format=size_format
        )
        width, height = image_size
        points = zip(points, [width, height] * 4)
        return list(map(lambda x: x[0] / x[1], points))

    def __init__(self,
                 left_top: Point,
                 right_top: Point,
                 right_bottom: Point,
                 left_bottom: Point):
        self.__left_top = left_top
        self.__right_top = right_top
        self.__right_bottom = right_bottom
        self.__left_bottom = left_bottom

    @property
    def left_top(self) -> Point:
        return self.__left_top

    @property
    def right_top(self) -> Point:
        return self.__right_top

    @property
    def right_bottom(self) -> Point:
        return self.__right_bottom

    @property
    def left_bottom(self) -> Point:
        return self.__left_bottom

    @property
    def array_form(self) -> np.ndarray:
        return np.asarray(self.list_form)

    @property
    def list_form(self) -> List[Number]:
        points = self.__get_points_in_clockwise_order()
        return reduce(lambda acc, x: acc + [x[0], x[1]], points, [])

    def __get_points_in_clockwise_order(self):
        return [
            self.__left_top, self.__right_top,
            self.__right_bottom, self.__left_bottom
        ]
