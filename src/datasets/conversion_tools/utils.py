from enum import Enum
from typing import Union, Tuple, List, Optional
from functools import reduce

import argparse
import numpy as np

from src.datasets.conversion_tools.errors import DataConversionError


Number = Union[int, float]
Point = Tuple[Number, Number]
ImageSize = Tuple[int, int]


INVALID_ORIENTED_BOX_MSG = 'Oriented box should be list of number in format: ' \
                           '[left_top_x, left_top_y, right_top_x, ' \
                           'right_top_y, ..., left_bottom_x, left_bottom_y]'
INVALID_IMAGE_SIZE_MSG = 'Imag size is invalid, both factors must be ' \
                         'greater than 0.'
INVALID_BOX_COORDS_MSG = 'Bounding box coordinates do not lay at the image ' \
                         'interior or on its boundaries.'


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
    def assembly(coords: List[Number],
                 image_size: Optional[Tuple[int, int]] = None,
                 size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> 'OrientedBoundingBox':
        OrientedBoundingBox.__check_input(
            coords=coords,
            image_size=image_size
        )
        if image_size is not None:
            coords = OrientedBoundingBox.__standardize_box(
                coords=coords,
                image_size=image_size,
                size_format=size_format
            )
        return OrientedBoundingBox(
            left_top=(coords[0], coords[1]),
            right_top=(coords[2], coords[3]),
            right_bottom=(coords[4], coords[5]),
            left_bottom=(coords[6], coords[7])
        )

    @staticmethod
    def __check_input(coords: List[Number],
                      image_size: Optional[Tuple[int, int]]) -> None:
        if len(coords) is not 8 or \
                OrientedBoundingBox.__coords_not_aligned_clockwise(coords):
            raise DataConversionError(INVALID_ORIENTED_BOX_MSG)
        if image_size is None:
            return None
        print(image_size)
        if ImageSizeFormat.size_is_valid(image_size) is False:
            raise DataConversionError(INVALID_IMAGE_SIZE_MSG)

    @staticmethod
    def __coords_not_aligned_clockwise(coords: List[Number]) -> bool:
        left_top = coords[0], coords[1]
        right_top = coords[2], coords[3]
        right_bottom = coords[4], coords[5]
        left_bottom = coords[6], coords[7]
        return left_top[0] >= right_top[0] or \
               left_bottom[0] >= right_bottom[0] or \
               left_top[1] >= left_bottom[1] or \
               right_top[1] >= right_bottom[1]

    @staticmethod
    def __standardize_box(coords: List[Number],
                          image_size: ImageSize,
                          size_format: ImageSizeFormat) -> List[Number]:
        image_size = ImageSizeFormat.convert_to_width_height(
            image_size=image_size,
            source_format=size_format
        )
        width, height = image_size
        coords = zip(coords, [width, height] * 4)
        coords = list(map(lambda x: x[0] / x[1], coords))
        coords_error_exists = any([c < 0.0 or c > 1.0 for c in coords])
        if coords_error_exists:
            raise DataConversionError(INVALID_BOX_COORDS_MSG)
        return coords

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
