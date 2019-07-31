from abc import ABC, abstractmethod
import os
from enum import Enum
from functools import partial, reduce
from typing import Optional, List, Tuple, Union

from tensorflow.contrib.data import TFRecordWriter
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

from src.common.utils import read_csv_file, extract_file_name
from src.datasets.utils import convert_to_example, OptionalBBoxes
from src.logger.logger import get_logger

logger = get_logger(__file__)

Number = Union[int, float]
Point = Tuple[Number, Number]

INVALID_FORMAT_ERROR_MSG = 'Some entries contain not sufficient number ' \
                           'of elements. Valid format of entry: ' \
                           'image_path, gt_path'
INVALID_ORIENTED_BOX_MSG = 'Oriented box should be list of number in format: ' \
                           '[left_top_x, left_top_y, right_top_x, ' \
                           'right_top_y, ..., left_bottom_x, left_bottom_y]'
INVALID_IMAGE_SIZE_MSG = 'Imag size is invalid, both factors must be ' \
                         'greater than 0.'


class DataConversionError(Exception):
    pass


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


class TrainingExample:

    def __init__(self,
                 image: np.ndarray,
                 file_name: str,
                 oriented_bboxes: List[OrientedBoundingBox]):
        self.__image = image
        self.__file_name = file_name
        oriented_bboxes = list(map(lambda x: x.array_form, oriented_bboxes))
        self.__oriented_bboxes = oriented_bboxes

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def file_name(self) -> str:
        return self.__file_name

    @property
    def oriented_bboxes(self) -> List[np.ndarray]:
        return self.__oriented_bboxes


class DatasetParser(ABC):

    @abstractmethod
    def parse_example(self, image_path: str, gt_path: str) -> TrainingExample:
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')


class PascalVOCParser(DatasetParser):

    def parse_example(self, image_path: str, gt_path: str) -> TrainingExample:
        image_name = extract_file_name(image_path)
        image = cv.imread(image_path)
        annotation_parsed = ET.parse(gt_path)
        image_size = image.shape[:2]
        oriented_bboxes = self.__get_bboxes(
            annotation=annotation_parsed,
            image_size=image_size
        )
        return TrainingExample(
            image=image,
            file_name=image_name,
            oriented_bboxes=oriented_bboxes
        )

    def __get_bboxes(self,
                     annotation: ET.ElementTree,
                     image_size: Tuple[int, int]) -> List[OrientedBoundingBox]:
        oriented_bboxes = []
        annotation_objects = annotation.findall('object')
        annotation_objects = list(filter(
            self.__annotation_is_valid,
            annotation_objects
        ))
        for annotation_object in annotation_objects:
            voc_bbox = self.__extract_voc_bbox(annotation_object)
            oriented_bbox = self.__get_oriented_bbox(
                voc_bbox=voc_bbox,
                image_size=image_size
            )
            oriented_bboxes.append(oriented_bbox)
        return oriented_bboxes

    def __annotation_is_valid(self, annotation: ET.Element) -> bool:
        bbox = annotation.find('bndbox')
        if bbox is None:
            return False
        elements_to_test = ['xmin', 'ymin', 'xmax', 'ymax']
        return all([
            bbox.find(elem_to_test) is not None
            for elem_to_test in elements_to_test
        ])

    def __extract_voc_bbox(self, annotation_object: ET.Element) -> np.ndarray:
        bbox = annotation_object.find('bndbox')
        min_x = int(bbox.find('xmin').text)
        min_y = int(bbox.find('ymin').text)
        max_x = int(bbox.find('xmax').text)
        max_y = int(bbox.find('ymax').text)
        return np.asarray([min_x, min_y, max_x, max_y])

    def __get_oriented_bbox(self,
                            voc_bbox: np.ndarray,
                            image_size: ImageSizeFormat.ImageSize) -> OrientedBoundingBox:
        oriented_bbox_points = [
            voc_bbox[0], voc_bbox[1],
            voc_bbox[2], voc_bbox[1],
            voc_bbox[2], voc_bbox[3],
            voc_bbox[0], voc_bbox[3]
        ]
        return OrientedBoundingBox.assembly(
            points=oriented_bbox_points,
            image_size=image_size
        )


class DatasetConverter:

    def __init__(self,
                 dataset_parser: DatasetParser,
                 output_dir: str,
                 training_list_path: str,
                 test_list_path: Optional[str] = None):
        self.__dataset_parser = dataset_parser
        self.__output_dir = output_dir
        self.__training_list_path = training_list_path
        self.__test_list_path = test_list_path

    def convert(self) -> None:
        self.__convert_training_set()
        if self.__test_set_should_be_converted():
            self.__convert_test_set()

    def __convert_training_set(self) -> None:
        self.__convert_dataset_part(
            examples_list_path=self.__training_list_path,
            dataset_part_name='training'
        )

    def __test_set_should_be_converted(self) -> bool:
        return self.__test_list_path is not None

    def __convert_test_set(self) -> None:
        self.__convert_dataset_part(
            examples_list_path=self.__test_list_path,
            dataset_part_name='test'
        )

    def __convert_dataset_part(self,
                               examples_list_path: str,
                               dataset_part_name: str) -> None:
        examples = read_csv_file(examples_list_path)
        self.__check_input_integrity(examples)
        logger.info(f'{len(examples)} will be '
                    f'placed in {dataset_part_name} set')
        tfrecords_path = self.__get_target_tfrecords_path(dataset_part_name)
        with TFRecordWriter(tfrecords_path) as tfrecord_writer:
            self.__convert_examples(tfrecord_writer, examples)

    def __check_input_integrity(self, examples: List[list]) -> None:
        wrong_entries = list(filter(lambda x: len(x) != 2, examples))
        if len(wrong_entries) > 0:
            raise DataConversionError(INVALID_FORMAT_ERROR_MSG)

    def __get_target_tfrecords_path(self, dataset_part_name: str) -> str:
        return os.path.join(
            self.__output_dir,
            f'{dataset_part_name}-set.tfrecords'
        )

    def __convert_examples(self,
                           tfrecord_writer: TFRecordWriter,
                           examples: List[list]) -> None:
        convert_example = partial(
            self.__convert_example,
            tfrecord_writer=tfrecord_writer
        )
        for image_path, gt_path in examples:
            convert_example(image_path, gt_path)

    def __convert_example(self,
                          image_path: str,
                          gt_path: str,
                          tfrecord_writer: TFRecordWriter) -> None:
        annotation = self.__dataset_parser.parse_example(image_path, gt_path)
        example = convert_to_example(
            image=annotation.image,
            file_name=annotation.file_name,
            oriented_bboxes=annotation.oriented_bboxes)
        tfrecord_writer.write(example.SerializeToString())


