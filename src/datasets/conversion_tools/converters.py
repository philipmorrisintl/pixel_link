from abc import ABC, abstractmethod
import os
from functools import partial
from typing import Optional, List, Tuple

from tensorflow.contrib.data import TFRecordWriter
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

from src.common.utils import read_csv_file, extract_file_name, \
    dump_content_to_csv
from src.datasets.conversion_tools.errors import DataConversionError
from src.datasets.conversion_tools.utils import OrientedBoundingBox, \
    ImageSizeFormat
from src.datasets.utils import convert_to_example
from src.logger.logger import get_logger

logger = get_logger(__file__)


INVALID_FORMAT_ERROR_MSG = 'Some entries contain not sufficient number ' \
                           'of elements. Valid format of entry: ' \
                           'image_path, gt_path'


class TrainingExample:

    def __init__(self,
                 image_id: int,
                 image: np.ndarray,
                 file_name: str,
                 oriented_bboxes: List[OrientedBoundingBox]):
        self.__image_id = image_id
        self.__image = image
        self.__file_name = file_name
        oriented_bboxes = list(map(lambda x: x.array_form, oriented_bboxes))
        self.__oriented_bboxes = oriented_bboxes

    @property
    def image_id(self) -> int:
        return self.__image_id

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
    def parse_example(self,
                      example_id: int,
                      image_path: str,
                      gt_path: str) -> TrainingExample:
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')


class PascalVOCParser(DatasetParser):

    def parse_example(self,
                      example_id: int,
                      image_path: str,
                      gt_path: str) -> TrainingExample:
        image_name = extract_file_name(image_path)
        image = cv.imread(image_path)
        annotation_parsed = ET.parse(gt_path)
        image_size = image.shape[:2]
        oriented_bboxes = self.__get_bboxes(
            annotation=annotation_parsed,
            image_size=image_size
        )
        return TrainingExample(
            image_id=example_id,
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
                            image_size: ImageSize) -> OrientedBoundingBox:
        oriented_bbox_points = [
            voc_bbox[0], voc_bbox[1],
            voc_bbox[2], voc_bbox[1],
            voc_bbox[2], voc_bbox[3],
            voc_bbox[0], voc_bbox[3]
        ]
        return OrientedBoundingBox.assembly(
            coords=oriented_bbox_points,
            image_size=image_size
        )


class DatasetConverter:

    def __init__(self,
                 dataset_parser: DatasetParser,
                 max_bounding_boxes: int,
                 target_size: ImageSize,
                 output_dir: str,
                 training_list_path: str,
                 test_list_path: Optional[str] = None,
                 size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH):
        self.__dataset_parser = dataset_parser
        self.__max_bounding_boxes = max_bounding_boxes
        self.__target_size = target_size
        self.__output_dir = output_dir
        self.__training_list_path = training_list_path
        self.__test_list_path = test_list_path
        self.__size_format = size_format

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
        self.__save_examples_identity_list(
            examples=examples,
            dataset_part_name=dataset_part_name
        )

    def __check_input_integrity(self, examples: List[list]) -> None:
        wrong_entries = list(filter(lambda x: len(x) != 2, examples))
        if len(wrong_entries) > 0:
            raise DataConversionError(INVALID_FORMAT_ERROR_MSG)

    def __get_target_tfrecords_path(self, dataset_part_name: str) -> str:
        return os.path.join(
            self.__output_dir,
            f'{dataset_part_name}-set.tfrecords'
        )

    def __get_target_examples_identity_list_path(self,
                                                 dataset_part_name: str) -> str:
        return os.path.join(
            self.__output_dir,
            f'{dataset_part_name}-identyty_list.txt'
        )

    def __convert_examples(self,
                           tfrecord_writer: TFRecordWriter,
                           examples: List[list]) -> None:
        convert_example = partial(
            self.__convert_example,
            tfrecord_writer=tfrecord_writer
        )
        for example_id, (image_path, gt_path) in enumerate(examples):
            convert_example(example_id, image_path, gt_path)

    def __convert_example(self,
                          example_id: int,
                          image_path: str,
                          gt_path: str,
                          tfrecord_writer: TFRecordWriter) -> None:
        annotation = self.__dataset_parser.parse_example(
            example_id=example_id,
            image_path=image_path,
            gt_path=gt_path)
        example = convert_to_example(
            example_id=example_id,
            image=annotation.image,
            file_name=annotation.file_name,
            oriented_bboxes=annotation.oriented_bboxes,
            max_bounding_boxes=self.__max_bounding_boxes,
            target_size=self.__target_size,
            size_format=self.__size_format
        )
        tfrecord_writer.write(example.SerializeToString())

    def __save_examples_identity_list(self,
                                      examples: List[list],
                                      dataset_part_name: str) -> None:
        identified_examples = list(enumerate(examples))
        target_file_path = self.__get_target_examples_identity_list_path(
            dataset_part_name=dataset_part_name
        )
        dump_content_to_csv(target_file_path, identified_examples)

