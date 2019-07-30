from abc import ABC, abstractmethod
import os
from functools import partial
from typing import Optional, List, Tuple

from tensorflow.contrib.data import TFRecordWriter
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET

from src.common.utils import read_csv_file, extract_file_name
from src.datasets.utils import convert_to_example, OptionalBBoxes
from src.logger.logger import get_logger

logger = get_logger(__file__)


INVALID_FORMAT_ERROR_MSG = 'Some entries contain not sufficient number ' \
                           'of elements. Valid format of entry: ' \
                           'image_path, gt_path'


class DataConversionError(Exception):
    pass


class DatasetConverter(ABC):

    BBoxes = Tuple[List[np.ndarray], OptionalBBoxes]
    ParsedExample = Tuple[np.ndarray, str, List[np.ndarray], OptionalBBoxes]

    def __init__(self,
                 output_dir: str,
                 training_list_path: str,
                 test_list_path: Optional[str] = None):
        self._output_dir = output_dir
        self._training_list_path = training_list_path
        self._test_list_path = test_list_path

    def convert(self) -> None:
        self.__convert_training_set()
        if self.__test_set_should_be_converted():
            self.__convert_test_set()

    def __convert_training_set(self) -> None:
        self.__convert_dataset_part(
            examples_list_path=self._training_list_path,
            dataset_part_name='training'
        )

    def __test_set_should_be_converted(self) -> bool:
        return self._test_list_path is not None

    def __convert_test_set(self) -> None:
        self.__convert_dataset_part(
            examples_list_path=self._test_list_path,
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

    def __check_input_integrity(self, examples) -> None:
        wrong_entries = list(filter(lambda x: len(x) != 2, examples))
        if len(wrong_entries) > 0:
            raise DataConversionError(INVALID_FORMAT_ERROR_MSG)

    def __get_target_tfrecords_path(self, dataset_part_name: str) -> str:
        return os.path.join(
            self._output_dir,
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
        annotation = self._parse_example(image_path, gt_path)
        if annotation is None:
            return None
        image, file_name, oriented_bboxes, voc_bboxes = annotation
        example = convert_to_example(
            image=image,
            file_name=file_name,
            oriented_bboxes=oriented_bboxes,
            voc_bboxes=voc_bboxes)
        tfrecord_writer.write(example.SerializeToString())

    @abstractmethod
    def _parse_example(self, image_path: str, gt_path: str) -> ParsedExample:
        """
        The objective of this method is to return tuple, which elements
        will be passed as parameters of converting function invoked
        inside method __convert_example().
        """
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')


class PascalVOCConverter(DatasetConverter):

    def _parse_example(self,
                       image_path: str,
                       gt_path: str) -> DatasetConverter.ParsedExample:
        image_name = extract_file_name(image_path)
        image = cv.imread(image_path)
        annotation_parsed = ET.parse(gt_path)
        image_size = image.shape[:2]
        voc_bboxes, oriented_bboxes = self.__get_bboxes(
            annotation=annotation_parsed,
            image_size=image_size
        )
        return image, image_name, oriented_bboxes, voc_bboxes

    def __get_bboxes(self,
                     annotation: ET.ElementTree,
                     image_size: Tuple[int, int]) -> DatasetConverter.BBoxes:
        voc_bboxes, oriented_bboxes = [], []
        annotation_objects = annotation.findall('object')
        annotation_objects = list(filter(
            self.__annotation_is_valid,
            annotation_objects
        ))
        image_height, image_width = image_size
        for annotation_object in annotation_objects:
            voc_bbox = self.__extract_voc_bbox(annotation_object)
            oriented_bbox = self.__convert_voc_bbox_to_oriented(
                voc_bbox=voc_bbox,
                image_height=image_height,
                image_width=image_width)
            voc_bboxes.append(voc_bbox)
            oriented_bboxes.append(oriented_bbox)
        return voc_bboxes, oriented_bboxes

    def __annotation_is_valid(self, annotation: ET.Element):
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

    def __convert_voc_bbox_to_oriented(self,
                                       voc_bbox: np.ndarray,
                                       image_height: int,
                                       image_width: int) -> np.ndarray:
        image_height = float(image_height)
        image_width = float(image_width)
        oriented_box = [voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[1],
                        voc_bbox[2], voc_bbox[3], voc_bbox[0], voc_bbox[3]]
        standardization = ([image_width, image_height] * 4)
        oriented_box = np.asarray(oriented_box) / standardization
        return oriented_box
