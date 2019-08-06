from functools import partial
from typing import List, Tuple, Dict, Optional, Union

import tensorflow as tf
from tensorflow.train import Example, Features, Feature
import numpy as np
import cv2 as cv

from src.datasets.config import IMG_HEIGHT_KEY, IMG_WIDTH_KEY, \
    ORIENTED_BBOX_X1_KEY, ORIENTED_BBOX_Y1_KEY, ORIENTED_BBOX_X2_KEY, \
    ORIENTED_BBOX_Y2_KEY, ORIENTED_BBOX_X3_KEY, ORIENTED_BBOX_Y3_KEY, \
    ORIENTED_BBOX_X4_KEY, ORIENTED_BBOX_Y4_KEY, RAW_FILE_KEY, EXAMPLE_ID_KEY, \
    BBOXES_NUMBER_KEY
from src.datasets.conversion_tools.utils import ImageSizeFormat
from src.datasets.wrappers import int64_feature, float_feature, bytes_feature
from src.logger.logger import get_logger


ImageShape = Tuple[int, int, int]
FeatureDict = Dict[str, Feature]
OptionalBBoxes = Optional[List[np.ndarray]]
DecodedGT = Tuple[
    tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
    tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
]
DecodedExampleWithID = Tuple[tf.Tensor, tf.Tensor, DecodedGT]
DecodedExampleWithoutID = Tuple[tf.Tensor, DecodedGT]
DecodedExample = Union[DecodedExampleWithID, DecodedExampleWithoutID]

logger = get_logger(__file__)


def convert_to_example(example_id: int,
                       image: np.ndarray,
                       file_name: str,
                       oriented_bboxes: List[np.ndarray],
                       max_bounding_boxes: int,
                       target_size: ImageSizeFormat.ImageSize,
                       size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> Example:
    oriented_bboxes = _check_example_health(file_name, oriented_bboxes)
    oriented_bboxes = _prepare_bboxes_array(
        oriented_bboxes=oriented_bboxes,
        max_bounding_boxes=max_bounding_boxes,
        file_name=file_name
    )
    image = _adjust_image(
        image=image,
        target_size=target_size,
        size_format=size_format
    )
    feature = _construct_feature_dict(
        example_id=example_id,
        image=image,
        oriented_bboxes=oriented_bboxes
    )
    example = Example(features=Features(feature=feature))
    return example


def _adjust_image(image: np.ndarray,
                  target_size: ImageSizeFormat.ImageSize,
                  size_format: ImageSizeFormat) -> np.ndarray:
    target_size = ImageSizeFormat.convert_to_width_height(
        image_size=target_size,
        source_format=size_format
    )
    image = cv.resize(image, target_size)
    return image.astype(dtype=np.float32)


def _prepare_bboxes_array(oriented_bboxes: List[np.ndarray],
                          max_bounding_boxes: int,
                          file_name: str) -> np.ndarray:
    bboxes_array = np.asarray(oriented_bboxes, dtype=np.float32)
    if bboxes_array.shape[0] > max_bounding_boxes:
        to_cut_off = bboxes_array.shape[0] - max_bounding_boxes
        logger.warning(f'There is to much bboxes attached '
                       f'to file {file_name}. Max number of bboxes per '
                       f'file: {max_bounding_boxes}. {to_cut_off} last '
                       f'boxes will be cut off. If you see this warning '
                       f'often - please increase \'max_bounding_boxes\' '
                       f'parameter value.')
        bboxes_array = bboxes_array[:max_bounding_boxes, :]
    if bboxes_array.shape[0] < max_bounding_boxes:
        to_pad = max_bounding_boxes - bboxes_array.shape[0]
        pad = np.zeros((to_pad, 8), dtype=np.float32)
        bboxes_array = np.concatenate((bboxes_array, pad), axis=0)
    return bboxes_array


def _check_example_health(file_name: str,
                          oriented_bboxes: List[np.ndarray]) -> List[np.ndarray]:
    if len(oriented_bboxes) is 0:
        logger.warning(f'There is no bounding boxes '
                       f'attached to file {file_name}')
    trim_bounding_box = partial(_trim_bounding_box, file_name=file_name)
    oriented_bboxes = list(map(trim_bounding_box, oriented_bboxes))
    return oriented_bboxes


def _trim_bounding_box(oriented_bounding_box: np.ndarray,
                       file_name: str) -> np.ndarray:

    def __trim_standardization(elem: float) -> float:
        if elem < 0.0 or elem > 1.0:
            logger.warning(f'Bounding box in file {file_name} is being clipped '
                           f'due to corner position outside image. '
                           f'Large number of such warnings may indicate '
                           f'an issue with dataset.')
        return min(1.0, max(0.0, elem))

    return np.vectorize(__trim_standardization)(oriented_bounding_box)


def _construct_feature_dict(example_id: int,
                            image: np.ndarray,
                            oriented_bboxes: np.ndarray):
    image_height, image_width = image.shape[:2]
    feature = {
        EXAMPLE_ID_KEY: int64_feature([example_id]),
        IMG_HEIGHT_KEY: int64_feature([image_height]),
        IMG_WIDTH_KEY: int64_feature([image_width]),
        RAW_FILE_KEY: bytes_feature(image.tostring()),
        BBOXES_NUMBER_KEY: int64_feature([len(oriented_bboxes)])
    }
    feature = _put_oriented_bboxes_into_feature(
        feature=feature,
        bboxes=oriented_bboxes
    )
    return feature


def _put_oriented_bboxes_into_feature(feature: FeatureDict,
                                      bboxes: np.ndarray) -> FeatureDict:
    target_feature_keys = [
        ORIENTED_BBOX_X1_KEY, ORIENTED_BBOX_Y1_KEY,
        ORIENTED_BBOX_X2_KEY, ORIENTED_BBOX_Y2_KEY,
        ORIENTED_BBOX_X3_KEY, ORIENTED_BBOX_Y3_KEY,
        ORIENTED_BBOX_X4_KEY, ORIENTED_BBOX_Y4_KEY
    ]
    return _put_bboxes_into_feature(
        feature=feature,
        target_feature_keys=target_feature_keys,
        bboxes=bboxes
    )


def _put_bboxes_into_feature(feature: FeatureDict,
                             target_feature_keys: List[str],
                             bboxes: np.ndarray) -> FeatureDict:
    for coord_idx, feature_key in enumerate(target_feature_keys):
        bboxes_coord_column = _column_to_list(bboxes, coord_idx)
        print(bboxes_coord_column)
        feature[feature_key] = float_feature(bboxes_coord_column)
    return feature


def _column_to_list(array: np.ndarray, column_idx: int) -> List[np.ndarray]:
    if len(array) > 0:
        return list(array[:, column_idx])
    return []


def parse_example(raw_data: bytes,
                  decode_example_id: bool = False) -> DecodedExample:
    features_to_extract = _get_features_to_extract(decode_example_id)
    features = tf.parse_example(
        raw_data,
        features=features_to_extract
    )
    x = _decode_image(features)
    y = _decode_gt(features)
    if decode_example_id is True:
        example_id = features[EXAMPLE_ID_KEY]
        return example_id, x, y
    return x, y


def _get_features_to_extract(decode_example_id: bool) -> dict:
    coordinates_features_names = [
        ORIENTED_BBOX_X1_KEY, ORIENTED_BBOX_Y1_KEY,
        ORIENTED_BBOX_X2_KEY, ORIENTED_BBOX_Y2_KEY, ORIENTED_BBOX_X3_KEY,
        ORIENTED_BBOX_Y3_KEY, ORIENTED_BBOX_X4_KEY, ORIENTED_BBOX_Y4_KEY
    ]
    int_features_names = [
        IMG_WIDTH_KEY, IMG_HEIGHT_KEY, BBOXES_NUMBER_KEY
    ]
    if decode_example_id:
        int_features_names.append(EXAMPLE_ID_KEY)
    features_to_extract = {}
    for feature_name in coordinates_features_names:
        features_to_extract[feature_name] = tf.VarLenFeature(tf.float32)
    for feature_name in int_features_names:
        features_to_extract[feature_name] = tf.FixedLenFeature([], tf.int64)
    features_to_extract[RAW_FILE_KEY] = tf.FixedLenFeature([], tf.string)
    return features_to_extract


def _decode_image(features: Dict[str, tf.Tensor]) -> tf.Tensor:
    height, width = features[IMG_HEIGHT_KEY], features[IMG_WIDTH_KEY]
    image = tf.decode_raw(features[RAW_FILE_KEY], tf.float32)
    return tf.reshape(image, [height[0], width[0], 3])


def _decode_gt(features: Dict[str, tf.Tensor]) -> DecodedGT:
    decode_single_bbox_feature = partial(
        _decode_single_bbox_feature,
        features=features
    )
    gt_feature_names = [
        ORIENTED_BBOX_X1_KEY, ORIENTED_BBOX_Y1_KEY, ORIENTED_BBOX_X2_KEY,
        ORIENTED_BBOX_Y2_KEY, ORIENTED_BBOX_X3_KEY, ORIENTED_BBOX_Y3_KEY,
        ORIENTED_BBOX_X4_KEY, ORIENTED_BBOX_Y4_KEY
    ]
    features_decoded = list(map(decode_single_bbox_feature, gt_feature_names))
    return tuple(features_decoded)


def _decode_single_bbox_feature(feature_name: str,
                                features: Dict[str, tf.Tensor]) -> tf.Tensor:
    sparse_gt_feature = features[feature_name]
    return tf.sparse_to_dense(
        sparse_indices=sparse_gt_feature.indices,
        output_shape=sparse_gt_feature.dense_shape,
        sparse_values=sparse_gt_feature.values
    )

