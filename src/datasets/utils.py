from copy import deepcopy
from typing import List, Tuple, Dict, Optional

from tensorflow.train import Example, Features, Feature
import numpy as np

from src.datasets.config import IMG_SHAPE_KEY, \
    ORIENTED_BBOX_X1_KEY, ORIENTED_BBOX_Y1_KEY, ORIENTED_BBOX_X2_KEY, \
    ORIENTED_BBOX_Y2_KEY, ORIENTED_BBOX_X3_KEY, ORIENTED_BBOX_Y3_KEY, \
    ORIENTED_BBOX_X4_KEY, ORIENTED_BBOX_Y4_KEY, FILE_NAME_KEY, RAW_FILE_KEY
from src.datasets.wrappers import int64_feature, float_feature, bytes_feature
from src.logger.logger import get_logger


ImageShape = Tuple[int, int, int]
FeatureDict = Dict[str, Feature]
OptionalBBoxes = Optional[List[np.ndarray]]
logger = get_logger(__file__)


def convert_to_example(image: np.ndarray,
                       file_name: str,
                       oriented_bboxes: List[np.ndarray]) -> Example:
    oriented_bboxes = _check_example_health(file_name, oriented_bboxes)
    oriented_bboxes = np.asarray(oriented_bboxes)
    feature = _construct_feature_dict(
        image=image,
        file_name=file_name,
        oriented_bboxes=oriented_bboxes
    )
    example = Example(features=Features(feature=feature))
    return example


def _check_example_health(file_name: str,
                          oriented_bboxes: List[np.ndarray]) -> List[np.ndarray]:
    if len(oriented_bboxes) is 0:
        logger.warning(f'There is no bounding boxes '
                       f'attached to file {file_name}')
    oriented_bboxes = list(map(_trim_bounding_box, oriented_bboxes))
    return oriented_bboxes


def _trim_bounding_box(oriented_bounding_box: np.ndarray) -> np.ndarray:

    def __trim_standardization(elem: float) -> float:
        if elem < 0.0 or elem > 1.0:
            logger.warning('Bounding box is being clipped due to '
                           'corner position outside image. '
                           'Large number of such warnings may indicate '
                           'an issue with dataset.')
            return min(1.0, max(0.0, elem))

    return np.apply_along_axis(
        __trim_standardization,
        arr=oriented_bounding_box,
        axis=0
    )


def _construct_feature_dict(image: np.ndarray,
                            file_name: str,
                            oriented_bboxes: np.ndarray):
    image_shape = image.shape
    feature = {
        IMG_SHAPE_KEY: int64_feature(list(image_shape)),
        FILE_NAME_KEY: bytes_feature(file_name),
        RAW_FILE_KEY: bytes_feature(image.tostring())}
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
        feature[feature_key] = float_feature(bboxes_coord_column)
    return feature


def _column_to_list(array: np.ndarray, column_idx: int) -> List[np.ndarray]:
    if len(array) > 0:
        return list(array[:, column_idx])
    return []

