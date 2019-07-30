from copy import deepcopy
from typing import List, Tuple, Dict, Optional

from tensorflow.train import Example, Features, Feature
import numpy as np

from src.datasets.config import IMG_SHAPE_KEY, VOC_BBOX_X_MIN_KEY, \
    VOC_BBOX_Y_MIN_KEY, VOC_BBOX_X_MAX_KEY, VOC_BBOX_Y_MAX_KEY, \
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
                       oriented_bboxes: List[np.ndarray],
                       voc_bboxes: OptionalBBoxes = None) -> Example:
    _check_example_health(file_name, oriented_bboxes)
    if voc_bboxes is None:
        voc_bboxes = _convert_oriented_bboxes_to_voc(oriented_bboxes)
    oriented_bboxes = np.asarray(oriented_bboxes)
    feature = _construct_feature_dict(
        image=image,
        file_name=file_name,
        oriented_bboxes=oriented_bboxes,
        voc_bboxes=voc_bboxes
    )
    example = Example(features=Features(feature=feature))
    return example


def _check_example_health(file_name: str,
                          oriented_bboxes: List[np.ndarray]) -> None:
    if len(oriented_bboxes) is 0:
        logger.warning(f'There is no bounding boxes '
                       f'attached to file {file_name}')


def _convert_oriented_bboxes_to_voc(bboxes: List[np.ndarray]) -> np.ndarray:
    bboxes = deepcopy(bboxes)
    converted_bboxes = list(map(_convert_oriented_bbox_to_voc, bboxes))
    return np.asarray(converted_bboxes)


def _convert_oriented_bbox_to_voc(bbox: np.ndarray) -> List[float]:
    xs = bbox.reshape(4, 2)[:, 0]
    ys = bbox.reshape(4, 2)[:, 1]
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    return [x_min, y_min, x_max, y_max]


def _construct_feature_dict(image: np.ndarray,
                            file_name: str,
                            oriented_bboxes: np.ndarray,
                            voc_bboxes: np.ndarray):
    image_shape = image.shape
    feature = {
        IMG_SHAPE_KEY: int64_feature(list(image_shape)),
        FILE_NAME_KEY: bytes_feature(file_name),
        RAW_FILE_KEY: bytes_feature(image.tostring())}
    feature = _put_voc_bboxes_into_feature(
        feature=feature,
        bboxes=voc_bboxes
    )
    feature = _put_oriented_bboxes_into_feature(
        feature=feature,
        bboxes=oriented_bboxes
    )
    return feature


def _put_voc_bboxes_into_feature(feature: FeatureDict,
                                 bboxes: np.ndarray) -> FeatureDict:
    target_feature_keys = [
        VOC_BBOX_X_MIN_KEY, VOC_BBOX_Y_MIN_KEY,
        VOC_BBOX_X_MAX_KEY, VOC_BBOX_Y_MAX_KEY
    ]
    return _put_bboxes_into_feature(
        feature=feature,
        target_feature_keys=target_feature_keys,
        bboxes=bboxes
    )


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

