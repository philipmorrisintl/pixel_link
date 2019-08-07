import unittest
from typing import List

from parameterized import parameterized
import numpy as np

from src.datasets.conversion_tools.errors import DataConversionError
from src.datasets.conversion_tools.utils import ImageSizeFormat, Number, \
    OrientedBoundingBox, ImageSize, INVALID_ORIENTED_BOX_MSG, \
    INVALID_IMAGE_SIZE_MSG, INVALID_BOX_COORDS_MSG


class ImageSizeFormatTests(unittest.TestCase):

    @parameterized.expand([
        [(10, 20), ImageSizeFormat.HEIGHT_WIDTH, (10, 20)],
        [(10, 20), ImageSizeFormat.WIDTH_HEIGHT, (20, 10)],
        [(20, 20), ImageSizeFormat.HEIGHT_WIDTH, (20, 20)],
        [(20, 20), ImageSizeFormat.WIDTH_HEIGHT, (20, 20)],
        [(20, 10), ImageSizeFormat.HEIGHT_WIDTH, (20, 10)],
        [(20, 10), ImageSizeFormat.WIDTH_HEIGHT, (10, 20)],
    ])
    def test_conversion_to_height_width(self,
                                        image_size: ImageSize,
                                        source_format: ImageSizeFormat,
                                        expected_result: ImageSize) -> None:
        converted = ImageSizeFormat.convert_to_height_width(
            image_size=image_size,
            source_format=source_format
        )
        self.assertEqual(converted, expected_result)

    @parameterized.expand([
        [(10, 20), ImageSizeFormat.HEIGHT_WIDTH, (20, 10)],
        [(10, 20), ImageSizeFormat.WIDTH_HEIGHT, (10, 20)],
        [(20, 20), ImageSizeFormat.HEIGHT_WIDTH, (20, 20)],
        [(20, 20), ImageSizeFormat.WIDTH_HEIGHT, (20, 20)],
        [(20, 10), ImageSizeFormat.HEIGHT_WIDTH, (10, 20)],
        [(20, 10), ImageSizeFormat.WIDTH_HEIGHT, (20, 10)],
    ])
    def test_conversion_to_width_height(self,
                                        image_size: ImageSize,
                                        source_format: ImageSizeFormat,
                                        expected_result: ImageSize) -> None:
        converted = ImageSizeFormat.convert_to_width_height(
            image_size=image_size,
            source_format=source_format
        )
        self.assertEqual(converted, expected_result)

    @parameterized.expand([
        [(10, 20), True],
        [(20, 20), True],
        [(20, 10), True],
        [(0, 10), False],
        [(10, 0), False],
        [(0, 0), False],
        [(-10, 10), False],
        [(10, -10), False],
        [(-10, -10), False],
        [(0, -10), False],
        [(-10, 0), False]
    ])
    def test_size_si_valid(self,
                           image_size: ImageSize,
                           expected_result: bool) -> None:
        result = ImageSizeFormat.size_is_valid(image_size=image_size)
        self.assertEqual(result, expected_result)


class OrientedBoundingBoxTests(unittest.TestCase):

    @parameterized.expand([
        [[]],
        [[], (10, 20)],
        [[], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1]],
        [[1], (10, 20)],
        [[1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2]],
        [[1, 2], (10, 20)],
        [[1, 2], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3]],
        [[1, 2, 3], (10, 20)],
        [[1, 2, 3], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3, 4]],
        [[1, 2, 3, 4], (10, 20)],
        [[1, 2, 3, 4], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3, 4, 5]],
        [[1, 2, 3, 4, 5], (10, 20)],
        [[1, 2, 3, 4, 5], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3, 4, 5, 6]],
        [[1, 2, 3, 4, 5, 6], (10, 20)],
        [[1, 2, 3, 4, 5, 6], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3, 4, 5, 6, 7]],
        [[1, 2, 3, 4, 5, 6, 7], (10, 20)],
        [[1, 2, 3, 4, 5, 6, 7], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], (10, 20)],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 0, 0, 0, 1, 1, 0, 1]],
        [[1, 0, 0, 0, 1, 1, 0, 1], (10, 20)],
        [[1, 0, 0, 0, 1, 1, 0, 1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[1, 1, 1, 0, 0, 0, 0, 1]],
        [[1, 1, 1, 0, 0, 0, 0, 1], (10, 20)],
        [[1, 1, 1, 0, 0, 0, 0, 1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 1, 1, 0, 1, 1, 0, 0]],
        [[0, 1, 1, 0, 1, 1, 0, 0], (10, 20)],
        [[0, 1, 1, 0, 1, 1, 0, 0], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 1, 1, 1, 0, 0, 1]],
        [[0, 0, 1, 1, 1, 0, 0, 1], (10, 20)],
        [[0, 0, 1, 1, 1, 0, 0, 1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 0, 1, 1, 1, 1, 0]],
        [[0, 0, 0, 1, 1, 1, 1, 0], (10, 20)],
        [[0, 0, 0, 1, 1, 1, 1, 0], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 1, 0, 0, 1, 1, 1]],
        [[0, 0, 1, 0, 0, 1, 1, 1], (10, 20)],
        [[0, 0, 1, 0, 0, 1, 1, 1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
    ])
    def test_wrong_assembly(self,
                            coords: List[Number],
                            image_size: ImageSize = None,
                            size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> None:
        context_str = self.__get_assembly_invocation_context_msg(
            coords=coords,
            image_size=image_size,
            size_format=size_format
        )
        self.assertTrue(INVALID_ORIENTED_BOX_MSG == context_str)

    @parameterized.expand([
        [[0, 0, 11, 0, 11, 1, 0, 1], (20, 10)],
        [[0, 0, 1, 0, 1, 21, 0, 21], (20, 10)],
        [[0, 0, 11, 0, 11, 21, 0, 21], (20, 10)],
        [[0, 0, 1, 0, 1, 11, 0, 11], (20, 10), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 11, 0, 11, 1, 0, 1], (10, 20), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 11, 0, 11, 21, 0, 21], (20, 10), ImageSizeFormat.WIDTH_HEIGHT],
    ])
    def test_wrong_standardization_assembly(self,
                                            coords: List[Number],
                                            image_size: ImageSize = None,
                                            size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> None:
        context_str = self.__get_assembly_invocation_context_msg(
            coords=coords,
            image_size=image_size,
            size_format=size_format
        )
        self.assertTrue(INVALID_BOX_COORDS_MSG == context_str)

    @parameterized.expand([
        [[0, 0, 11, 0, 11, 1, 0, 1], (0, 200)],
        [[0, 0, 1, 0, 1, 21, 0, 21], (100, 0)],
        [[0, 0, 11, 0, 11, 21, 0, 21], (0, 0)],
        [[0, 0, 1, 0, 1, 11, 0, 11], (200, 0), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 11, 0, 11, 1, 0, 1], (0, 100), ImageSizeFormat.WIDTH_HEIGHT],
        [[0, 0, 11, 0, 11, 21, 0, 21], (0, 0), ImageSizeFormat.WIDTH_HEIGHT],
    ])
    def test_wrong_size_assembly(self,
                                 coords: List[Number],
                                 image_size: ImageSize = None,
                                 size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> None:
        context_str = self.__get_assembly_invocation_context_msg(
            coords=coords,
            image_size=image_size,
            size_format=size_format
        )
        self.assertTrue(INVALID_IMAGE_SIZE_MSG == context_str)

    @parameterized.expand([
        [[0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1]],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0.05, 0, 0.05, 0.1, 0, 0.1],
            (10, 20)
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0.1, 0, 0.1, 0.05, 0, 0.05],
            (20, 10)
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0.05, 0, 0.05, 0.1, 0, 0.1],
            (20, 10),
            ImageSizeFormat.WIDTH_HEIGHT
        ],
        [
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0.1, 0, 0.1, 0.05, 0, 0.05],
            (10, 20),
            ImageSizeFormat.WIDTH_HEIGHT
        ]
    ])
    def test_correct_assembly(self,
                              coords: List[Number],
                              expected_result: List[Number],
                              image_size: ImageSize = None,
                              size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> None:
        bbox = OrientedBoundingBox.assembly(
                coords=coords,
                image_size=image_size,
                size_format=size_format
        )
        expected_left_top = expected_result[0], expected_result[1]
        expected_right_top = expected_result[2], expected_result[3]
        expected_right_bottom = expected_result[4], expected_result[5]
        expected_left_bottom = expected_result[6], expected_result[7]
        expected_array = np.asarray(expected_result)
        self.assertListEqual(bbox.list_form, expected_result)
        self.assertEqual(bbox.left_top, expected_left_top)
        self.assertEqual(bbox.right_top, expected_right_top)
        self.assertEqual(bbox.right_bottom, expected_right_bottom)
        self.assertEqual(bbox.left_bottom, expected_left_bottom)
        self.assertTrue(np.array_equal(bbox.array_form, expected_array))

    def __get_assembly_invocation_context_msg(self,
                                              coords: List[Number],
                                              image_size: ImageSize = None,
                                              size_format: ImageSizeFormat = ImageSizeFormat.HEIGHT_WIDTH) -> str:
        with self.assertRaises(DataConversionError) as context:
            OrientedBoundingBox.assembly(
                coords=coords,
                image_size=image_size,
                size_format=size_format
            )
        return str(context.exception)
