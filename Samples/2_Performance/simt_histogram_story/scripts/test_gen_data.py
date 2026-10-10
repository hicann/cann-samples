# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import unittest

import numpy as np

from gen_data import compute_histogram


class ComputeHistogramTest(unittest.TestCase):
    def test_constant_range_uses_center_bin_for_finite_float32_values(self):
        values = [
            np.float32(0.0),
            np.float32(1e20),
            np.float32(-1e20),
            np.finfo(np.float32).max,
            np.finfo(np.float32).min,
        ]
        for value in values:
            with self.subTest(value=value):
                actual = compute_histogram(np.full(4, value, dtype=np.float32), value, value, 100)
                expected = np.zeros(100, dtype=np.int32)
                expected[50] = 4
                np.testing.assert_array_equal(actual, expected)

    def test_nonconstant_range_keeps_inclusive_upper_endpoint(self):
        actual = compute_histogram(np.array([-1.0, 0.0, 1.0], dtype=np.float32), -1.0, 1.0, 100)
        expected = np.zeros(100, dtype=np.int32)
        expected[[0, 50, 99]] = 1
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
