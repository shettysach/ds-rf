import numpy as np

from shared.g1 import MJLAB_FROM_SONIC, SONIC_FROM_MJLAB


def test_joint_mappings_are_inverses() -> None:
    values = np.arange(29)
    np.testing.assert_array_equal(values[SONIC_FROM_MJLAB][MJLAB_FROM_SONIC], values)
    np.testing.assert_array_equal(values[MJLAB_FROM_SONIC][SONIC_FROM_MJLAB], values)
