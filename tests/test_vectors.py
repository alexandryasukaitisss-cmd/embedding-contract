import math
import unittest

from embedding_contract.errors import EmbeddingError
from embedding_contract.vectors import assert_dimensions, is_valid_vector, l2_normalize


class VectorTests(unittest.TestCase):
    def test_l2_normalization(self):
        vector = l2_normalize([3.0, 4.0], 2)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertAlmostEqual(vector[0], 0.6)
        self.assertAlmostEqual(vector[1], 0.8)

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(EmbeddingError) as context:
            assert_dimensions([1.0, 2.0], 3)
        self.assertEqual(context.exception.code, "dimension_mismatch")

    def test_non_finite_and_zero_vectors_are_rejected(self):
        for vector in ([0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(EmbeddingError):
                l2_normalize(vector, 2)
            self.assertFalse(is_valid_vector(vector, 2))

    def test_valid_vector_helper(self):
        self.assertTrue(is_valid_vector([1, 2, 3], 3))
        self.assertFalse(is_valid_vector([1, 2], 3))
        self.assertFalse(is_valid_vector(None, 3))
        self.assertFalse(is_valid_vector([True, False, True], 3))


if __name__ == "__main__":
    unittest.main()
