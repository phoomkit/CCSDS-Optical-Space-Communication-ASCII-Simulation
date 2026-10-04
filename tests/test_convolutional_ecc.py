import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import convolutional_ecc as ecc


class TestConvolutionalEcc(unittest.TestCase):
    """Unit tests for the shortened rate-1/3 ECC component."""

    def test_known_encoder_sequence(self):
        bits = np.array([1, 0, 0], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(ecc.convolutional_encode(bits), expected)

    def test_termination_returns_encoder_to_zero_state(self):
        information = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
        terminated = ecc.append_zero_termination(information)
        state = 0
        for bit in terminated:
            state = int(ecc.NEXT_STATE[state, int(bit)])
        self.assertEqual(state, 0)

    def test_noiseless_viterbi_round_trip(self):
        rng = np.random.default_rng(10)
        information = rng.integers(0, 2, size=80, dtype=np.uint8)
        codeword = ecc.convolutional_encode(ecc.append_zero_termination(information))
        decoded = ecc.viterbi_decode_hard(codeword, terminated=True)
        np.testing.assert_array_equal(decoded.decoded_bits, information)
        self.assertEqual(decoded.path_metric, 0)

    def test_viterbi_corrects_one_coded_bit_error(self):
        information = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        codeword = ecc.convolutional_encode(ecc.append_zero_termination(information))
        corrupted = codeword.copy()
        corrupted[9] ^= 1
        decoded = ecc.viterbi_decode_hard(corrupted, terminated=True)
        np.testing.assert_array_equal(decoded.decoded_bits, information)

    def test_batch_matches_scalar_codec(self):
        rng = np.random.default_rng(11)
        information = rng.integers(0, 2, size=(8, 20), dtype=np.uint8)
        terminated = np.pad(information, ((0, 0), (0, ecc.TERMINATION_BITS)))
        batch_codewords = ecc.convolutional_encode_batch(terminated)
        for row, codeword in zip(terminated, batch_codewords):
            np.testing.assert_array_equal(codeword, ecc.convolutional_encode(row))
        decoded, metrics = ecc.viterbi_decode_hard_batch(batch_codewords, terminated=True)
        np.testing.assert_array_equal(decoded, information)
        np.testing.assert_array_equal(metrics, np.zeros(information.shape[0], dtype=int))


if __name__ == "__main__":
    unittest.main()
