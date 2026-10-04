import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import ccsds_optical_sim as sim


class TestCcsdsOpticalSimulation(unittest.TestCase):
    def test_known_crc_vector(self):
        # Independent reference vector for CCSDS 142.0-B-1 section 3.6.2.
        bits = np.array([int(ch) for ch in "1111111101001000000011101100000010011010"], dtype=np.uint8)
        self.assertEqual(sim.crc32_ccsds_optical(bits), 0xF1CA8B81)

    def test_crc_lut_matches_scalar_implementation(self):
        lut = sim.make_crc_lut_16bit()
        for value in (0x0000, 0x0001, 0x4849, 0xFFFF, 0xA55A):
            bits = sim.int_to_bits(value, 16)
            self.assertEqual(int(lut[value]), sim.crc32_ccsds_optical(bits))

    def test_ascii_segmentation_and_padding(self):
        frames, original_length, padded = sim.ascii_to_payload_frames("HELLO")
        self.assertEqual(frames.shape, (3, 16))
        self.assertEqual(original_length, 5)
        self.assertTrue(padded)
        self.assertEqual(sim.bits_to_int(frames[-1]), 0x4F00)
        self.assertEqual(sim.payload_frames_to_ascii(frames, original_length), "HELLO")

    def test_even_ascii_needs_no_padding(self):
        frames, original_length, padded = sim.ascii_to_payload_frames("TEST")
        self.assertEqual(frames.shape, (2, 16))
        self.assertEqual(original_length, 4)
        self.assertFalse(padded)

    def test_packet_layout_and_crc(self):
        payload = sim.int_to_bits(0x4849, 16)
        packet = sim.build_packet(payload)
        self.assertEqual(packet.size, 80)
        np.testing.assert_array_equal(packet[:32], sim.ASM_BITS)
        np.testing.assert_array_equal(packet[32:48], payload)
        self.assertTrue(sim.verify_packet_crc(packet))
        corrupted = packet.copy()
        corrupted[35] ^= 1
        self.assertFalse(sim.verify_packet_crc(corrupted))

    def test_ppm_mapping_guard_and_demapping(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols, slots = sim.ppm4_modulate(bits)
        np.testing.assert_array_equal(symbols, np.array([0, 1, 2, 3], dtype=np.uint8))
        expected = np.array(
            [
                1, 0, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(slots, expected)
        counts = slots.astype(int) * 100
        recovered_bits, recovered_symbols = sim.ppm4_demodulate(counts, np.random.default_rng(1))
        np.testing.assert_array_equal(recovered_symbols, symbols)
        np.testing.assert_array_equal(recovered_bits, bits)

    def test_lognormal_has_requested_mean(self):
        values = sim.unit_mean_lognormal(np.random.default_rng(2), 0.4, 300_000)
        self.assertAlmostEqual(float(np.mean(values)), 1.0, delta=0.01)

    def test_coded_message_dimensions_and_noiseless_recovery(self):
        result = sim.simulate_message(
            "HI",
            signal_photons=100.0,
            background_photons=0.0,
            sigma_ln=0.0,
            rng=np.random.default_rng(12),
            coding=sim.CODING_CONVOLUTIONAL,
        )
        record = result["records"][0]
        self.assertEqual(record["packet"].size, 80)
        self.assertEqual(record["terminated_packet"].size, 82)
        self.assertEqual(record["coded_bits"].size, 246)
        self.assertEqual(record["tx_symbols"].size, 123)
        self.assertEqual(record["tx_slots"].size, 615)
        self.assertEqual(result["recovered_message"], "HI")
        self.assertTrue(record["crc_pass"])


if __name__ == "__main__":
    unittest.main()
