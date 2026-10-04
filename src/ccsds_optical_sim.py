#!/usr/bin/env python3
"""Educational CCSDS-aligned 4-PPM optical link simulation.

The packet used here is the deliberately simplified format agreed for the
project:

    32-bit ASM + 16-bit ASCII payload + 32-bit optical CRC

The uncoded mode preserves that original frame.  The coded mode appends two
zero termination bits and applies the rate-1/3 ``[5, 7, 7]`` convolutional
component used by SCPPM, followed by hard-decision Viterbi decoding.  This is
a shortened educational model, not a complete SCPPM implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Keep Matplotlib's writable cache inside this project.  This avoids depending
# on write access to the user's global profile directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convolutional_ecc import (
    CODE_RATE_DENOMINATOR,
    GENERATORS_OCTAL,
    TERMINATION_BITS,
    append_zero_termination,
    convolutional_encode,
    convolutional_encode_batch,
    viterbi_decode_hard,
    viterbi_decode_hard_batch,
)


# CCSDS 142.0-B-1, section 3.3.2.
ASM_VALUE = 0x1ACFFC1D
ASM_LENGTH = 32

# CCSDS 142.0-B-1, section 3.6.2:
# h(x) = x^32 + x^29 + x^18 + x^14 + x^3 + 1.
# The x^32 term is implicit in the 32-bit shift-register representation.
CRC_POLY = 0x20044009
CRC_LENGTH = 32
CRC_MASK = 0xFFFFFFFF

PAYLOAD_BITS = 16
PACKET_BITS = ASM_LENGTH + PAYLOAD_BITS + CRC_LENGTH
PPM_ORDER = 4
BITS_PER_PPM_SYMBOL = 2
GUARD_SLOTS = PPM_ORDER // 4
SLOTS_PER_SYMBOL = PPM_ORDER + GUARD_SLOTS

DEFAULT_SLOT_WIDTH_NS = 512.0
DEFAULT_SIGNAL_PHOTONS = 5.0
DEFAULT_BACKGROUND_PHOTONS = 0.1
DEFAULT_SIGMA_LN = 0.4
DEFAULT_SIGNAL_SWEEP = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
DEFAULT_SEED = 2026

CODING_UNCODED = "uncoded"
CODING_CONVOLUTIONAL = "convolutional-r1-3"
CODING_CHOICES = (CODING_UNCODED, CODING_CONVOLUTIONAL)
TERMINATED_PACKET_BITS = PACKET_BITS + TERMINATION_BITS
CODED_PACKET_BITS = TERMINATED_PACKET_BITS * CODE_RATE_DENOMINATOR


# ---------------------------------------------------------------------------
# Bit utilities, ASCII conversion, framing, and CRC
# ---------------------------------------------------------------------------

def int_to_bits(value: int, width: int) -> np.ndarray:
    """Return an MSB-first uint8 bit vector."""
    return np.array([(value >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def bits_to_int(bits: Iterable[int]) -> int:
    """Convert an MSB-first bit iterable to an integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


ASM_BITS = int_to_bits(ASM_VALUE, ASM_LENGTH)


def crc32_ccsds_optical(data_bits: np.ndarray) -> int:
    """Compute the CCSDS 142.0-B-1 optical CRC-32, MSB first.

    The all-ones register initialization is equivalent to the additive term
    in the polynomial definition in section 3.6.2.  There is no final XOR.
    """
    register = CRC_MASK
    for bit in np.asarray(data_bits, dtype=np.uint8).ravel():
        feedback = ((register >> 31) & 1) ^ int(bit)
        register = (register << 1) & CRC_MASK
        if feedback:
            register ^= CRC_POLY
    return register


def make_crc_lut_16bit() -> np.ndarray:
    """Precompute the optical CRC for every possible 16-bit payload."""
    values = np.arange(1 << PAYLOAD_BITS, dtype=np.uint32)
    registers = np.full(values.shape, CRC_MASK, dtype=np.uint32)
    polynomial = np.uint32(CRC_POLY)
    mask = np.uint32(CRC_MASK)
    for shift in range(PAYLOAD_BITS - 1, -1, -1):
        input_bits = (values >> np.uint32(shift)) & np.uint32(1)
        feedback = ((registers >> np.uint32(31)) & np.uint32(1)) ^ input_bits
        registers = (registers << np.uint32(1)) & mask
        registers ^= np.where(feedback != 0, polynomial, np.uint32(0))
    return registers


def ascii_to_payload_frames(message: str) -> tuple[np.ndarray, int, bool]:
    """Encode ASCII and split it into fixed 16-bit payloads.

    An odd-length message receives one trailing 0x00 padding byte.  The
    original byte length is returned so that padding can be removed without
    losing or inventing user data after reassembly.
    """
    try:
        raw = message.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("The message must contain ASCII characters only.") from exc
    if not raw:
        raise ValueError("The message must not be empty.")

    original_length = len(raw)
    padded = bool(original_length % 2)
    if padded:
        raw += b"\x00"
    byte_matrix = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
    frames = np.unpackbits(byte_matrix, axis=1, bitorder="big")
    return frames.astype(np.uint8), original_length, padded


def payload_frames_to_ascii(payload_frames: np.ndarray, original_length: int) -> str:
    """Reassemble 16-bit payloads and remove only the known padding byte."""
    frames = np.asarray(payload_frames, dtype=np.uint8)
    raw = np.packbits(frames.reshape(-1), bitorder="big").tobytes()[:original_length]
    return raw.decode("ascii", errors="replace")


def build_packet(payload_bits: np.ndarray) -> np.ndarray:
    """Build ASM(32) + payload(16) + CRC(32)."""
    payload = np.asarray(payload_bits, dtype=np.uint8).ravel()
    if payload.size != PAYLOAD_BITS:
        raise ValueError(f"Payload must contain exactly {PAYLOAD_BITS} bits.")
    crc_bits = int_to_bits(crc32_ccsds_optical(payload), CRC_LENGTH)
    return np.concatenate((ASM_BITS, payload, crc_bits))


def build_packets_batch(payload_bits: np.ndarray, crc_lut: np.ndarray) -> np.ndarray:
    """Vectorized packet construction for BER simulation."""
    payload = np.asarray(payload_bits, dtype=np.uint8)
    if payload.ndim != 2 or payload.shape[1] != PAYLOAD_BITS:
        raise ValueError("Batch payload shape must be (n_frames, 16).")
    weights16 = (np.uint32(1) << np.arange(15, -1, -1, dtype=np.uint32))
    values = np.sum(payload.astype(np.uint32) * weights16, axis=1, dtype=np.uint32)
    crc_values = crc_lut[values]
    shifts32 = np.arange(31, -1, -1, dtype=np.uint32)
    crc_bits = ((crc_values[:, None] >> shifts32[None, :]) & np.uint32(1)).astype(np.uint8)
    asm = np.broadcast_to(ASM_BITS, (payload.shape[0], ASM_LENGTH))
    return np.concatenate((asm, payload, crc_bits), axis=1)


def verify_packet_crc(packet_bits: np.ndarray) -> bool:
    packet = np.asarray(packet_bits, dtype=np.uint8).ravel()
    if packet.size != PACKET_BITS:
        raise ValueError(f"Packet must contain exactly {PACKET_BITS} bits.")
    payload = packet[ASM_LENGTH : ASM_LENGTH + PAYLOAD_BITS]
    received_crc = bits_to_int(packet[-CRC_LENGTH:])
    return received_crc == crc32_ccsds_optical(payload)


# ---------------------------------------------------------------------------
# 4-PPM modulation and hard-decision demodulation
# ---------------------------------------------------------------------------

def ppm4_modulate(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map two bits to one of four slots and append one zero guard slot."""
    bit_vector = np.asarray(bits, dtype=np.uint8).ravel()
    if bit_vector.size % BITS_PER_PPM_SYMBOL:
        raise ValueError("The bit count must be even for 4-PPM.")
    pairs = bit_vector.reshape(-1, BITS_PER_PPM_SYMBOL)
    symbols = (pairs[:, 0] << 1) | pairs[:, 1]
    slots = np.zeros((symbols.size, SLOTS_PER_SYMBOL), dtype=np.uint8)
    slots[np.arange(symbols.size), symbols] = 1
    return symbols, slots.reshape(-1)


def ppm4_demodulate(received_counts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Demodulate by maximum photon count; randomize exact ties."""
    counts = np.asarray(received_counts)
    if counts.size % SLOTS_PER_SYMBOL:
        raise ValueError("Received count length must be a multiple of five slots.")
    count_matrix = counts.reshape(-1, SLOTS_PER_SYMBOL)
    data_counts = count_matrix[:, :PPM_ORDER]
    tie_break = rng.random(data_counts.shape) * 1e-9
    symbols = np.argmax(data_counts + tie_break, axis=1).astype(np.uint8)
    bits = np.empty(symbols.size * BITS_PER_PPM_SYMBOL, dtype=np.uint8)
    bits[0::2] = (symbols >> 1) & 1
    bits[1::2] = symbols & 1
    return bits, symbols


# ---------------------------------------------------------------------------
# Photon-counting channel and atmospheric turbulence
# ---------------------------------------------------------------------------

def unit_mean_lognormal(
    rng: np.random.Generator, sigma_ln: float, size: int | tuple[int, ...]
) -> np.ndarray:
    """Sample H with E[H]=1 and ln(H) standard deviation sigma_ln."""
    if sigma_ln < 0:
        raise ValueError("sigma_ln must be non-negative.")
    if sigma_ln == 0:
        return np.ones(size, dtype=float)
    return rng.lognormal(mean=-0.5 * sigma_ln**2, sigma=sigma_ln, size=size)


def photon_count_channel(
    slots: np.ndarray,
    signal_photons: float,
    background_photons: float,
    fading: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Poisson photon counting to binary OOK slots."""
    if signal_photons < 0 or background_photons < 0:
        raise ValueError("Photon means must be non-negative.")
    binary_slots = np.asarray(slots, dtype=np.uint8)
    lambdas = background_photons + signal_photons * np.asarray(fading) * binary_slots
    return rng.poisson(lambdas), lambdas


# ---------------------------------------------------------------------------
# End-to-end message simulations
# ---------------------------------------------------------------------------

def simulate_uncoded_message(
    message: str,
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    rng: np.random.Generator,
) -> dict:
    payloads, original_length, padded = ascii_to_payload_frames(message)
    records: list[dict] = []
    recovered_payloads: list[np.ndarray] = []

    for frame_index, payload in enumerate(payloads):
        packet = build_packet(payload)
        tx_symbols, tx_slots = ppm4_modulate(packet)
        fading = float(unit_mean_lognormal(rng, sigma_ln, 1)[0])
        counts, lambdas = photon_count_channel(
            tx_slots, signal_photons, background_photons, fading, rng
        )
        rx_packet, rx_symbols = ppm4_demodulate(counts, rng)
        rx_payload = rx_packet[ASM_LENGTH : ASM_LENGTH + PAYLOAD_BITS]
        recovered_payloads.append(rx_payload)
        records.append(
            {
                "frame_index": frame_index,
                "payload": payload,
                "packet": packet,
                "tx_symbols": tx_symbols,
                "tx_slots": tx_slots,
                "fading": fading,
                "lambdas": lambdas,
                "counts": counts,
                "rx_symbols": rx_symbols,
                "rx_packet": rx_packet,
                "crc_pass": verify_packet_crc(rx_packet),
                "bit_errors": int(np.count_nonzero(packet != rx_packet)),
                "payload_bit_errors": int(np.count_nonzero(payload != rx_payload)),
            }
        )

    recovered_matrix = np.stack(recovered_payloads)
    recovered_message = payload_frames_to_ascii(recovered_matrix, original_length)
    return {
        "coding": CODING_UNCODED,
        "message": message,
        "original_length": original_length,
        "padded": padded,
        "payloads": payloads,
        "records": records,
        "recovered_message": recovered_message,
    }


def simulate_coded_message(
    message: str,
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate the shortened rate-1/3 convolutionally coded link.

    Every 80-bit educational frame is terminated with two zero bits, encoded
    into 246 bits, mapped to 4-PPM, sent through the photon-counting channel,
    and recovered by a hard-decision Viterbi decoder.
    """

    payloads, original_length, padded = ascii_to_payload_frames(message)
    records: list[dict] = []
    recovered_payloads: list[np.ndarray] = []

    for frame_index, payload in enumerate(payloads):
        packet = build_packet(payload)
        terminated_packet = append_zero_termination(packet)
        coded_bits = convolutional_encode(terminated_packet)
        tx_symbols, tx_slots = ppm4_modulate(coded_bits)
        fading = float(unit_mean_lognormal(rng, sigma_ln, 1)[0])
        counts, lambdas = photon_count_channel(
            tx_slots, signal_photons, background_photons, fading, rng
        )
        rx_coded_bits, rx_symbols = ppm4_demodulate(counts, rng)
        decoder = viterbi_decode_hard(rx_coded_bits, terminated=True)
        rx_packet = decoder.decoded_bits
        rx_payload = rx_packet[ASM_LENGTH : ASM_LENGTH + PAYLOAD_BITS]
        recovered_payloads.append(rx_payload)
        records.append(
            {
                "frame_index": frame_index,
                "payload": payload,
                "packet": packet,
                "terminated_packet": terminated_packet,
                "coded_bits": coded_bits,
                "tx_symbols": tx_symbols,
                "tx_slots": tx_slots,
                "fading": fading,
                "lambdas": lambdas,
                "counts": counts,
                "rx_symbols": rx_symbols,
                "rx_coded_bits": rx_coded_bits,
                "rx_packet": rx_packet,
                "crc_pass": verify_packet_crc(rx_packet),
                "pre_decoder_bit_errors": int(np.count_nonzero(coded_bits != rx_coded_bits)),
                "bit_errors": int(np.count_nonzero(packet != rx_packet)),
                "payload_bit_errors": int(np.count_nonzero(payload != rx_payload)),
                "viterbi_path_metric": decoder.path_metric,
            }
        )

    recovered_matrix = np.stack(recovered_payloads)
    recovered_message = payload_frames_to_ascii(recovered_matrix, original_length)
    return {
        "coding": CODING_CONVOLUTIONAL,
        "message": message,
        "original_length": original_length,
        "padded": padded,
        "payloads": payloads,
        "records": records,
        "recovered_message": recovered_message,
    }


def simulate_message(
    message: str,
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    rng: np.random.Generator,
    coding: str = CODING_UNCODED,
) -> dict:
    """Dispatch a message simulation to the selected coding mode."""

    if coding == CODING_UNCODED:
        return simulate_uncoded_message(
            message, signal_photons, background_photons, sigma_ln, rng
        )
    if coding == CODING_CONVOLUTIONAL:
        return simulate_coded_message(
            message, signal_photons, background_photons, sigma_ln, rng
        )
    raise ValueError(f"Unsupported coding mode: {coding!r}.")


# ---------------------------------------------------------------------------
# Monte Carlo BER/FER experiments
# ---------------------------------------------------------------------------

def payload_values(payload_bits: np.ndarray) -> np.ndarray:
    weights = np.uint32(1) << np.arange(15, -1, -1, dtype=np.uint32)
    return np.sum(payload_bits.astype(np.uint32) * weights, axis=1, dtype=np.uint32)


def crc_values_from_bits(crc_bits: np.ndarray) -> np.ndarray:
    weights = np.uint32(1) << np.arange(31, -1, -1, dtype=np.uint32)
    return np.sum(crc_bits.astype(np.uint32) * weights, axis=1, dtype=np.uint32)


def simulate_ber_point(
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    n_frames: int,
    batch_size: int,
    rng: np.random.Generator,
    crc_lut: np.ndarray,
) -> dict:
    counters = {
        "frames": 0,
        "payload_bits": 0,
        "payload_bit_errors": 0,
        "payload_symbols": 0,
        "payload_symbol_errors": 0,
        "payload_frame_errors": 0,
        "crc_failures": 0,
        "undetected_payload_errors": 0,
    }

    while counters["frames"] < n_frames:
        batch = min(batch_size, n_frames - counters["frames"])
        payload = rng.integers(0, 2, size=(batch, PAYLOAD_BITS), dtype=np.uint8)
        packets = build_packets_batch(payload, crc_lut)
        pairs = packets.reshape(batch, -1, BITS_PER_PPM_SYMBOL)
        tx_symbols = ((pairs[:, :, 0] << 1) | pairs[:, :, 1]).astype(np.uint8)
        one_hot = np.eye(PPM_ORDER, dtype=np.uint8)[tx_symbols]
        fading = unit_mean_lognormal(rng, sigma_ln, (batch, 1, 1))
        lambdas = background_photons + signal_photons * fading * one_hot
        counts = rng.poisson(lambdas)
        rx_symbols = np.argmax(counts + rng.random(counts.shape) * 1e-9, axis=2).astype(np.uint8)
        rx_packets = np.empty_like(packets)
        rx_packets[:, 0::2] = (rx_symbols >> 1) & 1
        rx_packets[:, 1::2] = rx_symbols & 1

        rx_payload = rx_packets[:, ASM_LENGTH : ASM_LENGTH + PAYLOAD_BITS]
        payload_error_mask = rx_payload != payload
        frame_error_mask = np.any(payload_error_mask, axis=1)

        # The 16 payload bits occupy symbols 16 through 23 of the 40-symbol packet.
        payload_tx_symbols = tx_symbols[:, ASM_LENGTH // 2 : (ASM_LENGTH + PAYLOAD_BITS) // 2]
        payload_rx_symbols = rx_symbols[:, ASM_LENGTH // 2 : (ASM_LENGTH + PAYLOAD_BITS) // 2]

        rx_crc = crc_values_from_bits(rx_packets[:, -CRC_LENGTH:])
        expected_crc = crc_lut[payload_values(rx_payload)]
        crc_pass = rx_crc == expected_crc

        counters["frames"] += batch
        counters["payload_bits"] += batch * PAYLOAD_BITS
        counters["payload_bit_errors"] += int(np.count_nonzero(payload_error_mask))
        counters["payload_symbols"] += payload_tx_symbols.size
        counters["payload_symbol_errors"] += int(
            np.count_nonzero(payload_tx_symbols != payload_rx_symbols)
        )
        counters["payload_frame_errors"] += int(np.count_nonzero(frame_error_mask))
        counters["crc_failures"] += int(np.count_nonzero(~crc_pass))
        counters["undetected_payload_errors"] += int(np.count_nonzero(frame_error_mask & crc_pass))

    bits = counters["payload_bits"]
    frames = counters["frames"]
    symbols = counters["payload_symbols"]
    return {
        "signal_photons": signal_photons,
        "background_photons": background_photons,
        "sigma_ln": sigma_ln,
        **counters,
        "ber": counters["payload_bit_errors"] / bits,
        "ser": counters["payload_symbol_errors"] / symbols,
        "fer": counters["payload_frame_errors"] / frames,
        "crc_failure_rate": counters["crc_failures"] / frames,
        "undetected_payload_error_rate": counters["undetected_payload_errors"] / frames,
        "zero_error_95pct_upper_bound": (3.0 / bits) if counters["payload_bit_errors"] == 0 else None,
    }


def simulate_coded_ber_point(
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    n_frames: int,
    batch_size: int,
    rng: np.random.Generator,
    crc_lut: np.ndarray,
) -> dict:
    """Estimate pre/post-decoder performance for the shortened coded link."""

    counters = {
        "frames": 0,
        "payload_bits": 0,
        "payload_bit_errors": 0,
        "coded_bits": 0,
        "pre_decoder_bit_errors": 0,
        "payload_frame_errors": 0,
        "crc_failures": 0,
        "undetected_payload_errors": 0,
    }

    while counters["frames"] < n_frames:
        batch = min(batch_size, n_frames - counters["frames"])
        payload = rng.integers(0, 2, size=(batch, PAYLOAD_BITS), dtype=np.uint8)
        packets = build_packets_batch(payload, crc_lut)
        terminated = np.pad(packets, ((0, 0), (0, TERMINATION_BITS)), constant_values=0)
        coded = convolutional_encode_batch(terminated)

        pairs = coded.reshape(batch, -1, BITS_PER_PPM_SYMBOL)
        tx_symbols = ((pairs[:, :, 0] << 1) | pairs[:, :, 1]).astype(np.uint8)
        one_hot = np.eye(PPM_ORDER, dtype=np.uint8)[tx_symbols]
        fading = unit_mean_lognormal(rng, sigma_ln, (batch, 1, 1))
        lambdas = background_photons + signal_photons * fading * one_hot
        counts = rng.poisson(lambdas)
        rx_symbols = np.argmax(counts + rng.random(counts.shape) * 1e-9, axis=2).astype(np.uint8)
        rx_coded = np.empty_like(coded)
        rx_coded[:, 0::2] = (rx_symbols >> 1) & 1
        rx_coded[:, 1::2] = rx_symbols & 1

        decoded_packets, _path_metrics = viterbi_decode_hard_batch(rx_coded, terminated=True)
        rx_payload = decoded_packets[:, ASM_LENGTH : ASM_LENGTH + PAYLOAD_BITS]
        payload_error_mask = rx_payload != payload
        frame_error_mask = np.any(payload_error_mask, axis=1)
        rx_crc = crc_values_from_bits(decoded_packets[:, -CRC_LENGTH:])
        expected_crc = crc_lut[payload_values(rx_payload)]
        crc_pass = rx_crc == expected_crc

        counters["frames"] += batch
        counters["payload_bits"] += batch * PAYLOAD_BITS
        counters["payload_bit_errors"] += int(np.count_nonzero(payload_error_mask))
        counters["coded_bits"] += coded.size
        counters["pre_decoder_bit_errors"] += int(np.count_nonzero(coded != rx_coded))
        counters["payload_frame_errors"] += int(np.count_nonzero(frame_error_mask))
        counters["crc_failures"] += int(np.count_nonzero(~crc_pass))
        counters["undetected_payload_errors"] += int(np.count_nonzero(frame_error_mask & crc_pass))

    bits = counters["payload_bits"]
    coded_bits = counters["coded_bits"]
    frames = counters["frames"]
    return {
        "signal_photons": signal_photons,
        "background_photons": background_photons,
        "sigma_ln": sigma_ln,
        **counters,
        "pre_decoder_ber": counters["pre_decoder_bit_errors"] / coded_bits,
        "ber": counters["payload_bit_errors"] / bits,
        "ser": None,
        "fer": counters["payload_frame_errors"] / frames,
        "crc_failure_rate": counters["crc_failures"] / frames,
        "undetected_payload_error_rate": counters["undetected_payload_errors"] / frames,
        "zero_error_95pct_upper_bound": (3.0 / bits) if counters["payload_bit_errors"] == 0 else None,
    }


def run_ber_sweep(
    signal_sweep: Iterable[float],
    background_photons: float,
    sigma_ln: float,
    n_frames: int,
    batch_size: int,
    seed: int,
    coding: str = CODING_UNCODED,
) -> list[dict]:
    signal_sweep = tuple(signal_sweep)
    crc_lut = make_crc_lut_16bit()
    results: list[dict] = []
    models = (("Poisson only", 0.0), ("Poisson + log-normal turbulence", sigma_ln))
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(models) * len(signal_sweep))
    child_index = 0
    for model_name, model_sigma in models:
        for signal_photons in signal_sweep:
            rng = np.random.default_rng(child_seeds[child_index])
            child_index += 1
            simulator = simulate_ber_point if coding == CODING_UNCODED else simulate_coded_ber_point
            row = simulator(
                float(signal_photons), background_photons, model_sigma,
                n_frames, batch_size, rng, crc_lut
            )
            row["model"] = model_name
            row["coding"] = coding
            results.append(row)
            print(
                f"BER: {coding:20s} {model_name:35s} Ns={signal_photons:5.2f} "
                f"BER={row['ber']:.3e} FER={row['fer']:.3e}"
            )
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def save_waveform_plot(record: dict, slot_width_ns: float, output_path: Path) -> None:
    packet = record["packet"]
    tx_slots = record["tx_slots"]
    counts = record["counts"]
    rx_packet = record["rx_packet"]
    time_us = np.arange(tx_slots.size + 1) * slot_width_ns / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
    bit_index = np.arange(packet.size)
    axes[0].step(bit_index, packet, where="mid", color="#12355b")
    axes[0].axvline(31.5, color="#00a6a6", linestyle="--", label="ASM / payload")
    axes[0].axvline(47.5, color="#e59f00", linestyle="--", label="payload / CRC")
    axes[0].set(
        title="Packet before modulation: ASM(32) + payload(16) + CRC-32(32)",
        ylabel="Bit",
        xlim=(-0.5, packet.size - 0.5),
        ylim=(-0.15, 1.25),
    )
    axes[0].legend(loc="upper right")

    axes[1].stairs(tx_slots, time_us, color="#d55e00", fill=True, alpha=0.65)
    coded = "coded_bits" in record
    modulation_input = record.get("coded_bits", packet)
    axes[1].set(
        title=(
            f"After rate-1/3 encoding ({modulation_input.size} bits) and 4-PPM slot mapping"
            if coded
            else "After 4-PPM slot mapping (four data slots + one guard slot)"
        ),
        ylabel="Laser ON/OFF",
        xlabel="Time (microseconds)",
        ylim=(-0.1, 1.2),
    )

    axes[2].stairs(counts, time_us, color="#6a3d9a", fill=True, alpha=0.55)
    axes[2].set(
        title=f"Received photon counts, frame fading H={record['fading']:.3f}",
        ylabel="Photons/slot",
        xlabel="Time (microseconds)",
    )

    axes[3].step(bit_index, packet, where="mid", label="Transmitted", color="#0072b2")
    axes[3].step(
        bit_index,
        rx_packet + 0.05,
        where="mid",
        label=("Viterbi decoded (+0.05 offset)" if coded else "Demodulated (+0.05 offset)"),
        color="#e69f00",
        alpha=0.8,
    )
    error_positions = np.flatnonzero(packet != rx_packet)
    if error_positions.size:
        axes[3].scatter(error_positions, np.full(error_positions.size, 1.18), marker="x", color="red", label="Error")
    result_stage = "Viterbi decoding" if coded else "demodulation"
    pre_decoder = (
        f"; pre-decoder errors={record['pre_decoder_bit_errors']}"
        if coded
        else ""
    )
    axes[3].set(
        title=(
            f"After {result_stage}: {record['bit_errors']} frame-bit errors"
            f"{pre_decoder}; CRC pass={record['crc_pass']}"
        ),
        xlabel="Packet bit index",
        ylabel="Bit",
        xlim=(-0.5, packet.size - 0.5),
        ylim=(-0.15, 1.3),
    )
    axes[3].legend(loc="upper right")
    fig.savefig(output_path)
    plt.close(fig)


def save_histogram_plot(
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    rng: np.random.Generator,
    output_path: Path,
) -> None:
    n_samples = 50_000
    fading = unit_mean_lognormal(rng, sigma_ln, n_samples)
    on_counts = rng.poisson(background_photons + signal_photons * fading)
    off_counts = rng.poisson(background_photons, size=n_samples)
    guard_counts = rng.poisson(background_photons, size=n_samples)
    maximum = int(max(8, np.percentile(on_counts, 99.8)))
    bins = np.arange(-0.5, maximum + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.hist(off_counts, bins=bins, density=True, alpha=0.58, label="OFF data slots", color="#0072b2")
    ax.hist(guard_counts, bins=bins, density=True, alpha=0.45, label="Guard slots", color="#009e73")
    ax.hist(on_counts, bins=bins, density=True, alpha=0.58, label="ON pulse slots", color="#d55e00")
    ax.set(
        title="Photon-count histogram under Poisson noise and log-normal fading",
        xlabel="Detected photons per 512 ns slot",
        ylabel="Probability density",
        xlim=(-0.5, maximum + 0.5),
    )
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def save_eye_diagram(
    signal_photons: float,
    background_photons: float,
    sigma_ln: float,
    samples_per_slot: int,
    rng: np.random.Generator,
    output_path: Path,
) -> None:
    n_traces = 350
    trace_length = 2 * samples_per_slot
    x = np.linspace(0.0, 2.0, trace_length, endpoint=False)

    # One pulse occurs in each five-slot 4-PPM-plus-guard symbol, so an
    # arbitrary slot is ON with probability 1/5.  H is constant over both
    # slots of a trace, consistent with slow turbulence within a frame.
    slot_states = (rng.random((n_traces, 2)) < (1.0 / SLOTS_PER_SYMBOL)).astype(np.uint8)
    fading = unit_mean_lognormal(rng, sigma_ln, (n_traces, 1))
    slot_means = background_photons + signal_photons * fading * slot_states
    ideal_rate = np.repeat(slot_means, samples_per_slot, axis=1)

    # No receiver filter is applied.  A sub-slot photon count is rescaled to
    # equivalent photons per full slot.  At five photons per pulse this view
    # is intentionally sparse: that is the photon-starved regime, not a
    # smoothed analog APD waveform.
    sample_means = ideal_rate / samples_per_slot
    sampled_rate = rng.poisson(sample_means) * samples_per_slot

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharex=True)
    for trace in range(n_traces):
        axes[0].plot(
            x,
            ideal_rate[trace],
            color="#0072b2",
            alpha=0.045,
            linewidth=0.8,
            drawstyle="steps-post",
        )
    axes[1].scatter(
        np.tile(x, n_traces),
        sampled_rate.reshape(-1),
        color="#6a3d9a",
        alpha=0.025,
        s=5,
        linewidths=0,
    )
    axes[0].axhline(background_photons, color="#009e73", linewidth=1.5, label="Mean OFF level")
    axes[0].axhline(
        background_photons + signal_photons,
        color="#d55e00",
        linewidth=1.5,
        label="Mean ON level at H=1",
    )
    axes[0].legend(loc="upper right")
    axes[0].set_title("Expected received rate (turbulence, no filter)")
    axes[1].set_title("Photon-counting eye (sparse Poisson samples, no filter)")
    for ax in axes:
        ax.axvline(1.0, color="black", alpha=0.35, linestyle="--")
        ax.set_xlabel("Time (slot intervals)")
        ax.set_ylabel("Equivalent photons per slot")
        ax.set_xlim(0, 2)
    fig.suptitle(f"OOK slot eye diagram, {samples_per_slot} samples/slot")
    fig.savefig(output_path)
    plt.close(fig)


def positive_for_log(value: float, trials: int) -> float:
    return value if value > 0 else 0.5 / trials


def save_ber_plots(results: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    series = dict.fromkeys((row.get("coding", CODING_UNCODED), row["model"]) for row in results)
    for coding, model in series:
        rows = [
            row for row in results
            if row.get("coding", CODING_UNCODED) == coding and row["model"] == model
        ]
        x = np.array([row["signal_photons"] for row in rows])
        y = np.array([positive_for_log(row["ber"], row["payload_bits"]) for row in rows])
        label = f"{coding}; {model}"
        ax.semilogy(x, y, marker="o", linewidth=2, label=label)
        zero_x = [row["signal_photons"] for row in rows if row["ber"] == 0]
        zero_y = [positive_for_log(0, row["payload_bits"]) for row in rows if row["ber"] == 0]
        if zero_x:
            ax.scatter(zero_x, zero_y, facecolors="none", edgecolors="black", s=70, zorder=5)
    ax.set(
        title="Payload BER versus mean detected signal photons per ON pulse",
        xlabel="Mean signal photons per ON pulse, Ns",
        ylabel="Payload BER",
        xscale="log",
    )
    ax.legend()
    ax.text(
        0.02,
        0.03,
        "Open markers: zero observed errors, plotted at 0.5/Nbits",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(output_dir / "ber_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for coding in dict.fromkeys(row.get("coding", CODING_UNCODED) for row in results):
        rows = [
            row for row in results
            if row.get("coding", CODING_UNCODED) == coding
            and row["model"] == "Poisson + log-normal turbulence"
        ]
        x = np.array([row["signal_photons"] for row in rows])
        for key, label, marker in (
            ("fer", "Payload FER", "o"),
            ("crc_failure_rate", "CRC failure", "s"),
        ):
            y = np.array([positive_for_log(row[key], row["frames"]) for row in rows])
            ax.semilogy(x, y, marker=marker, linewidth=2, label=f"{coding}; {label}")
    ax.set(
        title="Frame and CRC behavior with log-normal turbulence",
        xlabel="Mean signal photons per ON pulse, Ns",
        ylabel="Rate",
        xscale="log",
    )
    ax.legend()
    fig.savefig(output_dir / "frame_crc_rates.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV output and command-line interface
# ---------------------------------------------------------------------------

def write_packet_csv(simulation: dict, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "frame_index",
                "coding",
                "payload_hex",
                "crc_hex",
                "fading_h",
                "pre_decoder_bit_errors",
                "packet_bit_errors",
                "payload_bit_errors",
                "crc_pass",
                "received_payload_hex",
            ),
        )
        writer.writeheader()
        for record in simulation["records"]:
            writer.writerow(
                {
                    "frame_index": record["frame_index"],
                    "coding": simulation["coding"],
                    "payload_hex": f"{bits_to_int(record['payload']):04X}",
                    "crc_hex": f"{bits_to_int(record['packet'][-CRC_LENGTH:]):08X}",
                    "fading_h": f"{record['fading']:.8f}",
                    "pre_decoder_bit_errors": record.get("pre_decoder_bit_errors", ""),
                    "packet_bit_errors": record["bit_errors"],
                    "payload_bit_errors": record["payload_bit_errors"],
                    "crc_pass": record["crc_pass"],
                    "received_payload_hex": f"{bits_to_int(record['rx_packet'][ASM_LENGTH:ASM_LENGTH + PAYLOAD_BITS]):04X}",
                }
            )


def write_ber_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    columns = list(dict.fromkeys(key for row in results for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def parse_signal_sweep(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Signal sweep must be comma-separated numbers.") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("All signal sweep values must be positive.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", help="ASCII text. If omitted, the program prompts for it.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parents[1] / "results")
    parser.add_argument(
        "--run-name",
        default="simulation",
        help="Short label included in the new timestamped result directory.",
    )
    parser.add_argument(
        "--coding",
        choices=CODING_CHOICES,
        default=CODING_CONVOLUTIONAL,
        help="Coding used for the message demonstration.",
    )
    parser.add_argument("--slot-width-ns", type=float, default=DEFAULT_SLOT_WIDTH_NS)
    parser.add_argument("--signal-photons", type=float, default=DEFAULT_SIGNAL_PHOTONS)
    parser.add_argument("--background-photons", type=float, default=DEFAULT_BACKGROUND_PHOTONS)
    parser.add_argument("--sigma-ln", type=float, default=DEFAULT_SIGMA_LN)
    parser.add_argument("--samples-per-slot", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ber-frames", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--signal-sweep",
        type=parse_signal_sweep,
        default=DEFAULT_SIGNAL_SWEEP,
        help="Comma-separated mean signal photons/pulse for the BER sweep.",
    )
    parser.add_argument("--skip-ber", action="store_true", help="Skip the Monte Carlo BER sweep.")
    parser.add_argument(
        "--compare-coding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare uncoded and convolutionally coded BER (default: enabled).",
    )
    parser.add_argument("--show", action="store_true", help="Open figures after saving them.")
    return parser


def create_run_directory(base_directory: Path, run_name: str) -> Path:
    """Create a unique timestamped directory without overwriting old results."""

    safe_name = "".join(character if character.isalnum() or character in "-_" else "-" for character in run_name)
    safe_name = safe_name.strip("-") or "simulation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = base_directory / f"{timestamp}_{safe_name}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def validate_args(args: argparse.Namespace) -> None:
    if args.slot_width_ns <= 0:
        raise ValueError("slot-width-ns must be positive.")
    if args.signal_photons < 0 or args.background_photons < 0:
        raise ValueError("Photon means must be non-negative.")
    if args.sigma_ln < 0:
        raise ValueError("sigma-ln must be non-negative.")
    if args.samples_per_slot < 2:
        raise ValueError("samples-per-slot must be at least 2.")
    if args.ber_frames <= 0 or args.batch_size <= 0:
        raise ValueError("ber-frames and batch-size must be positive.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    message = args.message if args.message is not None else input("Enter an ASCII message: ")

    run_directory = create_run_directory(args.output_dir, args.run_name)
    configure_plot_style()
    rng = np.random.default_rng(args.seed)

    simulation = simulate_message(
        message,
        args.signal_photons,
        args.background_photons,
        args.sigma_ln,
        rng,
        coding=args.coding,
    )
    save_waveform_plot(simulation["records"][0], args.slot_width_ns, run_directory / "packet_waveforms.png")
    save_histogram_plot(
        args.signal_photons,
        args.background_photons,
        args.sigma_ln,
        rng,
        run_directory / "photon_count_histogram.png",
    )
    save_eye_diagram(
        args.signal_photons,
        args.background_photons,
        args.sigma_ln,
        args.samples_per_slot,
        rng,
        run_directory / "eye_diagram.png",
    )
    write_packet_csv(simulation, run_directory / "packet_results.csv")

    ber_results: list[dict] = []
    if not args.skip_ber:
        modes = CODING_CHOICES if args.compare_coding else (args.coding,)
        for mode_index, mode in enumerate(modes):
            ber_results.extend(
                run_ber_sweep(
                    args.signal_sweep,
                    args.background_photons,
                    args.sigma_ln,
                    args.ber_frames,
                    args.batch_size,
                    args.seed + 1 + mode_index,
                    coding=mode,
                )
            )
        write_ber_csv(ber_results, run_directory / "ber_results.csv")
        save_ber_plots(ber_results, run_directory)

    transmitted_bits = CODED_PACKET_BITS if args.coding == CODING_CONVOLUTIONAL else PACKET_BITS
    packet_duration_us = transmitted_bits / BITS_PER_PPM_SYMBOL * SLOTS_PER_SYMBOL * args.slot_width_ns / 1000.0
    scintillation_index = math.exp(args.sigma_ln**2) - 1.0
    summary = {
        "scope": "Shortened CCSDS-aligned educational model; not a complete SCPPM implementation",
        "run_id": run_directory.name,
        "coding": args.coding,
        "input_message": message,
        "recovered_message": simulation["recovered_message"],
        "original_ascii_bytes": simulation["original_length"],
        "padding_added": simulation["padded"],
        "frame_count": len(simulation["records"]),
        "packet_format": "ASM(32) + payload(16) + CCSDS optical CRC-32(32)",
        "termination_bits": TERMINATION_BITS if args.coding == CODING_CONVOLUTIONAL else 0,
        "convolutional_generators_octal": [format(value, "o") for value in GENERATORS_OCTAL],
        "convolutional_rate": "1/3" if args.coding == CODING_CONVOLUTIONAL else None,
        "transmitted_bits_per_frame": transmitted_bits,
        "asm_hex": f"0x{ASM_VALUE:08X}",
        "crc_generator_polynomial": "x^32 + x^29 + x^18 + x^14 + x^3 + 1",
        "crc_polynomial_hex_without_x32": f"0x{CRC_POLY:08X}",
        "ppm_order": PPM_ORDER,
        "guard_slots_per_symbol": GUARD_SLOTS,
        "slot_width_ns": args.slot_width_ns,
        "slots_per_packet": transmitted_bits // BITS_PER_PPM_SYMBOL * SLOTS_PER_SYMBOL,
        "packet_duration_us": packet_duration_us,
        "transmitted_coded_bit_rate_bps": transmitted_bits / (packet_duration_us * 1e-6),
        "uncoded_frame_bit_rate_bps": PACKET_BITS / (packet_duration_us * 1e-6),
        "payload_rate_bps": PAYLOAD_BITS / (packet_duration_us * 1e-6),
        "message_match": message == simulation["recovered_message"],
        "total_pre_decoder_bit_errors": sum(
            record.get("pre_decoder_bit_errors", record["bit_errors"])
            for record in simulation["records"]
        ),
        "total_post_decoder_payload_bit_errors": sum(
            record["payload_bit_errors"] for record in simulation["records"]
        ),
        "all_crc_pass": all(record["crc_pass"] for record in simulation["records"]),
        "demo_signal_photons_per_on_pulse": args.signal_photons,
        "background_photons_per_slot": args.background_photons,
        "turbulence_model": "unit-mean log-normal, one independent H per frame",
        "sigma_ln_intensity": args.sigma_ln,
        "scintillation_index": scintillation_index,
        "timing": "ideal",
        "eye_samples_per_slot": args.samples_per_slot,
        "receiver_filter": "none",
        "ber_frames_per_point": None if args.skip_ber else args.ber_frames,
        "ber_payload_bits_per_point": None if args.skip_ber else args.ber_frames * PAYLOAD_BITS,
        "random_seed": args.seed,
        "standard_values": {
            "ASM, CRC polynomial, rate-1/3 [5,7,7] component, 4-PPM, guard slot, slot width": "CCSDS 141.0-B-2 and 142.0-B-2"
        },
        "educational_values_not_set_by_ccsds": {
            "signal_photon_sweep": list(args.signal_sweep),
            "demo_signal_photons": args.signal_photons,
            "background_photons": args.background_photons,
            "sigma_ln": args.sigma_ln,
        },
    }
    with (run_directory / "simulation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    (run_directory / "decoded_message.txt").write_text(
        simulation["recovered_message"] + "\n", encoding="utf-8"
    )

    print(f"Input message:     {message!r}")
    print(f"Recovered message: {simulation['recovered_message']!r}")
    print(f"Frames: {len(simulation['records'])}; padding added: {simulation['padded']}")
    print(f"Coding:            {args.coding}")
    print(f"Results written to: {run_directory.resolve()}")

    if args.show:
        for image_path in sorted(run_directory.glob("*.png")):
            try:
                os.startfile(image_path)  # type: ignore[attr-defined]
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
