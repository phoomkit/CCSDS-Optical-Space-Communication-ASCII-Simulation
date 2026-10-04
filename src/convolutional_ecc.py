"""Small, readable rate-1/3 convolutional codec for the mini-project.

The encoder uses the constraint-length-three generator polynomials
``[5, 7, 7]`` in octal.  These polynomials are used by the rate-1/3 mother
convolutional code in the CCSDS optical SCPPM coding chain.  This module only
implements the convolutional component and a hard-decision Viterbi decoder;
it is not a complete SCPPM implementation.

Bit ordering
------------
For an input bit ``u[k]`` and the two previous input bits, the encoder shift
register is ``[u[k], u[k-1], u[k-2]]``.  The encoder starts in the all-zero
state.  Appending two zero termination bits returns it to that state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Code configuration
# ---------------------------------------------------------------------------

MEMORY = 2
CONSTRAINT_LENGTH = MEMORY + 1
N_STATES = 1 << MEMORY
GENERATORS_OCTAL = (0o5, 0o7, 0o7)
CODE_RATE_NUMERATOR = 1
CODE_RATE_DENOMINATOR = len(GENERATORS_OCTAL)
TERMINATION_BITS = MEMORY


def _validate_binary_array(bits: np.ndarray, *, dimensions: tuple[int, ...]) -> np.ndarray:
    """Return ``bits`` as uint8 after checking shape and binary values."""

    array = np.asarray(bits)
    if array.ndim not in dimensions:
        expected = " or ".join(str(value) for value in dimensions)
        raise ValueError(f"Bits must have {expected} dimension(s).")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("Bits must contain only zeroes and ones.")
    return array.astype(np.uint8, copy=False)


def _parity(value: int) -> int:
    """Return the modulo-2 sum of the set bits in ``value``."""

    return value.bit_count() & 1


def _build_trellis() -> tuple[np.ndarray, np.ndarray]:
    """Construct next-state and three-output-bit lookup tables."""

    next_state = np.empty((N_STATES, 2), dtype=np.uint8)
    output_bits = np.empty((N_STATES, 2, CODE_RATE_DENOMINATOR), dtype=np.uint8)
    for state in range(N_STATES):
        for input_bit in (0, 1):
            register = (input_bit << MEMORY) | state
            next_state[state, input_bit] = (input_bit << (MEMORY - 1)) | (state >> 1)
            output_bits[state, input_bit] = [
                _parity(register & generator) for generator in GENERATORS_OCTAL
            ]
    return next_state, output_bits


NEXT_STATE, OUTPUT_BITS = _build_trellis()


# ---------------------------------------------------------------------------
# Transmitter-side functions
# ---------------------------------------------------------------------------

def append_zero_termination(bits: np.ndarray) -> np.ndarray:
    """Append two zero bits so the convolutional encoder ends in state zero."""

    vector = _validate_binary_array(bits, dimensions=(1,)).ravel()
    return np.concatenate((vector, np.zeros(TERMINATION_BITS, dtype=np.uint8)))


def convolutional_encode(bits: np.ndarray) -> np.ndarray:
    """Encode one binary vector using the rate-1/3 ``[5, 7, 7]`` code.

    Termination is deliberately separate.  Call :func:`append_zero_termination`
    first when a zero final state is required.
    """

    vector = _validate_binary_array(bits, dimensions=(1,)).ravel()
    output = np.empty(vector.size * CODE_RATE_DENOMINATOR, dtype=np.uint8)
    state = 0
    for index, input_bit in enumerate(vector):
        bit = int(input_bit)
        output[index * CODE_RATE_DENOMINATOR : (index + 1) * CODE_RATE_DENOMINATOR] = (
            OUTPUT_BITS[state, bit]
        )
        state = int(NEXT_STATE[state, bit])
    return output


def convolutional_encode_batch(bits: np.ndarray) -> np.ndarray:
    """Vectorized rate-1/3 encoding for a matrix shaped ``(blocks, bits)``."""

    matrix = _validate_binary_array(bits, dimensions=(2,))
    blocks, bit_count = matrix.shape
    output = np.empty((blocks, bit_count * CODE_RATE_DENOMINATOR), dtype=np.uint8)
    states = np.zeros(blocks, dtype=np.uint8)
    rows = np.arange(blocks)
    for index in range(bit_count):
        input_bits = matrix[:, index]
        output[:, index * CODE_RATE_DENOMINATOR : (index + 1) * CODE_RATE_DENOMINATOR] = (
            OUTPUT_BITS[states, input_bits]
        )
        states = NEXT_STATE[states, input_bits]
    return output


# ---------------------------------------------------------------------------
# Receiver-side functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViterbiResult:
    """Decoded information and diagnostic information from one Viterbi run."""

    decoded_bits: np.ndarray
    path_metric: int
    final_state: int


def _incoming_transitions() -> list[list[tuple[int, int]]]:
    """List the two predecessor transitions entering every trellis state."""

    incoming: list[list[tuple[int, int]]] = [[] for _ in range(N_STATES)]
    for previous_state in range(N_STATES):
        for input_bit in (0, 1):
            state = int(NEXT_STATE[previous_state, input_bit])
            incoming[state].append((previous_state, input_bit))
    return incoming


INCOMING_TRANSITIONS = _incoming_transitions()


def viterbi_decode_hard(
    received_bits: np.ndarray,
    *,
    terminated: bool = True,
) -> ViterbiResult:
    """Hard-decision Viterbi decode one rate-1/3 codeword.

    The initial state is fixed to zero.  When ``terminated`` is true, the
    final state is also fixed to zero and the two termination bits are removed
    from ``decoded_bits``.
    """

    vector = _validate_binary_array(received_bits, dimensions=(1,)).ravel()
    if vector.size % CODE_RATE_DENOMINATOR:
        raise ValueError("A rate-1/3 codeword length must be divisible by three.")

    symbol_count = vector.size // CODE_RATE_DENOMINATOR
    received = vector.reshape(symbol_count, CODE_RATE_DENOMINATOR)
    infinity = np.iinfo(np.int32).max // 4
    metrics = np.full(N_STATES, infinity, dtype=np.int32)
    metrics[0] = 0
    predecessors = np.empty((symbol_count, N_STATES), dtype=np.uint8)
    survivor_inputs = np.empty((symbol_count, N_STATES), dtype=np.uint8)

    for time_index, received_symbol in enumerate(received):
        next_metrics = np.full(N_STATES, infinity, dtype=np.int32)
        for state, transitions in enumerate(INCOMING_TRANSITIONS):
            for previous_state, input_bit in transitions:
                branch_metric = int(
                    np.count_nonzero(received_symbol != OUTPUT_BITS[previous_state, input_bit])
                )
                candidate = int(metrics[previous_state]) + branch_metric
                if candidate < next_metrics[state]:
                    next_metrics[state] = candidate
                    predecessors[time_index, state] = previous_state
                    survivor_inputs[time_index, state] = input_bit
        metrics = next_metrics

    final_state = 0 if terminated else int(np.argmin(metrics))
    decoded = np.empty(symbol_count, dtype=np.uint8)
    state = final_state
    for time_index in range(symbol_count - 1, -1, -1):
        decoded[time_index] = survivor_inputs[time_index, state]
        state = int(predecessors[time_index, state])

    if terminated:
        if decoded.size < TERMINATION_BITS:
            raise ValueError("The terminated codeword is too short.")
        decoded = decoded[:-TERMINATION_BITS]
    return ViterbiResult(decoded, int(metrics[final_state]), final_state)


def viterbi_decode_hard_batch(
    received_bits: np.ndarray,
    *,
    terminated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hard Viterbi decoder for ``(blocks, coded_bits)`` input.

    Returns a decoded-bit matrix and one final path metric per block.
    """

    matrix = _validate_binary_array(received_bits, dimensions=(2,))
    if matrix.shape[1] % CODE_RATE_DENOMINATOR:
        raise ValueError("A rate-1/3 codeword length must be divisible by three.")

    blocks = matrix.shape[0]
    symbol_count = matrix.shape[1] // CODE_RATE_DENOMINATOR
    received = matrix.reshape(blocks, symbol_count, CODE_RATE_DENOMINATOR)
    infinity = np.iinfo(np.int32).max // 4
    metrics = np.full((blocks, N_STATES), infinity, dtype=np.int32)
    metrics[:, 0] = 0
    predecessors = np.empty((symbol_count, blocks, N_STATES), dtype=np.uint8)
    survivor_inputs = np.empty((symbol_count, blocks, N_STATES), dtype=np.uint8)
    rows = np.arange(blocks)

    for time_index in range(symbol_count):
        next_metrics = np.full_like(metrics, infinity)
        received_symbol = received[:, time_index, :]
        for state, transitions in enumerate(INCOMING_TRANSITIONS):
            first_previous, first_input = transitions[0]
            second_previous, second_input = transitions[1]
            first = metrics[:, first_previous] + np.count_nonzero(
                received_symbol != OUTPUT_BITS[first_previous, first_input], axis=1
            )
            second = metrics[:, second_previous] + np.count_nonzero(
                received_symbol != OUTPUT_BITS[second_previous, second_input], axis=1
            )
            choose_second = second < first
            next_metrics[:, state] = np.where(choose_second, second, first)
            predecessors[time_index, :, state] = np.where(
                choose_second, second_previous, first_previous
            )
            survivor_inputs[time_index, :, state] = np.where(
                choose_second, second_input, first_input
            )
        metrics = next_metrics

    states = np.zeros(blocks, dtype=np.uint8) if terminated else np.argmin(metrics, axis=1).astype(np.uint8)
    final_states = states.copy()
    decoded = np.empty((blocks, symbol_count), dtype=np.uint8)
    for time_index in range(symbol_count - 1, -1, -1):
        decoded[:, time_index] = survivor_inputs[time_index, rows, states]
        states = predecessors[time_index, rows, states]

    final_metrics = metrics[rows, final_states]
    if terminated:
        if decoded.shape[1] < TERMINATION_BITS:
            raise ValueError("The terminated codeword is too short.")
        decoded = decoded[:, :-TERMINATION_BITS]
    return decoded, final_metrics
