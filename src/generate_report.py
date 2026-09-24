#!/usr/bin/env python3
"""Generate the Section 3 simulation evidence and Thai discussion report."""

from __future__ import annotations

import csv
import inspect
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

import ccsds_optical_sim as sim


PROJECT_ROOT = Path(__file__).parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_ROOT / "figures"
DATA_DIR = RESULTS_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "report"

N_BER_FRAMES = 50_000
BATCH_SIZE = 2_500
SIGNAL_POINTS = sim.DEFAULT_SIGNAL_SWEEP
BACKGROUND_POINTS = (0.0, 0.01, 0.1, 0.5, 1.0)
TURBULENCE_POINTS = (0.0, 0.2, 0.4, 0.6, 0.8)
REPORT_SEED = 314159


def ensure_directories() -> None:
    for directory in (FIGURES_DIR, DATA_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def group_bits(bits: np.ndarray, width: int = 8) -> str:
    raw = "".join(str(int(bit)) for bit in np.asarray(bits).ravel())
    return " ".join(raw[index : index + width] for index in range(0, len(raw), width))


def make_ideal_record(message: str) -> tuple[dict, dict]:
    payloads, original_length, padded = sim.ascii_to_payload_frames(message)
    payload = payloads[0]
    packet = sim.build_packet(payload)
    symbols, slots = sim.ppm4_modulate(packet)
    counts = slots.astype(int) * 100
    rx_packet, rx_symbols = sim.ppm4_demodulate(counts, np.random.default_rng(1))
    record = {
        "frame_index": 0,
        "payload": payload,
        "packet": packet,
        "tx_symbols": symbols,
        "tx_slots": slots,
        "fading": 1.0,
        "lambdas": counts.astype(float),
        "counts": counts,
        "rx_symbols": rx_symbols,
        "rx_packet": rx_packet,
        "crc_pass": sim.verify_packet_crc(rx_packet),
        "bit_errors": int(np.count_nonzero(packet != rx_packet)),
        "payload_bit_errors": int(np.count_nonzero(payload != rx_packet[32:48])),
    }
    metadata = {
        "message": message,
        "original_length": original_length,
        "padded": padded,
        "recovered_message": sim.payload_frames_to_ascii(rx_packet[32:48].reshape(1, 16), original_length),
        "records": [record],
    }
    return metadata, record


def make_verification_cases() -> list[dict]:
    ideal_hi_meta, ideal_hi = make_ideal_record("HI")
    ideal_a_meta, ideal_a = make_ideal_record("A")

    low_meta = sim.simulate_message(
        "HI",
        signal_photons=20.0,
        background_photons=0.01,
        sigma_ln=0.0,
        rng=np.random.default_rng(REPORT_SEED),
    )
    normal_meta = sim.simulate_message(
        "HI",
        signal_photons=5.0,
        background_photons=0.1,
        sigma_ln=0.4,
        # This fixed seed deliberately produces a visible payload error in the
        # short teaching example; aggregate BER still comes from independent
        # Monte Carlo runs below.
        rng=np.random.default_rng(REPORT_SEED + 8),
    )
    return [
        {"name": "ideal_HI", "description": "Ideal deterministic channel, no noise", "meta": ideal_hi_meta, "record": ideal_hi},
        {"name": "ideal_A_padding", "description": "Ideal channel with one 0x00 padding byte", "meta": ideal_a_meta, "record": ideal_a},
        {"name": "low_noise_HI", "description": "Poisson: Ns=20, Nb=0.01, no turbulence", "meta": low_meta, "record": low_meta["records"][0]},
        {"name": "normal_HI", "description": "Poisson: Ns=5, Nb=0.1, sigma_ln=0.4", "meta": normal_meta, "record": normal_meta["records"][0]},
    ]


def write_block_trace(cases: list[dict]) -> None:
    lines = [
        "ผลเอาต์พุตทีละบล็อกของระบบ CCSDS-inspired 4-PPM",
        "=" * 72,
        "หมายเหตุ: ideal case ใช้ count=100 ใน ON slot และ count=0 ใน OFF/guard slot",
        "เพื่อพิสูจน์ความถูกต้องของ logic โดยแยกออกจากแบบจำลอง Poisson",
        "",
    ]
    csv_rows: list[dict] = []
    for case in cases:
        record = case["record"]
        meta = case["meta"]
        payload = record["payload"]
        packet = record["packet"]
        crc_value = sim.bits_to_int(packet[-32:])
        raw = meta["message"].encode("ascii")
        padded_bytes = np.packbits(payload, bitorder="big").tobytes()

        lines.extend(
            [
                f"CASE: {case['name']}",
                f"คำอธิบาย: {case['description']}",
                "-" * 72,
                f"ข้อความอินพุต                  : {meta['message']!r}",
                f"ASCII bytes                    : {' '.join(f'0x{byte:02X}' for byte in raw)}",
                f"bytes หลัง padding             : {' '.join(f'0x{byte:02X}' for byte in padded_bytes)}",
                f"มี padding                     : {meta['padded']}",
                f"ASM hex                        : 0x{sim.ASM_VALUE:08X}",
                f"ASM bits                       : {group_bits(packet[:32])}",
                f"Payload hex                    : 0x{sim.bits_to_int(payload):04X}",
                f"Payload bits                   : {group_bits(payload)}",
                f"CRC-32 hex                     : 0x{crc_value:08X}",
                f"CRC-32 bits                    : {group_bits(packet[-32:])}",
                f"Packet 80 bits                 : {group_bits(packet)}",
                f"Fading coefficient H           : {record['fading']:.6f}",
                f"จำนวน packet bit errors        : {record['bit_errors']}",
                f"จำนวน payload bit errors       : {record['payload_bit_errors']}",
                f"CRC pass                       : {record['crc_pass']}",
                f"ข้อความหลัง demodulation        : {meta['recovered_message']!r}",
                "",
                "symbol | input bits | PPM index | transmitted slots [s0 s1 s2 s3 guard] | received counts | detected index | output bits",
            ]
        )

        tx_slots = record["tx_slots"].reshape(-1, 5)
        counts = record["counts"].reshape(-1, 5)
        for index, (tx_symbol, rx_symbol) in enumerate(zip(record["tx_symbols"], record["rx_symbols"])):
            input_pair = packet[2 * index : 2 * index + 2]
            output_pair = record["rx_packet"][2 * index : 2 * index + 2]
            slot_text = " ".join(str(int(value)) for value in tx_slots[index])
            count_text = " ".join(str(int(value)) for value in counts[index])
            lines.append(
                f"{index:>6d} | {input_pair[0]}{input_pair[1]}         | {int(tx_symbol):>9d} | "
                f"[{slot_text}]                         | [{count_text}] | "
                f"{int(rx_symbol):>14d} | {output_pair[0]}{output_pair[1]}"
            )
            csv_rows.append(
                {
                    "case": case["name"],
                    "message": meta["message"],
                    "symbol_index": index,
                    "input_bits": f"{input_pair[0]}{input_pair[1]}",
                    "tx_symbol": int(tx_symbol),
                    "tx_slots": slot_text,
                    "received_counts": count_text,
                    "rx_symbol": int(rx_symbol),
                    "output_bits": f"{output_pair[0]}{output_pair[1]}",
                    "symbol_correct": bool(tx_symbol == rx_symbol),
                    "guard_count": int(counts[index, -1]),
                }
            )
        lines.extend(("", ""))

    (REPORT_DIR / "block_output_trace.txt").write_text("\n".join(lines), encoding="utf-8")
    with (DATA_DIR / "block_output_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def poisson_pmf(lam: float, maximum: int) -> np.ndarray:
    values = np.empty(maximum + 1, dtype=float)
    values[0] = math.exp(-lam)
    for k in range(1, maximum + 1):
        values[k] = values[k - 1] * lam / k
    return values


def theoretical_4ppm_ber(signal_photons: float, background_photons: float) -> float:
    """Exact uncoded BER with maximum-count detection and random tie breaking."""
    lam_on = signal_photons + background_photons
    lam_off = background_photons
    maximum = max(80, int(lam_on + 15 * math.sqrt(lam_on + 1) + 30))
    on_pmf = poisson_pmf(lam_on, maximum)
    off_pmf = poisson_pmf(lam_off, maximum)
    off_cdf = np.cumsum(off_pmf)
    probability_correct = 0.0
    for k in range(maximum + 1):
        below = 0.0 if k == 0 else float(off_cdf[k - 1])
        equal = float(off_pmf[k])
        conditional_correct = 0.0
        for ties in range(4):
            conditional_correct += (
                math.comb(3, ties)
                * equal**ties
                * below ** (3 - ties)
                / (ties + 1)
            )
        probability_correct += float(on_pmf[k]) * conditional_correct
    ser = max(0.0, 1.0 - probability_correct)
    return (2.0 / 3.0) * ser


def count_snr_linear(signal_photons: float, background_photons: float) -> float:
    return signal_photons**2 / (signal_photons + 2.0 * background_photons)


def wilson_interval(errors: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = errors / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def simulate_performance() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    main_results = sim.run_ber_sweep(
        SIGNAL_POINTS,
        background_photons=0.1,
        sigma_ln=0.4,
        n_frames=N_BER_FRAMES,
        batch_size=BATCH_SIZE,
        seed=REPORT_SEED + 100,
    )
    for row in main_results:
        low, high = wilson_interval(row["payload_bit_errors"], row["payload_bits"])
        row["ber_ci95_low"] = low
        row["ber_ci95_high"] = high
        row["count_snr_linear"] = count_snr_linear(row["signal_photons"], row["background_photons"])
        row["count_snr_db"] = 10.0 * math.log10(row["count_snr_linear"])

    crc_lut = sim.make_crc_lut_16bit()
    background_results: list[dict] = []
    for index, background in enumerate(BACKGROUND_POINTS):
        row = sim.simulate_ber_point(
            signal_photons=5.0,
            background_photons=background,
            sigma_ln=0.4,
            n_frames=N_BER_FRAMES,
            batch_size=BATCH_SIZE,
            rng=np.random.default_rng(REPORT_SEED + 200 + index),
            crc_lut=crc_lut,
        )
        row["model"] = "Ns=5, sigma_ln=0.4"
        row["scintillation_index"] = math.exp(0.4**2) - 1.0
        background_results.append(row)

    turbulence_results: list[dict] = []
    for index, sigma_ln in enumerate(TURBULENCE_POINTS):
        row = sim.simulate_ber_point(
            signal_photons=5.0,
            background_photons=0.1,
            sigma_ln=sigma_ln,
            n_frames=N_BER_FRAMES,
            batch_size=BATCH_SIZE,
            rng=np.random.default_rng(REPORT_SEED + 300 + index),
            crc_lut=crc_lut,
        )
        row["model"] = "Ns=5, Nb=0.1"
        row["scintillation_index"] = math.exp(sigma_ln**2) - 1.0
        turbulence_results.append(row)

    theory_results = []
    poisson_rows = [row for row in main_results if row["model"] == "Poisson only"]
    for row in poisson_rows:
        theoretical_ber = theoretical_4ppm_ber(row["signal_photons"], row["background_photons"])
        theory_results.append(
            {
                "signal_photons": row["signal_photons"],
                "background_photons": row["background_photons"],
                "simulation_ber": row["ber"],
                "theoretical_ber": theoretical_ber,
                "absolute_difference": abs(row["ber"] - theoretical_ber),
            }
        )
    return main_results, background_results, turbulence_results, theory_results


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_value(value: float, trials: int) -> float:
    return value if value > 0 else 0.5 / trials


def plot_performance(
    main_results: list[dict],
    background_results: list[dict],
    turbulence_results: list[dict],
    theory_results: list[dict],
) -> None:
    sim.configure_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model in ("Poisson only", "Poisson + log-normal turbulence"):
        rows = [row for row in main_results if row["model"] == model]
        x = np.array([row["signal_photons"] for row in rows])
        y = np.array([log_value(row["ber"], row["payload_bits"]) for row in rows])
        low = np.array([max(row["ber_ci95_low"], 0.25 / row["payload_bits"]) for row in rows])
        high = np.array([max(row["ber_ci95_high"], 0.25 / row["payload_bits"]) for row in rows])
        ax.semilogy(x, y, marker="o", linewidth=2, label=model)
        ax.fill_between(x, low, high, alpha=0.14)
    ax.set(xscale="log", xlabel="Mean signal photons per ON pulse, Ns", ylabel="Payload BER", title="BER versus mean detected signal photons")
    ax.legend()
    fig.savefig(FIGURES_DIR / "ber_vs_photons.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for model in ("Poisson only", "Poisson + log-normal turbulence"):
        rows = [row for row in main_results if row["model"] == model]
        x = np.array([row["count_snr_db"] for row in rows])
        y = np.array([log_value(row["ber"], row["payload_bits"]) for row in rows])
        ax.semilogy(x, y, marker="o", linewidth=2, label=model)
    ax.set(xlabel="Nominal count-domain SNR (dB)", ylabel="Payload BER", title="BER versus nominal Poisson count-domain SNR")
    ax.legend()
    fig.savefig(FIGURES_DIR / "ber_vs_snr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.array([row["background_photons"] for row in background_results])
    y = np.array([log_value(row["ber"], row["payload_bits"]) for row in background_results])
    ax.semilogy(x, y, marker="o", linewidth=2, color="#d55e00")
    ax.set(xlabel="Background photons per slot, Nb", ylabel="Payload BER", title="Effect of background photons (Ns=5, sigma_ln=0.4)")
    fig.savefig(FIGURES_DIR / "ber_vs_background.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.array([row["sigma_ln"] for row in turbulence_results])
    y = np.array([log_value(row["ber"], row["payload_bits"]) for row in turbulence_results])
    ax.semilogy(x, y, marker="o", linewidth=2, color="#6a3d9a")
    labels = [f"SI={row['scintillation_index']:.2f}" for row in turbulence_results]
    for x_value, y_value, label in zip(x, y, labels):
        ax.annotate(label, (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Log-intensity standard deviation, sigma_ln", ylabel="Payload BER", title="Effect of log-normal turbulence (Ns=5, Nb=0.1)")
    fig.savefig(FIGURES_DIR / "ber_vs_turbulence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.array([row["signal_photons"] for row in theory_results])
    simulated = np.array([log_value(row["simulation_ber"], N_BER_FRAMES * 16) for row in theory_results])
    theoretical = np.array([row["theoretical_ber"] for row in theory_results])
    ax.semilogy(x, simulated, "o-", linewidth=2, label="Monte Carlo simulation")
    ax.semilogy(x, theoretical, "--", linewidth=2, label="Exact Poisson 4-PPM theory")
    ax.set(xscale="log", xlabel="Mean signal photons per ON pulse, Ns", ylabel="Payload BER", title="Verification of simulation against uncoded 4-PPM theory")
    ax.legend()
    fig.savefig(FIGURES_DIR / "theory_vs_simulation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    rows = [row for row in main_results if row["model"] == "Poisson + log-normal turbulence"]
    x = np.array([row["signal_photons"] for row in rows])
    for key, label, marker in (
        ("fer", "Payload FER", "o"),
        ("crc_failure_rate", "CRC failure rate", "s"),
        ("undetected_payload_error_rate", "Undetected payload-error rate", "x"),
    ):
        y = np.array([log_value(row[key], row["frames"]) for row in rows])
        ax.semilogy(x, y, marker=marker, linewidth=2, label=label)
    ax.set(xscale="log", xlabel="Mean signal photons per ON pulse, Ns", ylabel="Rate", title="Frame and CRC results with turbulence")
    ax.legend()
    ax.text(
        0.02,
        0.03,
        "Zero observed undetected errors are displayed at 0.5/Nframes",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(FIGURES_DIR / "fer_crc_vs_photons.png")
    plt.close(fig)


def find_result(rows: list[dict], model: str, signal_photons: float) -> dict:
    return next(row for row in rows if row["model"] == model and row["signal_photons"] == signal_photons)


def code_section(title: str, explanation: str, function) -> str:
    return f"\n{title}\n{'-' * len(title)}\n{explanation}\n\n{inspect.getsource(function)}\n"


def write_thai_report(
    cases: list[dict],
    main_results: list[dict],
    background_results: list[dict],
    turbulence_results: list[dict],
    theory_results: list[dict],
) -> None:
    poisson_5 = find_result(main_results, "Poisson only", 5.0)
    turbulent_5 = find_result(main_results, "Poisson + log-normal turbulence", 5.0)
    theory_5 = next(row for row in theory_results if row["signal_photons"] == 5.0)
    low_case = next(case for case in cases if case["name"] == "low_noise_HI")
    normal_case = next(case for case in cases if case["name"] == "normal_HI")

    background_lines = "\n".join(
        f"- Nb={row['background_photons']:.2f} photons/slot: BER={row['ber']:.6e}"
        for row in background_results
    )
    turbulence_lines = "\n".join(
        f"- sigma_ln={row['sigma_ln']:.1f}, SI={row['scintillation_index']:.4f}: BER={row['ber']:.6e}"
        for row in turbulence_results
    )

    report = f"""3. ผลการจำลองระบบและการอภิปรายผล
=======================================

ขอบเขตของงานนี้เป็นการจำลองระบบ CCSDS-inspired optical link สำหรับการศึกษา ไม่ใช่การทำ CCSDS 142.0-B-1 SCPPM แบบเต็ม ระบบใช้แพ็กเกจ ASM 32 บิต + payload 16 บิต + CRC-32 32 บิต แล้วทำ 4-PPM พร้อม guard slot หนึ่ง slot โดยไม่ใช้ RS, LDPC, convolutional encoder, interleaver, pseudo-randomizer หรือ SCPPM inner coding

3.1 ผลการจำลองระบบที่เอาต์พุตแต่ละบล็อก
------------------------------------------------

3.1.1 การตรวจสอบด้วยข้อมูลขนาดเล็ก

ใช้ข้อความ "HI" เนื่องจากตัวอักษร ASCII หนึ่งตัวใช้ 8 บิต จึงได้ payload 16 บิตพอดี ตัวอักษร H มีค่า 0x48 หรือ 01001000 และ I มีค่า 0x49 หรือ 01001001 ดังนั้น payload เท่ากับ 0x4849

ใช้ข้อความ "A" เป็นกรณีตรวจสอบ padding ตัวอักษร A มีค่า 0x41 และมีเพียง 8 บิต โปรแกรมจึงเติม 0x00 เพื่อให้ payload ครบ 16 บิต ได้ค่า 0x4100 การประกอบข้อความกลับใช้ความยาวเดิมหนึ่งไบต์ จึงตัดเฉพาะ padding ออกและไม่ทำให้ข้อมูล A สูญหาย

ผล ideal/noiseless ใช้ค่า count=100 ใน ON slot และ count=0 ใน OFF/guard slot เพื่อแยกการตรวจสอบ logic ออกจากความสุ่มของ Poisson ผลที่ได้คือ packet bit error เท่ากับศูนย์, payload bit error เท่ากับศูนย์, CRC ผ่าน และข้อความหลัง demodulation ตรงกับอินพุตทุกประการ

รายละเอียดบิตและตารางครบทุก 4-PPM symbol อยู่ใน report/block_output_trace.txt และ results/data/block_output_table.csv ตารางดังกล่าวแสดง input bit pair, PPM index, slot vector, received counts, detected index และ output bit pair จึงสามารถตรวจย้อนกลับได้ทุกบล็อก

3.1.2 การประกอบแพ็กเกจ

ASM ใช้ค่า 0x1ACFFC1D จำนวน 32 บิต payload มี 16 บิต และ CRC มี 32 บิต รวมเป็นแพ็กเกจ 80 บิต CRC คำนวณด้วยพหุนาม x^32 + x^29 + x^18 + x^14 + x^3 + 1 โดยคำนวณครอบคลุม payload ตามแบบจำลองย่อที่กำหนดไว้

3.1.3 การทำ 4-PPM และ guard slot

บิตถูกจัดกลุ่มครั้งละ 2 บิตและแปลงเป็น index 0 ถึง 3 ได้แก่ 00->0, 01->1, 10->2 และ 11->3 จากนั้นสร้าง one-hot slot vector ความยาว 4 แล้วเติม guard slot เป็นศูนย์อีกหนึ่งตำแหน่ง ดังนั้นหนึ่งสัญลักษณ์ใช้ 5 slots แพ็กเกจ 80 บิตมี 40 สัญลักษณ์และใช้ทั้งหมด 200 slots เมื่อ slot width เท่ากับ 512 ns ระยะเวลาแพ็กเกจเท่ากับ 102.4 microseconds

3.1.4 กรณี noise ต่ำ

กำหนด Ns=20 photons/pulse, Nb=0.01 photons/slot และไม่มี turbulence ผลของเฟรมตัวอย่าง "HI" คือ packet bit errors={low_case['record']['bit_errors']}, payload bit errors={low_case['record']['payload_bit_errors']} และ CRC pass={low_case['record']['crc_pass']} ผลนี้แสดงว่าเมื่อ ON slot มีจำนวนโฟตอนสูงกว่า OFF slots ชัดเจน maximum-count detector สามารถเลือกตำแหน่ง pulse ได้ถูกต้อง

3.1.5 กรณี photon-starved และมี turbulence

กำหนด Ns=5, Nb=0.1 และ sigma_ln=0.4 ผลของเฟรมตัวอย่าง "HI" คือ fading coefficient H={normal_case['record']['fading']:.4f}, packet bit errors={normal_case['record']['bit_errors']}, payload bit errors={normal_case['record']['payload_bit_errors']} และ CRC pass={normal_case['record']['crc_pass']} ในกรณีนี้ค่าเฉลี่ยของ ON slot เท่ากับ Nb+NsH ขณะที่ OFF และ guard slots มีค่าเฉลี่ย Nb ความสุ่มของ Poisson อาจทำให้ OFF slot มี count สูงกว่า ON slot หรือเกิดการเสมอกัน จึงทำให้เลือก PPM index ผิดได้

CRC ใช้ตรวจจับข้อผิดพลาดแต่ไม่สามารถแก้ข้อผิดพลาด หาก payload ถูกต้องแต่บิตใน CRC field ผิด CRC ก็ยังสามารถรายงาน fail ได้ ในทางกลับกัน undetected error มีโอกาสต่ำมากแต่ไม่เป็นศูนย์ทางทฤษฎี

3.2 ผลการจำลองสมรรถนะ
--------------------------------

3.2.1 วิธี Monte Carlo และความละเอียดของ BER

ใช้ 50,000 เฟรมต่อหนึ่งจุด แต่ละเฟรมมี payload 16 บิต จึงเท่ากับ 800,000 payload bits ต่อจุด กราฟ BER แสดงช่วงความเชื่อมั่น 95% แบบ Wilson interval ถ้าไม่พบ error ไม่ควรสรุปว่า BER เท่ากับศูนย์ แต่ควรรายงานขอบเขตบนโดยประมาณด้วยกฎ 3/N ซึ่งในงานนี้เท่ากับ 3.75e-6 ต่อจุด

3.2.2 BER เทียบกับจำนวนโฟตอน

เมื่อ Ns เพิ่มขึ้น separation ระหว่าง ON และ OFF photon-count distributions เพิ่มขึ้น โอกาสที่ receiver จะเลือก slot ผิดจึงลดลง ที่ Ns=5 และ Nb=0.1 ผล Poisson อย่างเดียวให้ BER={poisson_5['ber']:.6e} ส่วนเมื่อเพิ่ม log-normal turbulence ที่ sigma_ln=0.4 ได้ BER={turbulent_5['ber']:.6e} แสดงว่า fading ทำให้บางเฟรมได้รับพลังงานต่ำกว่าค่าเฉลี่ยและเพิ่ม BER

3.2.3 การตรวจสอบกับทฤษฎี

กรณีไม่มี turbulence ถูกเปรียบเทียบกับสมการ exact uncoded 4-PPM Poisson detection ซึ่งรวมการสุ่มตัดสินเมื่อหลาย slots มี count เท่ากัน ที่ Ns=5 ค่าทฤษฎีเท่ากับ {theory_5['theoretical_ber']:.6e} และ simulation เท่ากับ {theory_5['simulation_ber']:.6e} ผลที่ใกล้กันช่วยยืนยันว่า Poisson generator, 4-PPM mapper และ maximum-count detector ทำงานสอดคล้องกัน

3.2.4 BER เทียบกับ SNR

เนื่องจากช่องสัญญาณนี้เป็น photon-counting Poisson channel จึงไม่ใช้ Eb/N0 แบบ AWGN โดยตรง งานนี้นิยาม nominal count-domain SNR = Ns^2/(Ns+2Nb) จากกำลังของผลต่างค่าเฉลี่ยระหว่าง ON/OFF หารด้วยผลรวม variance ของ Poisson สอง slots และแปลงเป็น dB ด้วย 10log10(SNR) นิยามนี้ใช้เพื่อจัดแกนเปรียบเทียบภายในแบบจำลอง ไม่ใช่ managed parameter ของ CCSDS

3.2.5 ผลของ background photons

เมื่อคง Ns=5 และ sigma_ln=0.4 ได้ผลดังนี้:
{background_lines}

เมื่อ Nb เพิ่ม OFF slots มีโอกาสเกิด photon counts สูงขึ้น จึงแข่งขันกับ ON slot มากขึ้นและทำให้ BER เพิ่มขึ้น

3.2.6 ผลของ atmospheric turbulence

เมื่อคง Ns=5 และ Nb=0.1 ได้ผลดังนี้:
{turbulence_lines}

ความสัมพันธ์ระหว่าง sigma_ln และ scintillation index คือ SI=exp(sigma_ln^2)-1 เมื่อ sigma_ln เพิ่ม การกระจายของ irradiance กว้างขึ้น แม้ค่าเฉลี่ย H ถูก normalize ให้เท่ากับหนึ่ง แต่จะเกิด deep fades บ่อยขึ้นและ BER สูงขึ้น

3.2.7 FER และ CRC

FER มีค่าสูงกว่า BER เพราะหนึ่งเฟรมถูกนับว่าผิดทันทีเมื่อ payload ผิดอย่างน้อยหนึ่งบิต CRC failure rate อาจสูงกว่า payload FER เพราะ CRC ครอบคลุม payload แต่ CRC field เองก็ถูกส่งผ่านช่องสัญญาณและสามารถเสียหายได้เช่นกัน เนื่องจากระบบไม่มี ECC เฟรมที่ CRC fail จะตรวจพบได้แต่ไม่สามารถซ่อมข้อมูล

3.3 โค้ด Python ที่พัฒนาขึ้น
----------------------------------

โค้ดต่อไปนี้เป็นส่วนหลักที่พัฒนาขึ้นเองและถูกเรียกใช้จริงในการสร้างผลจำลอง รายละเอียดทั้งหมดอยู่ใน src/ccsds_optical_sim.py
"""

    report += code_section(
        "3.3.1 การแบ่งข้อความ ASCII เป็น payload",
        "ฟังก์ชันนี้ตรวจว่าเป็น ASCII แท้ แบ่งครั้งละสองไบต์ และเติม 0x00 เมื่อจำนวนไบต์เป็นเลขคี่",
        sim.ascii_to_payload_frames,
    )
    report += code_section(
        "3.3.2 การคำนวณ CCSDS optical CRC-32",
        "ใช้ shift register แบบ MSB-first, initial state เป็น all ones และไม่มี final XOR",
        sim.crc32_ccsds_optical,
    )
    report += code_section(
        "3.3.3 การประกอบแพ็กเกจ",
        "นำ ASM, payload และ CRC มาต่อกันตามโครงสร้างย่อ 80 บิต",
        sim.build_packet,
    )
    report += code_section(
        "3.3.4 การทำ 4-PPM และ guard slot",
        "จับคู่สองบิตเป็นหนึ่ง symbol แล้วสร้าง one-hot slots พร้อม guard slot",
        sim.ppm4_modulate,
    )
    report += code_section(
        "3.3.5 ช่องสัญญาณ Poisson photon counting",
        "ค่าเฉลี่ย count ของ ON slot คือ Nb+NsH ส่วน OFF/guard slot คือ Nb",
        sim.photon_count_channel,
    )
    report += code_section(
        "3.3.6 การดีมอดูเลต 4-PPM",
        "receiver เลือก data slot ที่มี count สูงสุดและสุ่มเฉพาะกรณี tie",
        sim.ppm4_demodulate,
    )
    report += code_section(
        "3.3.7 การสร้าง log-normal turbulence ที่มีค่าเฉลี่ยหนึ่ง",
        "เลือก mean ของ ln(H) เป็น -sigma^2/2 เพื่อให้ E[H]=1",
        sim.unit_mean_lognormal,
    )

    report += """
4. บทสรุป
============

งานนี้พัฒนาแบบจำลองลิงก์สื่อสารด้วยแสงสำหรับการศึกษาโดยอ้างอิงแนวคิดจาก CCSDS 141.0-B-1 และ CCSDS 142.0-B-1 ข้อมูล ASCII ถูกแบ่งเป็น payload ขนาด 16 บิต เติม ASM และ CRC-32 แล้วมอดูเลตด้วย 4-PPM พร้อม guard slot จากนั้นส่งผ่านช่องสัญญาณ Poisson photon-counting ที่มี log-normal atmospheric turbulence และดีมอดูเลตด้วย maximum-count detector

ผล ideal/noiseless ยืนยันว่า ASCII conversion, padding, CRC, packet construction, 4-PPM mapping, guard insertion และ demodulation ทำงานถูกต้อง เมื่อเพิ่ม shot noise พบว่า BER ลดลงเมื่อจำนวน signal photons เพิ่มขึ้น เนื่องจาก ON slot แยกออกจาก OFF slots ได้ชัดเจนขึ้น Background photons ทำให้ OFF slots มี count สูงขึ้นและเพิ่มโอกาสเลือกตำแหน่งผิด ส่วน atmospheric turbulence ทำให้กำลังรับผันผวนและเกิด deep fades ส่งผลให้ BER และ FER สูงกว่ากรณี Poisson อย่างเดียว

CRC สามารถตรวจจับความผิดพลาดได้แต่ไม่สามารถแก้ไขข้อมูล เนื่องจากระบบนี้ไม่มี ECC การเพิ่ม RS, LDPC หรือ SCPPM จึงเป็นแนวทางสำคัญสำหรับพัฒนาต่อ นอกจากนี้แบบจำลองยังสมมติ ideal timing และยังไม่รวม slot/frame synchronization, pointing loss, detector dark count, detector bandwidth, pulse-shape distortion และ link budget จากกำลังส่งจริง ดังนั้นผลที่รายงานควรตีความเป็นผลระดับ baseband/photon-counting เพื่อศึกษาพฤติกรรมของระบบ ไม่ใช่ผลรับรองสมรรถนะของฮาร์ดแวร์หรือ CCSDS-compliant modem แบบเต็ม

ไฟล์ประกอบผลการจำลอง
------------------------
- report/block_output_trace.txt: เอาต์พุตทุกบล็อกและทุก PPM symbol
- results/data/: ตารางตัวเลขดิบและผล Monte Carlo
- results/figures/: กราฟสำหรับหัวข้อ 3.1 และ 3.2
- src/ccsds_optical_sim.py: โปรแกรมจำลองหลัก
- tests/test_ccsds_optical_sim.py: ชุดทดสอบความถูกต้องของฟังก์ชัน
"""
    (REPORT_DIR / "RESULTS_AND_DISCUSSION_TH.txt").write_text(report, encoding="utf-8")


def main() -> int:
    ensure_directories()
    sim.configure_plot_style()
    cases = make_verification_cases()
    write_block_trace(cases)

    sim.save_waveform_plot(cases[0]["record"], 512.0, FIGURES_DIR / "ideal_block_verification.png")
    sim.save_waveform_plot(cases[2]["record"], 512.0, FIGURES_DIR / "low_noise_block_verification.png")
    sim.save_waveform_plot(cases[3]["record"], 512.0, FIGURES_DIR / "normal_noise_block_verification.png")

    main_results, background_results, turbulence_results, theory_results = simulate_performance()
    write_rows(DATA_DIR / "performance_results.csv", main_results)
    write_rows(DATA_DIR / "background_sweep.csv", background_results)
    write_rows(DATA_DIR / "turbulence_sweep.csv", turbulence_results)
    write_rows(DATA_DIR / "theory_vs_simulation.csv", theory_results)
    plot_performance(main_results, background_results, turbulence_results, theory_results)
    write_thai_report(cases, main_results, background_results, turbulence_results, theory_results)

    print(f"Report written to {REPORT_DIR / 'RESULTS_AND_DISCUSSION_TH.txt'}")
    print(f"Figures written to {FIGURES_DIR}")
    print(f"Data written to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
