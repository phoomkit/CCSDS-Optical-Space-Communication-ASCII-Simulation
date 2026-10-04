# CCSDS-aligned optical 4-PPM mini-project

โปรแกรม Python สำหรับศึกษาลิงก์สื่อสารเชิงแสงแบบ photon-counting โดยเปรียบเทียบ
ระบบ uncoded กับ convolutional ECC แบบ rate 1/3 โปรแกรมตั้งใจให้โค้ดอ่านง่ายและ
ตรวจเอาต์พุตของแต่ละบล็อกได้ ไม่ใช่โมเด็ม SCPPM เต็มมาตรฐาน

## ขอบเขตของแบบจำลอง

```text
ASCII message
  -> payload 16 bits (2 ASCII bytes)
  -> ASM(32) + payload(16) + optical CRC-32(32)
  -> append two zero termination bits
  -> rate-1/3 convolutional encoder, generators [5, 7, 7] octal
  -> 4-PPM: two coded bits per symbol
  -> four data slots + one guard slot
  -> Poisson photon counting + unit-mean log-normal turbulence
  -> maximum-count hard 4-PPM decision
  -> hard-decision Viterbi decoder
  -> CRC verification and ASCII recovery
```

โปรแกรมเก็บโหมด `uncoded` เดิมไว้เป็น baseline สำหรับเปรียบเทียบ BER และ FER
กับโหมด `convolutional-r1-3`

## ส่วนที่มาจาก CCSDS

| รายการ | ค่า |
|---|---:|
| ASM | `0x1ACFFC1D` |
| Optical CRC polynomial | `x^32 + x^29 + x^18 + x^14 + x^3 + 1` |
| Convolutional mother-code generators | `[5, 7, 7]` octal |
| Code rate ที่เลือก | `1/3` |
| Modulation | 4-PPM |
| Guard slots | `M/4 = 1` |
| Slot width | 512 ns |

อ้างอิงมาตรฐานฉบับปัจจุบัน CCSDS 141.0-B-2 และ CCSDS 142.0-B-2

## สิ่งที่ลดรูปเพื่อการศึกษา

- ใช้เฟรมสั้น `ASM(32) + payload(16) + CRC(32)` แทน Transfer Frame จริง
- Encode ทั้งเฟรม 80 บิตและ termination 2 บิต เป็น coded frame 246 บิต
- ใช้ hard-decision Viterbi decoder
- สมมติ timing และ frame boundary สมบูรณ์
- ไม่ทำ pseudo-randomizer, SCPPM interleaver, accumulator และ iterative decoder
- ไม่ทำ RS, LDPC, channel interleaver หรือ synchronization acquisition
- ไม่จำลอง pointing loss, detector bandwidth และ physical link budget

ดังนั้นชื่อที่เหมาะสมคือ **shortened CCSDS-aligned educational model** ไม่ใช่
fully CCSDS-compliant SCPPM implementation

กราฟ coded/uncoded ใช้ค่า signal photons ต่อ ON pulse เท่ากัน ระบบ coded มีจำนวน
pulses และระยะเวลาเฟรมมากกว่า จึงใช้พลังงานรวมต่อ payload มากกว่า กราฟนี้แสดง
ประโยชน์และ overhead ของ ECC ที่ operating point เดียวกัน ไม่ใช่ coding gain ที่
normalize ด้วยพลังงานต่อ information bit

## โครงสร้างโปรแกรม

```text
src/
├── ccsds_optical_sim.py   framing, modulation, channel, experiments and plots
├── convolutional_ecc.py   rate-1/3 encoder and hard Viterbi decoder
└── generate_report.py     report-result generator for the original analysis

tests/
├── test_ccsds_optical_sim.py
├── test_convolutional_ecc.py
└── test_report_math.py
```

หน้าที่สำคัญใน `convolutional_ecc.py`:

- `append_zero_termination`: เติม zero tail สองบิต
- `convolutional_encode`: encode ข้อมูลหนึ่งเฟรม
- `convolutional_encode_batch`: encode หลายเฟรมสำหรับ Monte Carlo
- `viterbi_decode_hard`: decode หนึ่งเฟรมพร้อม path metric
- `viterbi_decode_hard_batch`: decode หลายเฟรมสำหรับ BER sweep

หน้าที่สำคัญใน `ccsds_optical_sim.py`:

- `ascii_to_payload_frames`: แปลง ASCII เป็น payload 16 บิต
- `build_packet`: สร้าง ASM + payload + CRC
- `ppm4_modulate` / `ppm4_demodulate`: 4-PPM hard mapping
- `photon_count_channel`: Poisson photon-counting channel
- `simulate_uncoded_message`: จำลอง baseline
- `simulate_coded_message`: จำลองระบบที่มี ECC
- `simulate_coded_ber_point`: วัด BER ก่อนและหลัง Viterbi

## การติดตั้ง

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## การรัน

รัน coded demonstration และเปรียบเทียบ BER coded/uncoded:

```powershell
.\.venv\Scripts\python.exe .\src\ccsds_optical_sim.py `
  --message "HELLO" `
  --coding convolutional-r1-3 `
  --run-name hello-coded
```

รันเฉพาะข้อความโดยไม่ทำ BER sweep:

```powershell
.\.venv\Scripts\python.exe .\src\ccsds_optical_sim.py `
  --message "HI" `
  --coding convolutional-r1-3 `
  --skip-ber `
  --run-name hi-check
```

รัน uncoded baseline:

```powershell
.\.venv\Scripts\python.exe .\src\ccsds_optical_sim.py `
  --message "HI" `
  --coding uncoded `
  --no-compare-coding `
  --run-name hi-uncoded
```

## ผลลัพธ์แต่ละรอบ

โปรแกรมไม่เขียนทับรอบเก่า แต่สร้างโฟลเดอร์ใหม่ตาม timestamp และ run name:

```text
results/
└── 20261004_191424_619314_hello-coded/
    ├── simulation_summary.json
    ├── decoded_message.txt
    ├── packet_results.csv
    ├── ber_results.csv
    ├── packet_waveforms.png
    ├── photon_count_histogram.png
    ├── eye_diagram.png
    ├── ber_curve.png
    └── frame_crc_rates.png
```

`simulation_summary.json` ระบุ original/recovered message, seed, coding mode,
channel parameters, error ก่อนและหลัง decoder และ CRC result ทำให้ตรวจได้ว่ารูป
และตารางเป็นของการรันใด

## การทดสอบ

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

ชุดทดสอบครอบคลุม CRC reference vector, frame layout, 4-PPM, log-normal mean,
known convolutional-code sequence, termination, scalar/batch codec, Viterbi
round trip, single coded-bit correction และ coded-message dimensions

## ค่าทดลองเริ่มต้น

| ตัวแปร | ค่าเริ่มต้น | หมายเหตุ |
|---|---:|---|
| Signal photons, `Ns` | 5 photons/ON pulse | ค่าทดลอง |
| Background, `Nb` | 0.1 photons/slot | ค่าทดลอง |
| Turbulence, `sigma_ln` | 0.4 | weak-to-moderate educational case |
| Timing | ideal | ไม่จำลอง synchronization acquisition |
| Monte Carlo | 5,000 frames/point | ปรับได้ด้วย `--ber-frames` |

ค่า photon และ turbulence ไม่ใช่ค่าที่ CCSDS บังคับ เพราะขึ้นกับ link budget และ
สภาพช่องสัญญาณของภารกิจ
