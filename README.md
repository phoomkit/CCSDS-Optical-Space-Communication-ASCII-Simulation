# CCSDS-inspired optical 4-PPM simulation

โปรเจกต์สะอาดสำหรับจำลอง optical space communication ระดับการศึกษา โดยแยกออกจากงานอื่นใน workspace แล้ว

## โครงสร้าง

```text
CCSDS_Optical_Project/
├── src/
│   ├── ccsds_optical_sim.py       โปรแกรมจำลองหลัก
│   └── generate_report.py         สร้างผลหัวข้อ 3 และบทสรุป
├── tests/
│   └── test_ccsds_optical_sim.py
├── results/
│   ├── figures/                   กราฟพร้อมใช้ในรายงาน
│   └── data/                      CSV และผล Monte Carlo
├── report/
│   ├── RESULTS_AND_DISCUSSION_TH.txt
│   └── block_output_trace.txt
├── references/
│   └── CCSDS_Optical_Physical_Layer.pdf
└── requirements.txt
```

## ขอบเขตของแบบจำลอง

```text
ASCII message
  -> payload 16 บิต (2 ASCII bytes)
  -> ASM 0x1ACFFC1D จำนวน 32 บิต
  -> CCSDS optical CRC-32 จำนวน 32 บิต
  -> 4-PPM: 2 บิตต่อ symbol
  -> 4 data slots + 1 guard slot
  -> Poisson photon counting + log-normal turbulence
  -> maximum-count demodulation
  -> CRC verification และประกอบ ASCII กลับ
```

นี่เป็น CCSDS-inspired educational model ไม่ใช่ CCSDS 142.0-B-1 SCPPM implementation แบบเต็ม ระบบจงใจไม่ใช้ pseudo-randomizer, RS, LDPC, convolutional encoder, interleaver, CSM, repetition และ SCPPM inner coding

## ค่าจากมาตรฐาน

| รายการ | ค่า | แหล่งอ้างอิง |
|---|---:|---|
| ASM | `0x1ACFFC1D` | CCSDS 142.0-B-1 §3.3 |
| CRC | 32 บิต | CCSDS 142.0-B-1 §3.6 |
| CRC polynomial | `x^32 + x^29 + x^18 + x^14 + x^3 + 1` | CCSDS 142.0-B-1 §3.6.2 |
| PPM order | 4-PPM | CCSDS 142.0-B-1 §3.8.5 |
| Slot mapping | one-hot 4 slots | CCSDS 142.0-B-1 §3.12 |
| Guard slots | `M/4 = 1` | CCSDS 142.0-B-1 §3.13 |
| Slot width | 512 ns | CCSDS 141.0-B-1 Table 5-1 |

เอกสารทางการ:

- https://ccsds.org/Pubs/142x0b1.pdf
- https://ccsds.org/Pubs/141x0b1.pdf

## ค่าทดลอง

CCSDS ไม่ได้กำหนดจำนวนโฟตอนและความรุนแรงของ turbulence เป็นค่ากลาง เพราะขึ้นกับ link budget และสภาพช่องสัญญาณ โปรเจกต์จึงใช้ normalized educational scenario:

| ตัวแปร | ค่าเริ่มต้น | จุดประสงค์ |
|---|---:|---|
| Signal photons | 5 photons/ON pulse | แสดงพฤติกรรม photon-starved |
| Signal sweep | 0.1–20 photons/pulse | ครอบคลุมตั้งแต่ BER สูงจนถึงต่ำ |
| Background | 0.1 photons/slot | เพิ่ม false counts ใน OFF slots |
| Turbulence | unit-mean log-normal, `sigma_ln=0.4` | weak-to-moderate educational case |
| Scintillation index | `exp(0.4^2)-1 = 0.1735` | แปลงจาก log-intensity variance |
| Timing | ideal | ไม่จำลอง slot/frame synchronization |

Poisson detection ร่วมกับ log-normal scintillation อ้างอิงแนวทางจาก:

- W. E. Webb, *Threshold Detection in an On-Off Binary Communications Channel with Atmospheric Scintillation*, NASA-CR-120737: https://ntrs.nasa.gov/citations/19750013423
- NASA report เรื่อง Poisson detection และ log-normal atmospheric fading: https://ntrs.nasa.gov/api/citations/19820018777/downloads/19820018777.pdf

ค่าตัวเลข 5, 0.1 และ 0.4 เป็นค่าทดลอง ไม่ใช่ค่าบังคับจากแหล่งอ้างอิง

## การติดตั้ง

จากโฟลเดอร์ `C:\Users\phoom\OneDrive\Desktop\Test\CCSDS_Optical_Project`:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## รันโปรแกรมหลัก

ให้โปรแกรมถามข้อความ ASCII:

```powershell
..\.venv\Scripts\python.exe .\src\ccsds_optical_sim.py
```

หรือระบุข้อความโดยตรง:

```powershell
..\.venv\Scripts\python.exe .\src\ccsds_optical_sim.py --message HELLO
```

ผลทั่วไปจะอยู่ใน `results/`

## สร้างผลสำหรับรายงาน

```powershell
..\.venv\Scripts\python.exe .\src\generate_report.py
```

สคริปต์นี้สร้าง:

- กรณี ideal/noiseless สำหรับ `HI` และกรณี padding ด้วย `A`
- กรณี low noise และ photon-starved
- ตารางเอาต์พุตทุก 4-PPM symbol
- BER เทียบกับ photons และ nominal count-domain SNR
- BER เทียบกับ background และ turbulence
- simulation เทียบกับ exact uncoded 4-PPM Poisson theory
- FER และ CRC failure rate
- รายงานภาษาไทยหัวข้อ 3.1–3.3 และบทสรุป

## รันทดสอบ

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

ชุดทดสอบครอบคลุม CRC reference vector, padding, packet layout, 4-PPM mapping, guard slot, demodulation และค่าเฉลี่ยของ log-normal fading

## Data rate

แพ็กเกจหนึ่งชุดมี 80 บิต = 40 สัญลักษณ์ = 200 slots เมื่อ slot width เท่ากับ 512 ns:

- ระยะเวลาแพ็กเกจ = 102.4 microseconds
- package bit rate = 781.25 kb/s
- payload rate หลังรวม overhead = 156.25 kb/s

## ข้อจำกัด

- ไม่มี ECC
- CRC ตรวจจับแต่ไม่แก้ error
- ideal timing
- ไม่มี pointing loss, detector dark count หรือ detector bandwidth
- ไม่มี pulse-shape distortion และ physical link budget
- log-normal model เหมาะกับ weak-to-moderate turbulence มากกว่าสภาวะรุนแรง

