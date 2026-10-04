# บทพูดนำเสนอ Mini-Project

## CCSDS Optical Space Communication ASCII Simulation

เวลารวมเป้าหมาย: ประมาณ 6 นาที 35 วินาที  
รูปแบบภาษา: อธิบายภาษาไทย และใช้ English technical terms ตามสไลด์

> คำแนะนำ: ไม่ต้องอ่านชื่อกราฟหรือข้อความทุกบรรทัด ให้ชี้ส่วนที่กำลังอธิบาย และหยุดประมาณ 1 วินาทีก่อนพูดตัวเลขสำคัญ

---

## Slide 1 — ชื่อโครงงาน (ประมาณ 15 วินาที)

สวัสดีครับ วันนี้กลุ่มของผมขอนำเสนอ Mini-Project เรื่อง **การจำลองระบบสื่อสารเชิงแสงตามแนวทาง CCSDS โดยใช้ Convolutional Coding และ 4-PPM** ครับ งานนี้จำลองเส้นทางตั้งแต่ข้อมูล ASCII จนถึงการกู้คืนข้อความที่ฝั่งรับ โดยออกแบบเป็น **shortened CCSDS-aligned educational model** เพื่อให้เห็นการทำงานของแต่ละบล็อกอย่างชัดเจน ไม่ได้อ้างว่าเป็น SCPPM modem ที่ครบทุกส่วนของมาตรฐานครับ

**Transition:** ก่อนดูผลการจำลอง ผมขอเริ่มจากเป้าหมายและขอบเขตของงานครับ

---

## Slide 2 — เป้าหมายและขอบเขต (ประมาณ 40 วินาที)

เป้าหมายหลักคือจำลองการส่งข้อมูลแบบ **end-to-end** ตั้งแต่ ASCII, framing, error-control coding, 4-PPM modulation, optical channel, detection จนกลับมาเป็นข้อความ และวัดความถูกต้องด้วย **BER หรือ Bit Error Rate** และ **FER หรือ Frame Error Rate**

ส่วนที่อ้างอิงจาก CCSDS และนำมาใช้ ได้แก่ ASM, optical CRC-32, convolutional code อัตรา 1 ส่วน 3 ที่มี generator [5, 7, 7] แบบ octal, 4-PPM พร้อม guard slot และ slot width 512 nanoseconds

เพื่อให้งานอยู่ในขอบเขต Mini-Project เราไม่ได้จำลอง pseudo-randomizer, interleaver, accumulator, iterative soft decoder, synchronization acquisition และ physical link budget ดังนั้นคำที่ถูกต้องสำหรับงานนี้คือ **CCSDS-aligned model** ไม่ใช่ fully CCSDS-compliant SCPPM system ครับ

**Transition:** จากขอบเขตนี้ ระบบที่จำลองมีลำดับการทำงานดังนี้ครับ

---

## Slide 3 — Block diagram ของระบบ (ประมาณ 45 วินาที)

เริ่มจากฝั่ง **transmitter** ข้อมูล ASCII ถูกแบ่งเป็น payload ขนาด 16 บิต หรือครั้งละ 2 ตัวอักษร จากนั้นสร้าง frame โดยเติม ASM และ CRC แล้วส่งเข้า convolutional encoder ก่อนนำ coded bits ไปมอดูเลตด้วย 4-PPM

ใน **optical channel** โปรแกรมจำลองจำนวนโฟตอนที่ตรวจจับได้ด้วย Poisson distribution และจำลอง atmospheric turbulence ด้วย log-normal fading

ฝั่ง **receiver** จะเลือก data slot ที่มี photon count สูงที่สุด จากนั้นแปลงกลับเป็นบิต ใช้ hard-decision Viterbi decoder แก้ข้อผิดพลาด ตรวจ CRC และประกอบกลับเป็นข้อความ ASCII

งานนี้สมมติว่า receiver รู้ timing และ frame boundary อย่างสมบูรณ์ เพื่อแยกผลของ modulation, photon noise และ ECC ออกจากปัญหา synchronization ครับ

**Transition:** ต่อไปเป็นรายละเอียดของหนึ่ง frame และ overhead ที่เกิดขึ้นครับ

---

## Slide 4 — Frame และค่าพารามิเตอร์ (ประมาณ 50 วินาที)

หนึ่ง packet ในแบบจำลองมี **ASM 32 บิต, payload 16 บิต และ CRC-32 อีก 32 บิต** รวมเป็น 80 บิต โดย ASM ใช้ระบุจุดเริ่มต้นของ frame ส่วน CRC ใช้ตรวจจับข้อผิดพลาด แต่ CRC ไม่ได้ทำหน้าที่แก้ข้อผิดพลาด

เมื่อใช้ convolutional code เราเติม termination bits 2 บิต แล้วเข้ารหัสด้วย rate 1 ส่วน 3 จึงได้ [80 บวก 2] คูณ 3 เท่ากับ **246 coded bits** เมื่อ 4-PPM ใช้ 2 บิตต่อ symbol จึงได้ 123 symbols และแต่ละ symbol มี 4 data slots กับ 1 guard slot รวม **615 slots**

เมื่อ slot width เท่ากับ 512 nanoseconds ระยะเวลา coded frame คือ **314.88 microseconds** ขณะที่ uncoded frame ใช้ 102.4 microseconds หรือ coded frame ยาวขึ้น **3.075 เท่า** นี่คือราคาที่ต้องจ่ายเพื่อให้ ECC มีข้อมูลซ้ำสำหรับแก้ข้อผิดพลาดครับ

**ตัวอย่างสั้น:** payload 16 บิตสามารถเก็บตัวอักษร ASCII ได้ 2 ตัว เช่น “H” และ “I”

---

## Slide 5 — Channel model และ detector (ประมาณ 50 วินาที)

เนื่องจาก receiver เชิงแสงตรวจจับโฟตอน ผลลัพธ์ในแต่ละ slot จึงเป็นจำนวนเต็ม 0, 1, 2 และต่อไป แบบจำลองจึงใช้ **Poisson photon counting**

สำหรับ ON slot ค่าเฉลี่ยคือ **lambda ON เท่ากับ Nb บวก Ns คูณ H** โดย Ns คือจำนวน signal photons เฉลี่ยต่อ pulse, Nb คือ background photons ต่อ slot และ H คือ fading coefficient ส่วน OFF slot และ guard slot มีค่าเฉลี่ยเท่ากับ Nb

ค่า H ใช้ log-normal distribution ที่ปรับให้มีค่าเฉลี่ยเท่ากับ 1 เพื่อแทน atmospheric turbulence กล่าวง่าย ๆ คือ turbulence ทำให้ความเข้มแสงแกว่ง แม้กำลังส่งเฉลี่ยเท่าเดิม บาง frame อาจเกิด **deep fade** และรับโฟตอนได้น้อยมาก

receiver ใช้ maximum-count detector เลือก data slot ที่มี count สูงที่สุด ถ้า histogram ของ ON และ OFF ซ้อนกัน ก็มีโอกาสเลือกตำแหน่ง pulse ผิด ส่วน eye diagram ในสไลด์เป็นภาพประกอบ เพราะโมเดลนี้ยังไม่มี receiver filter, bandwidth limitation หรือ timing jitter ครับ

---

## Slide 6 — การตรวจสอบ End-to-End (ประมาณ 50 วินาที)

สไลด์นี้แสดงตัวอย่างส่งข้อความ **“HI”** โดยใช้ seed 2026, Ns เท่ากับ 5, Nb เท่ากับ 0.1 และ log-normal sigma เท่ากับ 0.4

เส้นทางเริ่มจาก packet 80 บิต เข้ารหัสเป็น 246 บิต แล้วทำ 4-PPM และส่งผ่าน photon-counting channel ในตัวอย่างนี้ หลัง detector มี coded bits ผิด **8 บิตก่อนเข้า Viterbi decoder** แต่ Viterbi ใช้โครงสร้าง trellis และเลือกเส้นทางที่มี accumulated Hamming distance ต่ำที่สุด จึงกู้ payload กลับมาได้โดยไม่มี bit error

ผลสุดท้าย CRC ผ่าน และข้อความที่รับได้คือ “HI” ตรงกับข้อมูลต้นฉบับ ตัวอย่างนี้ยืนยันการทำงานของทั้ง chain แต่เป็นเพียงหนึ่ง frame จึงยังใช้สรุปสมรรถนะเชิงสถิติไม่ได้ ผลเชิงสถิติจะแสดงในสไลด์ถัดไปครับ

---

## Slide 7 — ผลของ ECC ต่อ BER (ประมาณ 65 วินาที)

กราฟนี้มาจาก Monte Carlo simulation จำนวน **50,000 frames ต่อหนึ่งจุด** หรือ 800,000 payload bits ต่อจุด แกนนอนคือ Ns หรือ signal photons ต่อ ON pulse ส่วนแกนตั้งคือ log base 10 ของ payload BER ดังนั้นเส้นที่อยู่ต่ำกว่าหมายถึง error ต่ำกว่า

ที่ Ns เท่ากับ 5 และ Nb เท่ากับ 0.1 ในช่องสัญญาณ Poisson อย่างเดียว ระบบ uncoded มี BER ประมาณ **6.81 คูณ 10 ยกกำลังลบ 3** ส่วนระบบ coded มี BER **4 คูณ 10 ยกกำลังลบ 5** จึงลด BER ได้ประมาณ **170 เท่า**

เมื่อเพิ่ม turbulence ที่ sigma เท่ากับ 0.4 ค่า BER ของ uncoded เป็นประมาณ **1.83 คูณ 10 ยกกำลังลบ 2** และ coded เป็น **2.81 คูณ 10 ยกกำลังลบ 3** หรือลดลงประมาณ **6.52 เท่า** ผลดีของ ECC จึงยังเห็นได้ แต่ turbulence ทำให้ deep fade ซึ่งแก้ยากกว่าความผิดพลาดแบบกระจายทั่วไป

อย่างไรก็ตาม การเปรียบเทียบนี้กำหนด Ns ต่อ pulse เท่ากัน ระบบ coded ส่ง pulse มากกว่าและ frame ยาวขึ้น 3.075 เท่า จึงควรเรียกว่า **BER improvement under equal photons per pulse** ไม่ใช่ energy-normalized coding gain ครับ

---

## Slide 8 — ผลของ Background และ Turbulence (ประมาณ 50 วินาที)

สไลด์นี้แยกดูผลของ channel parameters โดยใช้ uncoded baseline และกำหนด Ns เท่ากับ 5

กราฟซ้ายแสดงว่าเมื่อ **Nb เพิ่มขึ้น** BER จะสูงขึ้น เพราะ OFF slots มี background photon count มากขึ้น จึงมีโอกาสแซง ON slot และทำให้ detector เลือก PPM position ผิด ตัวอย่างง่าย ๆ คือ ถ้า ON slot ได้ 2 photons แต่ OFF slot บังเอิญได้ 3 photons receiver ก็จะตัดสินใจผิด แม้ตำแหน่ง pulse ที่ส่งมาจะถูกต้อง

กราฟขวาแสดงว่าเมื่อ **log-normal sigma เพิ่มขึ้น** BER จะสูงขึ้นเช่นกัน เพราะความเข้มแสงกระจายกว้างขึ้นและเกิด deep fades บ่อยขึ้น

ผลเหล่านี้เป็นผลระดับ baseband และ photon counting ภายใต้ ideal timing โมเดลยังไม่รวม pointing loss, detector bandwidth, optical filter, synchronization error และ physical link budget จึงควรตีความภายในขอบเขตนี้ครับ

---

## Slide 9 — บทสรุป (ประมาณ 30 วินาที)

สรุปได้สามประเด็นครับ หนึ่ง ระบบสามารถจำลองเส้นทางจาก ASCII ผ่าน framing, ECC, 4-PPM และ optical channel จนกู้คืนข้อความและตรวจ CRC ได้ สอง convolutional ECC ลด BER และ FER ได้ชัดเจน แต่แลกกับ frame duration และจำนวน pulse ที่เพิ่มขึ้น และสาม แบบจำลองนี้เหมาะกับการศึกษาแต่ละบล็อกและการทดลองผลของ channel parameters

หากพัฒนาต่อ ควรเพิ่ม pseudo-randomizer, SCPPM interleaver, accumulator, soft iterative decoding, synchronization และเปรียบเทียบแบบ normalize พลังงานต่อ information bit เพื่อเข้าใกล้ระบบตามมาตรฐานมากขึ้นครับ ขอบคุณครับ

---

# คำถามที่อาจารย์อาจถาม

## 1. ทำไมใช้ Ns แทน SNR?

ระบบนี้จำลอง receiver แบบ photon counting โดยตรง จึงใช้จำนวน signal photons ต่อ ON pulse หรือ Ns และ background photons ต่อ slot หรือ Nb เป็นตัวแปรทางกายภาพที่สัมพันธ์กับ Poisson distribution โดยตรง การแปลงเป็น electrical SNR ต้องกำหนด detector gain, responsivity, bandwidth และ noise sources เพิ่มเติม ซึ่งอยู่นอกขอบเขตของแบบจำลองนี้

## 2. งานนี้ตรงตาม CCSDS หรือไม่?

งานนี้ใช้พารามิเตอร์และองค์ประกอบที่อ้างอิง CCSDS ได้แก่ ASM, optical CRC-32, convolutional component [5, 7, 7], 4-PPM, guard slot และ slot width 512 ns แต่ใช้ shortened frame และไม่ได้ทำ SCPPM coding chain ครบทั้งหมด จึงเรียกว่า CCSDS-aligned educational model ไม่ใช่ fully compliant implementation

## 3. CRC และ ECC ต่างกันอย่างไร?

CRC ใช้ตรวจจับว่าข้อมูลผิดหรือไม่ แต่ไม่แก้ข้อมูล ส่วน ECC เพิ่มข้อมูลซ้ำเพื่อให้ decoder สามารถประมาณและแก้ข้อผิดพลาดได้ ในระบบนี้ Viterbi decoder ทำการแก้ก่อน แล้ว CRC ใช้ยืนยันความถูกต้องของ frame ที่กู้คืนมา

## 4. ทำไม ECC ลด BER ได้น้อยลงเมื่อมี turbulence?

Convolutional code จัดการความผิดพลาดแบบกระจายได้ดี แต่ deep fade สามารถทำให้หลาย symbols ใน frame ได้รับสัญญาณอ่อนพร้อมกัน เกิด error ที่สัมพันธ์กันมากขึ้น จึงทำให้ประสิทธิภาพของ hard-decision Viterbi ลดลงเมื่อเทียบกับกรณี Poisson-only

## 5. ทำไมยังสรุป coding gain ไม่ได้?

เพราะการทดลองกำหนด Ns ต่อ transmitted pulse เท่ากัน แต่ coded frame มี pulse และระยะเวลามากกว่า จึงใช้พลังงานต่อ payload มากกว่า การหาค่า coding gain อย่างเป็นธรรมต้องเปรียบเทียบที่พลังงานต่อ information bit เท่ากัน และควรรวมอัตรา code กับ overhead ทั้งหมด

## 6. ถ้า Monte Carlo ไม่พบ error แปลว่า BER เป็นศูนย์หรือไม่?

ไม่ควรสรุปว่าเป็นศูนย์ เพราะจำนวนตัวอย่างมีจำกัด ถ้าทดสอบ 800,000 payload bits แล้วไม่พบ error สามารถรายงาน upper bound โดยประมาณจากกฎ 3/N เท่ากับ 3.75 คูณ 10 ยกกำลังลบ 6 ที่ระดับความเชื่อมั่นประมาณ 95 เปอร์เซ็นต์

## 7. Eye diagram ใช้ยืนยันคุณภาพระบบนี้ได้เต็มที่หรือไม่?

ยังไม่ได้ เพราะแบบจำลองหลักเป็น discrete photon counts และสมมติ ideal timing ภาพ eye diagram จึงใช้ประกอบการมองเห็น waveform เท่านั้น การวิเคราะห์ eye opening อย่างสมบูรณ์ต้องมี pulse shape, receiver filter, bandwidth, sampling phase และ timing jitter

