import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Scanner LJK Pintar", layout="wide")
st.title("Web App Koreksi LJK")
st.write("Gap Detection + Reference X Mapping + Winner-Takes-All.")

# --- SIDEBAR ---
st.sidebar.header("Pengaturan")
kunci_input = st.sidebar.text_input("Kunci Jawaban (tepat 50 karakter)", value="CCBBDDCDDBCCBBBDCCDCBCDBCCCCCCCBCBBBCCCDCDBCDBBCBC").upper()
poin_benar = st.sidebar.number_input("Poin per Soal Benar", min_value=0.1, value=2.0, step=0.5)

if len(kunci_input) != 50:
    st.sidebar.warning(f"⚠️ Kunci jawaban harus tepat 50 karakter! Saat ini: {len(kunci_input)} karakter. "
                       f"Kelebihan/kekurangan karakter akan menggeser posisi kunci untuk soal-soal berikutnya.")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 1. Unggah Foto LJK")
    foto = st.file_uploader("Pilih foto LJK", type=['jpg', 'jpeg', 'png'])

if foto and len(kunci_input) >= 50:
    if st.button("Mulai Koreksi"):
        with st.spinner("Memproses..."):
            image = Image.open(foto)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_debug = img_cv.copy()
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Normalisasi cahaya
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

            # Adaptive threshold - parameter teruji optimal
            thresh = cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)

            # RETR_LIST agar kotak yang bersarang di border terdeteksi
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Filter kontur yang berbentuk kotak jawaban
            raw_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 12 < w < 70 and 12 < h < 70 and 0.4 < (w / h) < 2.5:
                    raw_boxes.append((x, y, w, h))

            if len(raw_boxes) < 100:
                st.error(f"Hanya {len(raw_boxes)} kotak terdeteksi. Foto kurang jelas.")
            else:
                # Filter outlier ukuran
                med_w = int(np.median([b[2] for b in raw_boxes]))
                med_h = int(np.median([b[3] for b in raw_boxes]))
                size_filtered = [b for b in raw_boxes
                                 if 0.5 * med_w < b[2] < 1.8 * med_w
                                 and 0.5 * med_h < b[3] < 1.8 * med_h]

                # Hapus duplikat (kotak yang tumpang tindih dari RETR_LIST)
                size_filtered.sort(key=lambda b: b[2] * b[3], reverse=True)
                unique = []
                for box in size_filtered:
                    cx1, cy1 = box[0] + box[2] // 2, box[1] + box[3] // 2
                    is_dup = False
                    for kept in unique:
                        cx2 = kept[0] + kept[2] // 2
                        cy2 = kept[1] + kept[3] // 2
                        if (abs(cx1 - cx2) < min(box[2], kept[2]) * 0.5
                                and abs(cy1 - cy2) < min(box[3], kept[3]) * 0.5):
                            is_dup = True
                            break
                    if not is_dup:
                        unique.append(box)

                # STEP 1: Urutkan X, cari 4 gap terbesar → 5 kolom
                unique.sort(key=lambda b: b[0] + b[2] // 2)
                cx_list = [b[0] + b[2] // 2 for b in unique]

                x_gaps = [(cx_list[i] - cx_list[i - 1], i) for i in range(1, len(cx_list))]
                x_gaps.sort(key=lambda g: g[0], reverse=True)
                split_x = sorted([g[1] for g in x_gaps[:4]])

                columns = []
                prev = 0
                for idx in split_x:
                    columns.append(unique[prev:idx])
                    prev = idx
                columns.append(unique[prev:])

                # STEP 2: Proses setiap kolom
                hasil = {}
                debug_info = []

                for col_idx in range(min(5, len(columns))):
                    col_boxes = columns[col_idx]
                    col_boxes.sort(key=lambda b: b[1] + b[3] // 2)

                    # Kelompokkan berdasarkan kedekatan Y → baris
                    rows = []
                    current_row = [col_boxes[0]]
                    for i in range(1, len(col_boxes)):
                        cy_prev = current_row[-1][1] + current_row[-1][3] // 2
                        cy_curr = col_boxes[i][1] + col_boxes[i][3] // 2
                        if abs(cy_curr - cy_prev) < med_h * 0.7:
                            current_row.append(col_boxes[i])
                        else:
                            rows.append(current_row)
                            current_row = [col_boxes[i]]
                    rows.append(current_row)

                    # Ambil hanya baris dengan 3-7 kotak (baris jawaban)
                    valid_rows = [r for r in rows if 3 <= len(r) <= 7]
                    valid_rows.sort(key=lambda r: r[0][1])
                    valid_rows = valid_rows[:10]

                    # Bangun REFERENSI X dari baris yang punya tepat 5 kotak
                    ref_x_all = []
                    for row in valid_rows:
                        rs = sorted(row, key=lambda b: b[0])
                        if len(rs) == 5:
                            ref_x_all.append([b[0] + b[2] // 2 for b in rs])

                    ref_x = np.mean(ref_x_all, axis=0) if ref_x_all else None

                    # STEP 3: Evaluasi setiap soal
                    for row_idx in range(min(10, len(valid_rows))):
                        row = sorted(valid_rows[row_idx], key=lambda b: b[0])
                        no_soal = (row_idx + 1) + (col_idx * 10)

                        intensities = [255, 255, 255, 255, 255]  # default 5 posisi

                        if len(row) >= 5:
                            # 5 kotak lengkap → gunakan posisi langsung
                            for i, (x, y, w, h) in enumerate(row[:5]):
                                mx = int(w * 0.3)
                                my = int(h * 0.3)
                                roi = gray[y + my:y + h - my, x + mx:x + w - mx]
                                intensities[i] = np.mean(roi) if roi.size > 0 else 255
                        elif ref_x is not None and len(row) >= 3:
                            # Kurang dari 5 kotak → mapping ke posisi reference
                            row_cy = int(np.mean([b[1] + b[3] // 2 for b in row]))
                            for bx in row:
                                bcx = bx[0] + bx[2] // 2
                                dists = [abs(bcx - ref_x[j]) for j in range(5)]
                                closest = np.argmin(dists)
                                x, y, w, h = bx
                                mx = int(w * 0.3)
                                my = int(h * 0.3)
                                roi = gray[y + my:y + h - my, x + mx:x + w - mx]
                                val = np.mean(roi) if roi.size > 0 else 255
                                intensities[closest] = val
                            # Posisi yg tidak ditemukan → sampling langsung dari gambar
                            for k in range(5):
                                if intensities[k] == 255:
                                    tx = int(ref_x[k])
                                    hw = med_w // 3
                                    hh = med_h // 3
                                    roi = gray[row_cy - hh:row_cy + hh, tx - hw:tx + hw]
                                    intensities[k] = np.mean(roi) if roi.size > 0 else 255

                        pilihan = "ABCDE"
                        sorted_vals = sorted(intensities)
                        darkest_idx = np.argmin(intensities)

                        # Margin check: jika 2 tergelap terlalu dekat, crop lebih ketat
                        if len(row) >= 5 and sorted_vals[1] - sorted_vals[0] < 15:
                            tight = []
                            for (x, y, w, h) in row[:5]:
                                mx = int(w * 0.4)
                                my = int(h * 0.4)
                                roi = gray[y + my:y + h - my, x + mx:x + w - mx]
                                tight.append(np.mean(roi) if roi.size > 0 else 255)
                            darkest_idx = np.argmin(tight)
                            intensities = tight

                        ans = pilihan[darkest_idx] if darkest_idx < 5 else "?"
                        hasil[no_soal] = ans

                        # Debug
                        vals = ", ".join([f"{pilihan[i]}={intensities[i]:.0f}" for i in range(min(5, len(intensities)))])
                        debug_info.append(f"No.{no_soal:2d}: {vals} → {ans}")

                        # Visualisasi
                        for i, (x, y, w, h) in enumerate(row[:5]):
                            if i < len(pilihan):
                                color = (0, 255, 0) if pilihan[i] == ans else (200, 200, 200)
                                cv2.rectangle(img_debug, (x, y), (x + w, y + h), color, 2)
                        lx = row[0][0] - 25
                        ly = row[0][1] + row[0][3] // 2 + 5
                        cv2.putText(img_debug, f"{no_soal}{ans}", (max(0, lx), ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

                # TAMPILKAN HASIL
                with col2:
                    st.write("### 2. Hasil Analisis")
                    st.image(img_debug, channels="BGR", use_column_width=True)

                    correct = sum(1 for i in range(1, 51) if hasil.get(i) == kunci_input[i - 1])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Benar", correct)
                    m2.metric("Salah", 50 - correct)
                    m3.metric("Nilai", correct * poin_benar)

                    rincian = []
                    for i in range(1, 51):
                        s, k = hasil.get(i, "?"), kunci_input[i - 1]
                        rincian.append({"No": i, "Siswa": s, "Kunci": k,
                                        "Status": "✅" if s == k else "❌"})
                    st.dataframe(rincian, height=400, use_container_width=True)

                    with st.expander("Debug: Intensitas Per Soal"):
                        for d in debug_info:
                            st.text(d)

                    st.info(f"Terdeteksi {len(unique)} kotak unik dari {len(raw_boxes)} kontur. "
                            f"Kolom: {[len(c) for c in columns]}")
else:
    st.info("Masukkan kunci jawaban dan unggah foto LJK.")