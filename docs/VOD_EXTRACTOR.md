# Extractor urutan pick: berbasis posisi slot

Urutan pick diambil dari **posisi slot** di draft bar, bukan dari waktu nama hero pertama kali terlihat.

```
A1 → B1 → B2 → A2 → A3 → B3 → B4 → A4 → A5 → B5
A = tim Blue, slot ke-n dihitung dari tepi luar panelnya (panel kiri: dari kiri)
B = tim Red,  slot ke-n dihitung dari tepi luar panelnya (panel kanan: dari kanan)
```

## 0. Status (baca dulu)

| | Status |
|---|---|
| Kode dibuat | `scripts/extract_picks.py`, `scripts/vod_extract/{geometry,slots,detect,labels,stream,video}.py`, `tests/test_vod_extract.py`, `scripts/eval_cv_approaches.py` |
| Kode diuji | 82 tes lulus (22 lama proyek + 60 baru): frame asli, video sintetis, server HTTP lokal, ffmpeg palsu, logika murni. Dua mutasi sengaja (indeks slot Red diputar; urutan berdasarkan waktu) ditangkap tes. |
| **Ekstraksi berhasil pada VOD sungguhan** | **Belum.** Lingkungan penyusun tidak bisa mengakses YouTube. Yang terbukti pada data asli hanyalah **dua frame statis** dari m1113-g4 (§4). |

## 1. Arsitektur

```
ffmpeg (seek, fps, scale 1280x720, crop strip bar)  ──pipe──►  frame (memori saja)
   └─ stream.py (tanpa berkas; berhenti dini; retry + resolve ulang URL)
frame ──► geometry.py : batas 5 slot tiap panel, DIUKUR PER FRAME dari 5 pelat nama pemain
      ──► labels/detect.py : kotak label nama vertikal + OCR terhadap 96 nama hero
      ──► slots.py : label → slot (aturan "batas kiri terdekat"), per-slot multi-frame,
                     identitas dibekukan saat terkonfirmasi, lalu dipetakan ke A1…B5
```

- **Geometri per frame.** Lebar slot berubah menurut keadaan (terukur di frame asli: 72/72/70/138/138 px pada 05:30, 78/143/114/77/78 px tengah-draft, ±98 px saat semua sama), sehingga pembagian tetap salah. Pelat nama pemain (Karss, Joshuaa, …) selalu ada di tengah slotnya pada semua keadaan; dari pusat kelimanya, batas slot dihitung dengan kuadrat terkecil. Geometri **hanya diterima** bila ada tepat 5 pelat, residu ≤ 7 px, dan lebar slot 45–190 px; jika tidak, frame itu tidak dipakai untuk sisi tersebut.
- **Label → slot.** Label menempel di tepi kiri slotnya (offset −2 ± 0,2 px pada 7 label terukur). Slotnya adalah yang batas kirinya paling dekat dengan posisi label, dengan gerbang ±20 px. Karena batas kiri bertetangga berjarak ≥45 px, galat fit beberapa piksel tidak memindahkan label ke slot tetangga. Label di tengah slot tidak ditetapkan ke slot mana pun.
- **Slot terkunci** = label nama terbaca pada slot itu. Placeholder (logo/foto pemain), preview/hover (slot lebar tanpa label), dan ban (baris ikon di atas strip yang diproses) tidak punya label, sehingga tidak pernah terbaca. Label yang terlihat tetapi tidak bisa dibaca dibedakan dari slot kosong (`slot_label_unreadable` vs `slot_empty`).
- **Konfirmasi multi-frame.** Hero H dikonfirmasi bila terbaca ≥3 kali dalam 6 frame berurutan pada slot itu, dan tidak ada hero lain yang terbaca ≥2 kali dalam jendela yang sama. Identitas lalu **dibekukan**. Perubahan sesudahnya (swap pasca-draft, tata letak ulang) hanya dilaporkan (`slot_changed_after_lock`, level info), tidak mengubah hasil. Perubahan sesaat saat animasi (<2 baca) tidak mengonfirmasi apa pun.
- **Urutan.** Disusun dari `PICK_TURN_ORDER` proyek berdasarkan alamat slot. Waktu tidak dipakai untuk mengurutkan: dua tes membuktikan hasil identik walau slot kedua tiap pasangan terkunci lebih dulu, dan walau pemeriksaan waktu dimatikan.
- **Satu-satunya pemakaian waktu** adalah validasi `slot_order_inconsistent_with_turn_structure`: giliran yang lebih akhir tidak boleh terkunci lebih dari 2 dtk sebelum giliran yang lebih awal selesai. Pemeriksaan ini hanya mengarantina, dan ada untuk menangkap bila asumsi "slot = urutan pick" salah pada layout lain. Bisa dimatikan dengan `--no-turn-check`.
- **Swap dan lineup akhir.** Identitas dibaca saat lock, bukan dari layar akhir. `labels_appear_simultaneously` (semua slot sudah terkunci sejak awal jendela) dan `suspicious_lane_order` (hasil identik dengan urutan lane dataset di kedua sisi, peluang ±1/14.400) mengarantina kasus ketika yang terbaca adalah lineup akhir. Urutan yang berbeda dari urutan lane normal dan bukan masalah.
- **Keluaran.** `data/extractions/<game_id>.json`, skema `vod_extraction/3`: `result` (`extracted`/`quarantined`) dengan `quarantine_reasons`; `events` 10 baris (`position` A1…B5, `side_index`, `hero_id`, `slot.panel/slot_from_outer/slot_from_left`) hanya bila semua pemeriksaan lulus; `verification: machine_extracted_unreviewed`; `provenance` (`order_uses_slot_position: true`, `order_uses_first_seen_time: false`, `order_uses_source_lane_order: false`); `issues` spesifik per slot; diagnostik per slot. Tidak ada timestamp pada event.
- **Penyimpanan.** Jalur streaming tidak menulis video atau frame. Persisten: satu JSON ±5 KB (+ maksimal 12 JPEG kecil dengan `--debug-dir`).

Data proyek (`data/processed`, `data/annotations`, `data/reference`) tidak diubah. Berkas yang **diganti** dibanding iterasi sebelumnya adalah kode buatan sesi ini: `scripts/vod_extract/timeline.py` dihapus (logika urutan berdasarkan `first_seen_s`), `extract_picks.py` dan `tests/test_vod_extract.py` ditulis ulang.

## 2. Alasan kode karantina (yang akan Anda lihat)

| Kode | Arti | Tindakan |
|---|---|---|
| `layout_not_validated` / `geometry_unstable` | geometri slot tidak valid pada panel itu | siaran/layout berbeda: ukur ulang `geometry.NOMINAL_OUTER`; jalankan `--probe-at` |
| `slot_empty` (+posisi) | slot tidak pernah berlabel pada jendela | jendela salah, atau pick belum terjadi |
| `slot_label_unreadable` | ada label tetapi OCR tidak yakin | resolusi terlalu rendah (≥720p wajib) |
| `slot_identity_conflict` | dua hero bergantian terbaca stabil | periksa manual |
| `slot_insufficient_evidence` | terbaca <3 kali | draft berakhir terlalu cepat dalam jendela / fps terlalu rendah |
| `side_mismatch`, `hero_not_in_source`, `source_hero_not_extracted` | hasil berbeda dari catatan sumber | tidak dikoreksi; periksa dataset atau `--left-is` |
| `duplicate_hero` | hero yang sama di dua slot | periksa manual |
| `slot_order_inconsistent_with_turn_structure` | giliran terkunci tidak berurutan | asumsi slot=urutan mungkin salah pada layout ini |
| `labels_appear_simultaneously`, `suspicious_lane_order` | kemungkinan lineup akhir terbaca | jendela waktu salah |
| `low_ocr_confidence` | margin/skor OCR rendah | periksa manual |

## 3. Menjalankan (Windows / VS Code)

Prasyarat: Tesseract OCR (installer UB Mannheim) dan ffmpeg/ffprobe di PATH; `pip install -r requirements.txt`.

Sumber: `--video` (berkas lokal atau URL media http(s), tidak pernah disimpan), `--frames-dir` (folder `<detik>.jpg`), atau `--yt-dlp` (opt-in; hanya meminta URL aliran; **belum teruji** karena tidak ada jaringan; perlu Anda timbang terhadap ketentuan YouTube yang dikutip `ANNOTATION_WORKFLOW.md` §7).

## 4. Yang terbukti pada data asli, dan batasnya

Dua frame asli m1113-g4 (`data/audit/m1113_720p_probe.jpg`, `m1113_panel_reference.jpg`), diuji otomatis:

| Hal | Hasil |
|---|---|
| Batas slot hasil fit vs ukuran independen (gradien/visual) | semua selisih ≤4,5 px (mis. 1080p kiri: fit 87/229/345/424 vs 85/228/343/422) |
| Frame 05:30 | A1 Masha, B1 Freya, B2 Zhuxin, A2 Marcel, A3 Kimmy, B3 Hirara, B4 Atlas terbaca; **A4, A5, B5 kosong** (preview Bruno di A5 tidak terbaca) |
| Frame tengah-draft | hanya A1 Masha, B1 Freya, B2 Zhuxin; preview A2/A3 diabaikan |

Tujuh pick pertama dari frame 05:30 sama dengan urutan yang Anda harapkan. **Batas:** ini dua gambar statis dari satu game; parameter (offset label, gerbang, ambang OCR) diukur dari frame yang sama, jadi bukan akurasi independen. Frame-frame itu tidak mengandung A4, A5, B5, sehingga **tiga pick terakhir (Suyou, Bruno, Claude) belum pernah dibaca dari video asli**. Kelanjutan temporal (konfirmasi multi-frame, animasi lock, swap) hanya teruji pada video sintetis.

Hal lain yang belum terbukti: asumsi bahwa label nama muncul tepat saat lock (didukung dua momen); aturan layout di game/pekan lain; perilaku `-ss` pada URL asli; resolusi URL yt-dlp.

## 5. Prosedur uji: VOD `mpl-id-s18-m1113-g4`

**Langkah 1: periksa geometri pada satu frame (teks saja).** Frame audit 05:30 diambil sekitar detik 330 VOD (coba 320–340):

```powershell
python scripts\extract_picks.py --game-id mpl-id-s18-m1113-g4 --video D:\vod\m1113_g4.mp4 --probe-at 330
```

Hasil yang diharapkan bila VOD dan layoutnya sama dengan frame audit:

```
left:  geometry OK; bounds ≈ [8, 81, 153, 224, 361, 500]
right: geometry OK; bounds ≈ [781, 879, 979, 1077, 1176, 1273]
A1: ... masha     B1: ... freya     B2: ... zhuxin
A2: ... marcel    A3: ... kimmy     B3: ... hirara    B4: ... atlas
A4, A5, B5: no label (empty / preview / placeholder)
```

Bila `geometry INVALID`, atau hero ada di slot yang salah, **berhenti dan kirim teksnya**; jangan lanjut ke langkah 2.

**Langkah 2: ekstraksi penuh.**

```powershell
python scripts\extract_picks.py --game-id mpl-id-s18-m1113-g4 --video D:\vod\m1113_g4.mp4 --debug-dir debug\m1113g4
```

Hasil yang diharapkan: `result: extracted`, event berurutan **Masha → Freya → Zhuxin → Marcel → Kimmy → Hirara → Atlas → Suyou → Bruno → Claude** pada posisi A1, B1, B2, A2, A3, B3, B4, A4, A5, B5. Bila `quarantined`, alasan spesifiknya ada di konsol dan di `issues` (tabel §2).

**Langkah 3: validasi manual.** Bandingkan urutan di JSON dengan VOD pada tiga titik (pick pertama, tengah, terakhir). Konfirmasi di vod-annotator; hanya konfirmasi manusia yang boleh menghasilkan `verified_by_vod`.

**Yang perlu dikirim kembali: teks saja.** Seluruh keluaran konsol langkah 1 dan 2, serta isi `data/extractions/mpl-id-s18-m1113-g4.json` (bagian `result`, `quarantine_reasons`, `issues`, `diagnostics.slots`).

## 6. Belum diputuskan

- **Integrasi dataset:** hasil mesin sengaja tidak masuk ke `data/processed`/`data/annotations`. `apply_annotation` mensyaratkan konfirmasi manusia untuk `verified_by_vod`. Perlu keputusan status apa untuk hasil mesin agar engine boleh memakainya.
- Batch ±210 VOD belum dibuat (menunggu satu VOD asli lolos).
- Catatan lama `mpl_id_s18_m1113_g4_visual_labels.json` ("A5 sudah locked sebelum 05:30") berbeda dengan frame 05:30 (A5 = preview tanpa label); menunggu hasil nyata.
