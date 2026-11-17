# MPL ID S18 — Data draft dan alat anotasi VOD

Proyek terpisah dari website draft simulator. Tidak ada berkas website yang diubah. Tahap ini hanya menyiapkan data historis dan alat untuk melengkapinya.

## Status saat ini (hasil audit, bukan klaim)

| | |
|---|---|
| Game di jadwal resmi | **215** (174 Regular Season + 41 Playoffs, 80 match) |
| Game di dataset | **215**; setiap match punya jumlah game sama persis dengan skor |
| Game dikarantina (anomali) | **2** (`m1088-g1`, `m1107-g2`) |
| Game yang boleh dipakai engine | **213** |
| Ban `verified_by_vod` | **1** (GF Game 1, diverifikasi pengguna; per sisi saja) |
| Ban `source_order_unverified` | 214 |
| Pick dengan urutan terverifikasi | **0** |
| Game siap pencocokan pasangan A2+A3 | **0** |

**Artinya:** himpunan hero dan ban per sisi sudah bisa dipakai (dengan label belum terverifikasi); rekomendasi pasangan pick berurutan baru mungkin setelah urutan pick dianotasi dari VOD. Detail: `docs/COVERAGE_AUDIT.md`.

Tentang angka 215 vs 214: 174 + 41 = 215 di jadwal resmi dan di dataset. Angka 214 muncul dari laporan MLBB Lab (173 Regular Season) yang mengecualikan satu record; dataset ini mengidentifikasi `m1088-g1` sebagai satu-satunya game yang melanggar aturan "hero berbeda" yang dinyatakan Lab (inferensi, bukan konfirmasi dari Lab).

## Struktur

```text
mpl-s18-draft-project/
├── data/
│   ├── raw/            mpl_id_s18_raw_extract.json   (potongan pro.json, tak diubah)
│   ├── reference/      heroes.json
│   ├── processed/      games.json · games.csv · draft_events.json · engine_dataset.json · quarantine.json
│   ├── overrides/      manual_overrides.json          (fakta hasil verifikasi manusia)
│   ├── annotations/    ← taruh ekspor alat anotasi di sini
│   └── audit/          coverage_audit.json · validation_report.json · build_log.json
├── scripts/            build_dataset.py · audit_coverage.py · validate_dataset.py · pipeline_lib.py
├── schema/             game.schema.json
├── vod-annotator/      index.html · style.css · app.js · draft-core.js
├── docs/               DATA_SCHEMA.md · ANNOTATION_WORKFLOW.md · COVERAGE_AUDIT.md · examples/
└── tests/              test_pipeline.py · test_core.js · smoke_ui.js
```

## Menjalankan di Windows + VS Code

Prasyarat: Python 3.10+ (centang "Add python.exe to PATH" saat instal), VS Code, Chrome atau Edge. Node.js hanya untuk menjalankan `tests/test_core.js` (opsional).

1. **Buka folder** di VS Code: *File → Open Folder → `mpl-s18-draft-project`*. Buka terminal: *Terminal → New Terminal* (PowerShell).
2. **Lingkungan Python** (opsional tapi disarankan):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Jika PowerShell menolak skrip: `Set-ExecutionPolicy -Scope Process Bypass`, lalu ulangi aktivasi.
3. **Bangun ulang dataset**:
   ```powershell
   python scripts\build_dataset.py                          # dari potongan mentah yang tersimpan
   python scripts\build_dataset.py --pro C:\path\pro.json   # dari pro.json terbaru
   python scripts\audit_coverage.py
   python scripts\validate_dataset.py
   ```
   `validate_dataset.py` mengembalikan kode keluar 1 bila ada error. Tambahkan `--strict` agar peringatan juga dianggap gagal.
4. **Tes**:
   ```powershell
   python -m unittest discover -s tests -v
   node tests\test_core.js
   ```
5. **Alat anotasi VOD**:
   ```powershell
   python -m http.server 8000
   ```
   Buka `http://localhost:8000/vod-annotator/` di Chrome/Edge. Daftar game termuat otomatis. Jika tidak, pakai tombol pilih berkas di panel kiri dan arahkan ke `data\processed\mpl_id_s18_games.json`.
6. **Setelah menganotasi**: klik *Ekspor anotasi*, pindahkan berkas ke `data\annotations\`, lalu jalankan langkah 3 lagi.

Tersedia juga tugas VS Code (*Terminal → Run Task*): Build, Audit, Validate, Run tests, Serve annotator.

## Extractor urutan pick (draf mesin, belum terverifikasi)

`scripts/extract_picks.py` membaca satu VOD (berkas lokal atau aliran, tanpa menyimpan video) dan menghasilkan urutan pick **tak terverifikasi** di `data/extractions/`. Statusnya: kode dan tes ada; **belum pernah dijalankan pada VOD sungguhan**. Penjelasan, hasil uji terukur, dan cara memvalidasi: `docs/VOD_EXTRACTOR.md`. Angka status di tabel atas tidak berubah: pick `verified_by_vod` tetap 0.

## Pembaruan data

Unduh `https://meta.mlbblab.com/pro.json` yang baru dan jalankan `build_dataset.py --pro …`. Fakta di `manual_overrides.json` dan anotasi di `data/annotations/` diterapkan ulang otomatis, dan ID game stabil (`m{match}-g{game}`), jadi pekerjaan verifikasi tidak hilang. Periksa `validation_report.json` dan `coverage_audit.json` setiap kali.

## Batasan yang diketahui

- Pengambilan otomatis Liquipedia diblokir (deteksi bot) dan halaman id-mpl.com tidak terjangkau dari lingkungan penyusun. Semua data berasal dari `pro.json` MLBB Lab; Liquipedia dan id-mpl.com hanya terwakili lewat apa yang dimuat berkas itu. Tidak ada verifikasi langsung ke halaman aslinya.
- Lapisan awal id-mpl.com (126 map) menukar label tim di 52 game; pola ini konsisten dengan artefak parser tetapi belum diverifikasi VOD.
- 6 game (`m1090`, `m1091`) tidak punya pembanding independen untuk himpunan pick.
- Alat anotasi sudah diuji dengan tes logika dan simulasi DOM (jsdom), **belum** di browser sungguhan dengan pemutar YouTube. Uji satu game dulu sebelum mengerjakan banyak.
- Hasil tes anotasi memakai data sintetis yang hanya ada di memori; tidak ada urutan draft fiktif di `data/`.
