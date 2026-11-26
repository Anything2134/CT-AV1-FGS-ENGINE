# CT-AV1-FGS-ENGINE

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-2.3.0-green.svg)](https://github.com/Anything2134/CT-AV1-FGS-ENGINE)

A modular, zero-dependency, high-performance **AV1 Film Grain Synthesis (FGS)** profiler, **dual-domain motion-compensated auto-denoiser/degrainer**, and **SVT-AV1 4.3.0+ grain event engine** written in pure Rust.

---

## Purpose & Philosophy

Traditional video encoders struggle with film grain and sensor noise because random high-frequency entropy consumes **40%–70% of bitrate**. While AV1 solves this with Film Grain Synthesis (FGS), existing ecosystems have notable limitations:
1. **Digital Photon Noise (`--photon-noise`)**: Generates spatially uncorrelated pixel static ($Lag = 0$) and chromatic color speckles, often causing distracting *line boiling* on 2D animation.
2. **Encoder-Side Estimators (`--film-grain`)**: Incur heavy encoding speed penalties, lift shadow blacks, and lack artistic/photochemical modeling.
3. **Static Forum Tables**: Ignore scene transitions, lighting changes, and multi-format cuts (e.g., 35mm vs. 70mm IMAX).

**`CT-AV1-FGS-ENGINE`** bridges this gap. It models the organic, high-frequency structure of **70mm celluloid film** (clean luma clustering with zero chroma noise), optionally adapts to digital sensor noise, cleans compression artifacts natively in Rust without external VapourSynth plugins, and outputs strict `filmgrn1` tables, **Av1an `--zones` files**, and `FILM_GRAIN_EVENT` streams for **SVT-AV1**, **aomenc**, **rav1e**, and **Av1an**.

---

## Core Architecture & Modules

The codebase is organized cleanly into 8 focused Rust modules inside `src/`:

| Module | Responsibility |
| :--- | :--- |
| **`src/main.rs`** | CLI orchestration, argument validation, single-pass streaming, and per-scene `HashMap`-indexed MCTF ring buffer. |
| **`src/types.rs`** | Shared enums (`ContentType`, `ColorRange`, `Av1anEncoder`), `YuvFrame`, `MotionField`, and FGS entry types. |
| **`src/y4m.rs`** | Atomic Y4M/VPY parser & writer supporting **8, 9, 10, 12, 14, and 16-bit** (`GRAY`/`mono`, `4:2:0`, `4:2:2`, `4:4:4`), `Full`/`Limited` range, HDR tags, `Fnum:den` framerates, and **4×4 Bayer TPDF dithering**. |
| **`src/motion.rs`** | Pure-Rust hierarchical coarse-to-fine **Diamond Motion Estimation** with **4×4 Walsh-Hadamard SATD**, half-pixel bilinear refinement, and match confidence scoring. |
| **`src/scene.rs`** | Single-pass scene cut detector inspired by `av-scenechange` (directional Intra SATD, motion-compensated Inter SATD, block importance diff, and 32-bin Bhattacharyya luma histogram shift) + universal Av1an JSON/CSV parser. |
| **`src/mutagen.rs`** | **Mutagen v2 Engine**: 3×3 isotropic Scharr edge rejection, 2D planar linear-trend subtraction, Immerkaer 3×3 Laplacian MAD, Wang/Muijs 8×8 DCT blockiness detection, temporal motion-compensated noise verification, NLM patch consensus, **Yule-Walker Cholesky AR solver**, and ISO EOTF curves. |
| **`src/denoiser.rs`** | Zero-dependency Rust **MCTF + Bilateral Auto-Denoiser & Auto-Degrainer**: 3×3 spatial mosquito clamp, 5-frame MCTF with per-pixel 3×3 occlusion guard, **adaptive 5×5 spatial bilateral + 8×8 DCT deblocking** (for duplicated anime 2s/3s frames), Scharr detail protection, and `--grain-retain` blending. |
| **`src/fgs_out.rs`** | Strict `filmgrn1` writer (`100ns`, `frames`, duration-weighted `global`), perceptual scene clustering & **`--emit-av1an-zones`** generator, `FILM_GRAIN_EVENT` JSONL/C-bridge generator, and standalone `ct_fgs_vs.py` module. |

---

## Installation & Compilation

Requires Rust (`cargo`), `ffmpeg`, and optionally `vapoursynth` (`vspipe` for `.vpy` inputs).

```bash
git clone https://github.com/Anything2134/CT-AV1-FGS-ENGINE.git
cd CT-AV1-FGS-ENGINE

cargo build --release
cargo install --path .
```

---

## Production Workflows

### 1. Zero-Dependency Rust Dual-Domain Denoiser + SVT-AV1 4.3.0+ (`100ns` Timed Tables)
Because `SvtAv1EncApp` reads `--fgs-table` immediately on startup before consuming `stdin`, generate the `.tbl` first, then stream clean denoised & dithered Y4M frames directly into the encoder:

```bash
# Step 1: Analyze source and emit 50%-complementary 100ns FGS table
ct-av1-fgs-engine \
    -i "input_10bit.mkv" \
    -o "70mm_comp50.tbl" \
    --tune-grain \
    --fg-search-full \
    --lookahead 8 \
    --scenes scenes.csv \
    --grain-retain 50 \
    --fgs-balance complement \
    --time-unit 100ns

# Step 2: Denoise/degrain in pure Rust (50% grain retained) and pipe directly to SVT-AV1
ct-av1-fgs-engine \
    -i "input_10bit.mkv" \
    -o /dev/null \
    --tune-grain \
    --fg-search-full \
    --scenes scenes.csv \
    --grain-retain 50 \
    --denoise-strength 0.90 \
    --denoise-detail-protect 1.15 \
    --output-y4m - | SvtAv1EncApp -i stdin --fgs-table 70mm_comp50.tbl --preset 4 --crf 24 -b output.ivf
```

### 2. Av1an Parallel Chunked Encoding (Two Safe Options)

#### Option A: Per-Zone Variable Grain with Merged `zones.txt` *(Recommended for movies/OPs with changing grain)*
Instead of dumping hundreds of unlinked `.tbl` files, `--emit-av1an-zones` automatically merges consecutive scenes with similar grain (`--zone-merge-threshold`), emits `t=0` zone tables with absolute paths, and writes a ready-to-use `zones.txt`:

```bash
ct-av1-fgs-engine \
    -i "input.mkv" \
    --tune-grain \
    --fg-search-full \
    --scenes scenes.csv \
    --emit-chunk-tables ./zones_tbl \
    --emit-av1an-zones zones.txt \
    --av1an-encoder svt-av1 \
    --zone-merge-threshold 1.0

# Pass --zones zones.txt to av1an (DO NOT pass --fgs-table inside -v when using --zones)
av1an -i "input.mkv" -e svt-av1 -s scenes.csv --zones zones.txt -v " --crf 24 --preset 4 " -o output.mkv
```

#### Option B: Single Weighted Global Table for Av1an (`--time-unit global`)
If you prefer a single shared `.tbl` file across all Av1an workers without splitting into zones, use `--time-unit global`. The engine computes a duration-weighted average profile across all analyzed scenes (preventing dark intro logos from silencing the whole encode):

```bash
ct-av1-fgs-engine -i "movie.mkv" -o "global_70mm.tbl" --tune-grain --fg-search-full --time-unit global
av1an -i "movie.mkv" -e svt-av1 -v " --crf 24 --fgs-table global_70mm.tbl " -o output.mkv
```

### 3. VapourSynth Integration (`core.resize`, HDR, Plugins & `Trim`/`IVTC`)
You can pass any `.vpy` script directly via `-i zero.vpy`. Important safeguards:
- Ensure the final VapourSynth output node is an integer planar YUV or GRAY format (`YUV420P8/P10/P12/P16`, `YUV422P*`, `YUV444P*`, or `GRAY8/16`). RGB and float32 (`RGBS`/`YUV420PS`) are rejected cleanly with a reminder to convert via `core.resize`.
- **Important**: If your `.vpy` alters total frame count or framerate (e.g., `std.Trim()` or `vivtc.VDecimate()` IVTC), **omit `--scenes`** so the built-in scene detector analyzes the filtered timeline accurately.

### 4. SVT-AV1 4.3.0+ `FILM_GRAIN_EVENT` C Bridge
```bash
ct-av1-fgs-engine \
    -i "input.mkv" \
    --tune-grain \
    --fg-search-full \
    --emit-fgs-events grain_events.jsonl \
    --emit-svt-c-bridge ./svt_bridge
```
Include `./svt_bridge/ct_svt_fgs_bridge.h` in your C/C++ encoder harness and call `ct_attach_film_grain_event(&pic_hdr, &node, frame_idx)` before `svt_av1_enc_send_picture()`.

---

## Patent & License Transparency

All signal processing components—including 4×4 Walsh-Hadamard SATD, Diamond Block Motion Estimation, 3×3 Scharr edge detection, 2D planar linear-trend detrending, Immerkaer Laplacian MAD, Wang/Muijs 8×8 DCT blockiness metrics, Non-Local Means consensus, MCTF Wiener shrinkage, adaptive 5×5 bilateral filtering, 4×4 Bayer TPDF dithering, and classical 1927 Yule-Walker normal equations via Cholesky decomposition—are **clean-room implementations of public-domain mathematical algorithms** containing no patented or GPL-encumbered third-party code.

Licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
