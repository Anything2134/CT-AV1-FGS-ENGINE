# CT-AV1-FGS-ENGINE

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-2.2.0-green.svg)](https://github.com/Anything2134/av1-fgs-engine)

A modular, high-performance **AV1 Film Grain Synthesis (FGS)** profiler, **motion-compensated auto-denoiser/degrainer**, and **SVT-AV1 4.3.0+ grain event engine** written in pure Rust.

---

## Purpose & Philosophy

Traditional video encoders struggle with film grain and sensor noise because random high-frequency entropy consumes **40%–70% of bitrate**. While AV1 solves this with Film Grain Synthesis (FGS), existing ecosystems have notable limitations:
1. **Digital Photon Noise (`--photon-noise`)**: Generates spatially uncorrelated pixel static ($Lag = 0$) and chromatic color speckles, often causing distracting *line boiling* on 2D animation.
2. **Encoder-Side Estimators (`--film-grain`)**: Incur heavy encoding speed penalties, lift shadow blacks, and lack artistic/photochemical modeling.
3. **Static Forum Tables**: Ignore scene transitions, lighting changes, and multi-format cuts (e.g., 35mm vs. 70mm IMAX).

**`CT-AV1-FGS-ENGINE`** bridges this gap. It models the organic, high-frequency structure of **70mm celluloid film** (clean luma clustering with zero chroma noise), optionally adapts to digital sensor noise, cleans compression artifacts natively in Rust with zero external dependencies, and outputs strict `filmgrn1` tables and `FILM_GRAIN_EVENT` streams for **SVT-AV1**, **aomenc**, and **Av1an**.

---

## Core Architecture & Modules

The codebase is organized cleanly into 8 focused Rust modules inside `src/`:

| Module | Responsibility |
| :--- | :--- |
| **`src/main.rs`** | CLI orchestration, argument validation, single-pass streaming, and 5-frame MCTF ring buffer management. |
| **`src/types.rs`** | Shared enums, frame containers (`YuvFrame`), motion structures (`MotionField`), and FGS entry types. |
| **`src/y4m.rs`** | Atomic Y4M/VPY stream parser & writer supporting **8, 10, 12, and 16-bit** (`4:2:0`, `4:2:2`, `4:4:4`), HDR tags, and exact `Fnum:den` framerates. |
| **`src/motion.rs`** | Pure-Rust hierarchical coarse-to-fine **Diamond Motion Estimation** with **4×4 Walsh-Hadamard SATD**, half-pixel bilinear refinement, and match confidence scoring. |
| **`src/scene.rs`** | Single-pass scene cut detector inspired by `av-scenechange` (directional Intra SATD, motion-compensated Inter SATD, block importance diff, and luma histogram shift) + universal Av1an JSON/CSV parser. |
| **`src/mutagen.rs`** | **Mutagen v2 Engine**: 3×3 isotropic Scharr edge rejection, 2D planar linear-trend subtraction, Immerkaer 3×3 Laplacian MAD, Wang/Muijs 8×8 DCT blockiness detection, temporal motion-compensated noise verification, NLM patch consensus, **Yule-Walker Cholesky AR solver**, and ISO EOTF curves. |
| **`src/denoiser.rs`** | Zero-dependency Rust **MCTF Auto-Denoiser & Auto-Degrainer**: 3×3 spatial mosquito spike clamping, 5-frame confidence-weighted temporal filtering, Scharr detail protection, and `--grain-retain` blending. |
| **`src/fgs_out.rs`** | Strict `filmgrn1` writer (`100ns` SVT-AV1 4.3.0+, `frames`, `global`), per-chunk `t=0` exporter, `FILM_GRAIN_EVENT` JSONL/C-bridge generator, and standalone `ct_fgs_vs.py` module emitter. |

---

## Key Capabilities

- **True 70mm Celluloid & Yule-Walker AR Modeling**: Solves causal normal equations directly from motion-compensated noise autocorrelation to produce authentic $Lag = 1, 2, 3$ autoregressive coefficients ($cY$) with clean zero-noise chroma (`sCb 0`, `sCr 0`).
- **Native Zero-Dependency Rust Denoiser (`--output-y4m`)**: Strips compression macroblocking and mosquito noise automatically while retaining a configurable percentage of authentic film grain (`--grain-retain`), paired with closed-loop FGS compensation (`--fgs-balance`).
- **Denoise Controls**: Full automatic by default (`1.0`), with fine-grained control via `--denoise-strength` and `--denoise-detail-protect`.
- **Deep Bit Depth & HDR**: Native support for 8-bit to 16-bit streams (`yuv420p10le`, `yuv420p12le`, `yuv420p16le`), BT.2020, HDR10 (SMPTE ST 2084 PQ), HLG, and Dolby Vision. Automatic `-vf format=...` handling converts non-standard container pixel formats (`yuvj420p`, `nv12`) seamlessly.
- **ISO Photon Noise Mode (`--iso` / `--photon-noise`)**: Physical shot-noise curves calibrated across SDR BT.1886, PQ, and HLG transfer functions.
- **SVT-AV1 4.3.0+ Ready**: Emits timed `100 ns` `filmgrn1` tables, `FILM_GRAIN_EVENT` JSONL streams, and a compile-safe C bridge header (`ct_svt_fgs_bridge.h`) matching local `EbSvtAv1Enc.h` structs.

---

## Installation & Compilation

Requires Rust (`cargo`), `ffmpeg`, and optionally `vapoursynth` (`vspipe` for `.vpy` inputs).

```bash
git clone https://github.com/Anything2134/av1-fgs-engine.git
cd av1-fgs-engine

cargo build --release
cargo install --path .
```

---

## Command-Line Interface (CLI)

```text
Usage: ct-av1-fgs-engine [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>                         Path to .vpy script or video container (.mkv, .mp4, .y4m)
  -o, --output <OUTPUT>                       Output .tbl file path [default: 70mm_grain.tbl]
      --force-type <2d|3d|live_action>        Force content classification
      --tune-grain                            Analyze and replicate existing real film grain (1:1 clone)
      --fg-search-full                        Exhaustive scene-by-scene search matching original grain (Requires --tune-grain)
      --forced-fg-search-full                 Dynamic scene-adaptive 70mm grain search for clean/grainless content
      --lookahead <SCENES>                    Temporal rolling lookahead window across scenes
      --scenes <SCENES>                       Path to Av1an scenes file (JSON or CSV); optional with native detector
      --intensity <FLOAT>                     Global grain intensity multiplier [default: 1.0]
      --no-noise-bias <0|1>                   Noise bias filter: 1 = pure grain; 0 = allow digital noise [default: 1]
      --tune-grain-noise-digital <1|2>        Digital tuning (1: Transparent digital, 2: Balanced 70mm hybrid)
      --mutagen <on|off>                      Advanced Mutagen NLM/MAD/Yule-Walker analysis module [default: on]
      --force-bypass <0|1|2>                  Bypass content bias & lock intensity on tuned grain [default: 1]
      --iso <50..25600>                       ISO-based photon noise scaling mode (e.g., 200, 800, 1600)
      --photon-noise <1..64>                  Av1an-compatible photon noise level (maps to ISO = value * 100)
      --transfer <auto|sdr|pq|hlg>            Transfer curve for ISO photon noise [default: auto]
      --grain-retain <0..100>                 Percentage of authentic film grain retained after auto-degraining [default: 0]
      --fgs-balance <complement|linear|full>  FGS compensation mode for retained grain [default: complement]
      --output-y4m <PATH>                     Native Rust MCTF auto-denoised Y4M output ('-' for stdout pipe)
      --denoise-strength <FLOAT>              MCTF denoiser strength (1.0 = auto, 0.5 = gentle, 0.0 = off) [default: 1.0]
      --denoise-detail-protect <FLOAT>        Scharr edge/texture protection scale (higher preserves more detail) [default: 1.0]
      --no-spatial-clamp                      Disable 3x3 spatial mosquito/impulse clamp in native denoiser
      --protect-lines <true|false>            Enable Scharr line-art protection in native denoiser [default: true]
      --time-unit <100ns|frames|global>       Timestamp unit in filmgrn1 table [default: 100ns]
      --emit-chunk-tables <DIR>               Directory to emit per-chunk t=0 .tbl files for Av1an workers
      --emit-fgs-events <PATH>                Emit per-scene FILM_GRAIN_EVENT JSONL stream for SVT-AV1 4.3.0+
      --emit-svt-c-bridge <DIR>               Generate ct_svt_fgs_bridge.h C header for libSvtAv1Enc 4.3.0+
      --export-vs-profile <PATH>              Export per-scene telemetry JSON for VapourSynth
      --emit-vs-plugin <PATH>                 Generate the standalone VapourSynth Python module (ct_fgs_vs.py)
  -h, --help                                  Print help
  -V, --version                               Print version
```

---

## Production Workflows

### 1. Zero-Dependency Rust MCTF Denoiser + SVT-AV1 4.3.0+ (`100ns` Timed Tables)
Because `SvtAv1EncApp` reads `--fgs-table` immediately on startup before consuming `stdin`, generate the `.tbl` first, then stream clean denoised Y4M frames directly into the encoder:

```bash
# Step 1: Analyze source and emit 50%-complementary 100ns FGS table
ct-av1-fgs-engine \
    -i "input_10bit.mkv" \
    -o "70mm_comp50.tbl" \
    --tune-grain \
    --fg-search-full \
    --lookahead 8 \
    --scenes escenas.csv \
    --grain-retain 50 \
    --fgs-balance complement \
    --time-unit 100ns

# Step 2: Denoise/degrain in pure Rust (50% grain retained) and pipe directly to SVT-AV1
ct-av1-fgs-engine \
    -i "input_10bit.mkv" \
    -o /dev/null \
    --tune-grain \
    --fg-search-full \
    --scenes escenas.csv \
    --grain-retain 50 \
    --denoise-strength 0.85 \
    --denoise-detail-protect 1.2 \
    --output-y4m - | SvtAv1EncApp -i stdin --fgs-table 70mm_comp50.tbl --preset 4 --crf 24 -b output.ivf
```

### 2. Av1an / `av1an-hdr` Chunked Parallel Encoding
When `av1an` splits a video into parallel chunks, each chunk's encoder starts at `t = 0`:
- For a **single shared table** across all chunks, use **`--time-unit global`** (writes a single `0..UINT64_MAX` block).
- For **scene-exact chunked tables**, pass **`--emit-chunk-tables ./chunks_tbl`** to generate `chunk_00000.tbl`, `chunk_00001.tbl`, etc. starting at `t = 0`.
- If using `ct_fgs_vs.py` inside a `.vpy` script under `av1an`/`VSScript`, ensure the working directory is in Python's module path:
  ```python
  import sys, os
  sys.path.insert(0, os.getcwd())
  import ct_fgs_vs as ctfgs
  ```

### 3. Clean Anime / Digital CGI (Dynamic Scene-Adaptive Subtle Grain)
```bash
ct-av1-fgs-engine \
    -i "Anime.1080p.10-Bit.mkv" \
    -o "anime_70mm.tbl" \
    --force-type 2d \
    --forced-fg-search-full \
    --lookahead 8 \
    --time-unit global \
    --intensity 1.3
```

### 4. SVT-AV1 4.3.0+ `FILM_GRAIN_EVENT` C Bridge Integration
```bash
ct-av1-fgs-engine \
    -i "movie.mkv" \
    --tune-grain \
    --fg-search-full \
    --emit-fgs-events grain_events.jsonl \
    --emit-svt-c-bridge ./svt_bridge
```
Include `./svt_bridge/ct_svt_fgs_bridge.h` in your C/C++ encoder harness and call `ct_attach_film_grain_event(&pic_hdr, &node, frame_idx)` before `svt_av1_enc_send_picture()`.

---

## Patent & License Transparency

All signal processing components—including 4×4 Walsh-Hadamard SATD, Diamond Block Motion Estimation, 3×3 Scharr edge detection, 2D planar linear-trend detrending, Immerkaer Laplacian MAD, Wang/Muijs 8×8 DCT blockiness metrics, Non-Local Means consensus, MCTF Wiener shrinkage, and classical 1927 Yule-Walker normal equations via Cholesky decomposition—are **clean-room implementations of public-domain mathematical algorithms** containing no patented or GPL-encumbered third-party code.

Licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
