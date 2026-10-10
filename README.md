# CT-AV1-FGS-ENGINE

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-1.2.0-green.svg)](https://github.com/Anything2134/av1-fgs-engine)

An advanced, high-performance Film Grain Synthesis (FGS) table generator written in Rust for **SVT-AV1**, **aomenc**, and **Av1an**. Emulates fine **70mm celluloid film grain** and provides an intelligent hybrid digital noise synthesis model.

---

## What's New in v1.2.0

- **`--force-bypass` Parameter**:
  - `1` (Default): Prevents `--force-type` from skewing/damping grain detection and locks `--intensity` to `1.0` for authentic, uncompromised source grain replication.
  - `2`: Bypasses `--force-type` skew during analysis, but allows manual scaling with `--intensity`.
  - `0`: Disabled (legacy behavior).
- **Deep Bit Depth Support (8-bit up to 16-bit)**: Native parsing of uncompressed 10-bit, 12-bit, and 16-bit Y4M streams (`yuv420p10le`, `yuv420p12le`, `yuv420p16le`) with `-strict -1` pipeline integration.
- **HDR & Wide Color Gamut**: Full compatibility with BT.2020, HDR10 (PQ / ST 2084), HLG, and Dolby Vision without highlight luminance skewing.
- **Mutagen Module**: Advanced spatial entropy estimation combining patch-based **Non-Local Means (NLM)** and **Median Absolute Deviation (MAD)** for robust noise separation.
- **Digital Noise & 70mm Hybrid Synthesis**: Capture digital sensor/photon noise or blend it seamlessly with celluloid grain (`--tune-grain-noise-digital` & `--no-noise-bias`).
- **Monotonic Scene Deduplication**: Chronological parsing supporting both Av1an JSON and CSV scene files.

---

## Installation & Compilation

Ensure you have Rust (`cargo`), `ffmpeg`, and optionally `vapoursynth` (`vspipe`) installed.

```bash
# Clone the repository
git clone https://github.com/Anything2134/av1-fgs-engine.git
cd av1-fgs-engine

# Compile the optimized binary
cargo build --release

# Install globally to your system PATH
cargo install --path .
```

### Command-Line Interface (CLI)

```
Usage: ct-av1-fgs-engine [OPTIONS] --input <INPUT>


Options:
  -i, --input <INPUT>                         Path to .vpy script or video container (.mkv, .mp4, .y4m)
  -o, --output <OUTPUT>                       Output .tbl file path [default: 70mm_grain.tbl]
      --force-type <2d|3d|live_action>        Force content classification
      --tune-grain                            Analyze and replicate existing real film grain (1:1 clone)
      --fg-search-full                        Exhaustive scene-by-scene search matching original grain (Requires --tune-grain)
      --forced-fg-search-full                 Dynamic scene-adaptive 70mm grain search for clean/grainless content
      --lookahead <FRAMES>                    Temporal rolling lookahead window in frames
      --scenes <SCENES>                       Path to Av1an scenes file (JSON or CSV)
      --intensity <FLOAT>                     Global grain intensity multiplier [default: 1.0]
      --no-noise-bias <0|1>                   Noise bias filter: 1 = pure grain; 0 = allow digital noise [default: 1]
      --tune-grain-noise-digital <1|2>        Digital tuning (1: Transparent digital, 2: Balanced 70mm hybrid)
      --mutagen <on|off>                      Advanced Mutagen NLM/MAD analysis module [default: on]
      --force-bypass <0|1|2>                  Bypass content bias & lock intensity on tuned grain [default: 1]
  -h, --help                                  Print help
  -V, --version                               Print version

```

#Parameter Reference

Input, Output & Content Classification

  - -i, --input <PATH>: Video file or VapourSynth script. Automatically spawns
    vspipe -c y4m for .vpy or ffmpeg -strict -1 -f yuv4mpegpipe for video
    containers.
  - -o, --output <PATH>: Destination path for the generated filmgrn1 table
    (defaults to 70mm_grain.tbl).
  - --force-type <2d | 3d | live_action>: Overrides spatial heuristic detection.
      - 2d: Suppresses dark noise to protect line art and prevents buzzing in
        flat color fields.
      - 3d: Optimizes midtone dithering to break 8/10-bit banding gradients in
        CGI.
      - live_action: Full sensitometric curve matching large-format
        photochemical stocks.
  - --force-bypass <0 | 1 | 2>: Controls whether content-type bias and intensity
    are bypassed during tuned grain analysis (default: 1).
      - 1: Bypasses --force-type attenuation and locks --intensity to 1.0 for
        faithful, objective source grain reproduction.
      - 2: Bypasses --force-type attenuation, but allows scaling the output
        using --intensity.
      - 0: Disabled (legacy behavior).

Grain Profiling, Digital Noise & Mutagen

  - --tune-grain: Activates deep block-based Fourier variance analysis.
    Replicates source grain (1:1) if detected; falls back to the
    imperceptible 70mm baseline if clean.
  - --no-noise-bias <0 | 1>: Controls the compression/sensor artifact filter
    (default 1).
      - 1: Filters out compression noise (DCT block boundaries) and sensor noise
        to isolate authentic celluloid grain.
      - 0: Disables the filter to allow synthesis of digital noise.
  - --tune-grain-noise-digital <1 | 2>: Only active when --no-noise-bias 0
    (defaults to 2).
      - 1: Transparent pass-through of original digital sensor/photon noise.
      - 2: Balanced hybrid. Blends digital noise amplitude with an organic 70mm
        celluloid curve to restore texture after VapourSynth degraining without
        introducing mosquito artifacts.
  - --mutagen <on | off>: Toggles the Non-Local Means (NLM) and Median Absolute
    Deviation (MAD) decomposition engine (default on).
  - --fg-search-full: Scans the entire video scene by scene. Requires
    --tune-grain.
  - --forced-fg-search-full: Scene-by-scene adaptive 70mm grain search for
    clean/grainless content. Modulates the subtle curve according to each
    scene's mean luminance without creating a static texture.
  - --lookahead <FRAMES>: Moving average buffer on noise variance (\sigma) and
    luminance across adjacent scenes to eliminate cut-to-cut grain flicker.
  - --scenes <PATH>: Ingests Av1an scene cut files (auto-detects JSON or CSV
    format and eliminates duplicate segments).
  - --intensity <FLOAT>: Global multiplier on synthesis amplitude (default:
    1.0).

### License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
See the LICENSE file for details.
