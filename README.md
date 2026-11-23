# CT-AV1-FGS-ENGINE

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-1.3.0-green.svg)](https://github.com/Anything2134/av1-fgs-engine)

An advanced, high-performance AV1 Film Grain Synthesis (FGS) table generator and VapourSynth auto-denoise/auto-degrain companion engine written in Rust for **SVT-AV1**, **aomenc**, and **Av1an**.

---

## What's New in v1.3.0

- **Refined Mutagen Algorithms**:
  - **Isotropic 3×3 Scharr Edge Detector**: Superior rotational accuracy to protect diagonal 2D line art.
  - **2D Planar Linear-Trend Subtraction**: Prevents smooth gradients (skies, skin shading) from triggering false grain readings.
  - **Immerkaer 3×3 Laplacian MAD + Wang/Muijs 8×8 DCT Blockiness Index**: Separates unwanted compression/sensor noise from authentic film grain.
  - **Patch-Similarity NLM Consensus**: Down-weights isolated deterministic micro-textures.
- **ISO-Based Photon Noise Mode (`--iso` & `--photon-noise`)**: Physical Poisson shot-noise curves calibrated across **BT.1886 (SDR)**, **ST 2084 (HDR PQ)**, and **HLG**.
- **VapourSynth Auto-Denoise & Auto-Degrain Plugin (`--emit-vs-plugin` & `--export-vs-profile`)**: Automatically strips compression/mosquito noise while preserving a user-defined percentage of film grain (`--grain-retain`) with closed-loop FGS variance compensation (`--fgs-balance`).

---

## Installation & Compilation

```bash
cargo build --release
cargo install --path .

Command-Line Interface (CLI)

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
      --mutagen <on|off>                      Advanced Mutagen spectral/NLM/MAD analysis module [default: on]
      --force-bypass <0|1|2>                  Bypass content bias & lock intensity on tuned grain [default: 1]
      --iso <50..25600>                       ISO-based photon noise scaling mode (e.g., 200, 800, 1600)
      --photon-noise <1..64>                  Av1an-compatible photon noise level (maps to ISO = value * 100)
      --transfer <auto|sdr|pq|hlg>            Transfer curve for ISO photon noise calculation [default: auto]
      --grain-retain <0..100>                 Percentage of film grain retained after VapourSynth auto-degrain [default: 0]
      --fgs-balance <complement|linear|full>  FGS compensation mode for retained grain [default: complement]
      --export-vs-profile <PATH>              Export per-scene telemetry JSON for VapourSynth plugin
      --emit-vs-plugin <PATH>                 Generate the VapourSynth Python plugin module (e.g., ct_fgs_vs.py)
  -h, --help                                  Print help
  -V, --version                               Print version

VapourSynth Auto-Denoise & Auto-Degrain Workflow

Step 1: Emit the VapourSynth Plugin & Analyze Source

Generate ct_fgs_vs.py, export per-scene telemetry (vs_profile.json), and build a
50% complementary FGS table:

ct-av1-fgs-engine \
    -i "input.mkv" \
    -o "grain_50_comp.tbl" \
    --emit-vs-plugin ct_fgs_vs.py \
    --export-vs-profile vs_profile.json \
    --tune-grain \
    --fg-search-full \
    --scenes escenas.csv \
    --grain-retain 50 \
    --fgs-balance complement

  - --grain-retain 50 instructs VapourSynth to strip 100% of
    compression/mosquito noise while retaining 50% of authentic film grain in
    the encoded pixels.
  - --fgs-balance complement automatically scales the FGS table to
    \sqrt{1 - 0.5^2} \approx 86.6\% amplitude so the encoded grain and
    synthesized FGS sum up to 100% of original grain variance. (Use
    --fgs-balance full to synthesize 100% of detected grain regardless of
    retention).

Step 2: Use inside your .vpy script

import vapoursynth as vs
from vapoursynth import core
import ct_fgs_vs as ctfgs

clip = core.bs.VideoSource(source="input.mkv")

# Automatically removes compression/mosquito noise per scene and keeps 50% of real grain
clip = ctfgs.auto_degrain(
    clip,
    profile="vs_profile.json",
    grain_retain=50,
    protect_lines=True,
)

clip.set_output()

Step 3: Encode with Av1an / SVT-AV1

av1an -i zero.vpy -e svt-av1 -s escenas.csv -v " --crf 24 --fgs-table grain_50_comp.tbl " -o output.mkv

ISO Photon-Noise Mode Examples

# Equivalent to Av1an --photon-noise 8 (ISO 800) with smooth 70mm spatial correlation
ct-av1-fgs-engine -i input.mkv -o iso800.tbl --iso 800

# HDR10 PQ photon noise (ISO 400)
ct-av1-fgs-engine -i movie_hdr.mkv -o hdr_iso400.tbl --photon-noise 4 --transfer pq

License

This project is licensed under the Apache License, Version 2.0. See the LICENSE
file for details.

