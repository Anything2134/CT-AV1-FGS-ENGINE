use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum ContentType {
    #[value(name = "2d")]
    TwoD,
    #[value(name = "3d")]
    ThreeD,
    #[value(name = "live_action", alias = "live-action")]
    LiveAction,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum MutagenState {
    #[value(name = "on")]
    On,
    #[value(name = "off")]
    Off,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum TransferCurve {
    #[value(name = "auto")]
    Auto,
    #[value(name = "sdr")]
    Sdr,
    #[value(name = "pq")]
    Pq,
    #[value(name = "hlg")]
    Hlg,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum FgsBalanceMode {
    #[value(name = "complement")]
    Complement,
    #[value(name = "linear")]
    Linear,
    #[value(name = "full")]
    Full,
}

#[derive(Parser, Debug)]
#[command(name = "CT-AV1-FGS-ENGINE")]
#[command(version = "1.3.0")]
#[command(about = "Advanced AV1 Film Grain Synthesis Engine (70mm Celluloid, ISO Photon-Noise & VapourSynth Auto-Degrain)")]
struct Args {
    #[arg(short, long, required_unless_present = "emit_vs_plugin", help = "Path to .vpy script or video container (.mkv, .mp4, .y4m)")]
    input: Option<PathBuf>,

    #[arg(short, long, default_value = "70mm_grain.tbl", help = "Output .tbl file path")]
    output: PathBuf,

    #[arg(long, value_enum, help = "Force content type classification (2d, 3d, live_action)")]
    force_type: Option<ContentType>,

    #[arg(long, help = "Analyze and replicate existing real film grain from source (1:1 clone)")]
    tune_grain: bool,

    #[arg(long, help = "Exhaustive scene-by-scene search matching original source grain (Requires --tune-grain)")]
    fg_search_full: bool,

    #[arg(long, help = "Dynamic scene-adaptive 70mm grain search for clean/grainless content")]
    forced_fg_search_full: bool,

    #[arg(long, help = "Temporal rolling lookahead window in frames (Only valid with full search modes)")]
    lookahead: Option<usize>,

    #[arg(long, help = "Path to Av1an scenes file (supports JSON or CSV format)")]
    scenes: Option<PathBuf>,

    #[arg(long, default_value_t = 1.0, help = "Global grain intensity multiplier")]
    intensity: f32,

    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=1), help = "Noise bias filter: 1 = isolate pure grain; 0 = allow digital/compression noise synthesis")]
    no_noise_bias: u8,

    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=2), help = "Digital noise tuning (1: Transparent digital, 2: Balanced 70mm hybrid). Only valid when --no-noise-bias is 0")]
    tune_grain_noise_digital: Option<u8>,

    #[arg(long, value_enum, default_value = "on", help = "Advanced Mutagen spectral/NLM/MAD analysis module (on/off)")]
    mutagen: MutagenState,

    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=2), help = "Bypass content-type bias & lock intensity on tuned grain (1: Lock both [default], 2: Lock force-type only, allow intensity, 0: Disabled)")]
    force_bypass: u8,

    #[arg(long, value_parser = clap::value_parser!(u32).range(50..=25600), conflicts_with = "photon_noise", help = "ISO-based photon noise scaling mode (e.g., 200, 400, 800, 1600)")]
    iso: Option<u32>,

    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=64), conflicts_with = "iso", help = "Av1an-compatible photon noise level 1..64 (maps to ISO = value * 100)")]
    photon_noise: Option<u8>,

    #[arg(long, value_enum, default_value = "auto", help = "Electro-optical transfer function for ISO photon noise curve (auto, sdr, pq, hlg)")]
    transfer: TransferCurve,

    #[arg(long, default_value_t = 0.0, value_parser = clap::value_parser!(f32), help = "Percentage (0.0..100.0) of authentic film grain retained in video after VapourSynth auto-degraining")]
    grain_retain: f32,

    #[arg(long, value_enum, default_value = "complement", help = "How FGS compensates for retained grain (complement: RMS variance match, linear: 1-r, full: 100% detected)")]
    fgs_balance: FgsBalanceMode,

    #[arg(long, help = "Export per-scene telemetry JSON for the VapourSynth auto-denoise/auto-degrain plugin")]
    export_vs_profile: Option<PathBuf>,

    #[arg(long, help = "Generate/install the VapourSynth Python plugin module (e.g., ct_fgs_vs.py)")]
    emit_vs_plugin: Option<PathBuf>,
}

const BIN_CENTERS: [u8; 8] = [16, 48, 80, 112, 144, 176, 208, 240];

#[derive(Clone, Debug)]
struct SceneSegment {
    start_frame: usize,
    end_frame: usize,
}

#[derive(Clone, Debug)]
struct VsSceneTelemetry {
    start_frame: usize,
    end_frame: usize,
    unwanted_noise_sigma: f32,
    authentic_grain_sigma: f32,
    blockiness: f32,
    mean_luma: f32,
    denoise_thsad: u32,
    degrain_thsad: u32,
}

// -------------------------------------------------------------------------
// Y4M STREAM READER (8, 10, 12, AND 16-BIT / HDR SUPPORT)
// -------------------------------------------------------------------------

struct Y4mStream {
    reader: BufReader<Box<dyn Read + Send>>,
    width: usize,
    height: usize,
    current_frame: usize,
    bit_depth: usize,
    bytes_per_sample: usize,
    luma_size: usize,
    chroma_size: usize,
    detected_transfer: TransferCurve,
}

impl Y4mStream {
    /// Spawns vspipe or ffmpeg to stream uncompressed Y4M frames directly into memory.
    fn from_input(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("File does not exist at path: '{}'.", path.display()));
        }

        let is_vpy = path.extension().and_then(|s| s.to_str()) == Some("vpy");

        let mut child = if is_vpy {
            Command::new("vspipe")
                .arg("-c")
                .arg("y4m")
                .arg(path)
                .arg("-")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to spawn vspipe: {e}"))?
        } else {
            Command::new("ffmpeg")
                .arg("-nostdin")
                .arg("-i")
                .arg(path)
                .arg("-map")
                .arg("0:v:0")
                .arg("-strict")
                .arg("-1")
                .arg("-f")
                .arg("yuv4mpegpipe")
                .arg("-")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to spawn ffmpeg: {e}"))?
        };

        let stdout: Box<dyn Read + Send> = Box::new(child.stdout.take().ok_or("Failed to capture stdout")?);
        let mut reader = BufReader::with_capacity(1024 * 1024, stdout);

        let mut header = String::new();
        if let Err(e) = reader.read_line(&mut header) {
            let mut err_msg = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut err_msg);
            }
            return Err(format!("Error reading Y4M stream: {e}\nDetails:\n{err_msg}"));
        }

        if !header.starts_with("YUV4MPEG2") {
            let mut err_msg = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut err_msg);
            }
            return Err(format!(
                "Invalid Y4M header.\nDetails:\n{}",
                if err_msg.trim().is_empty() { "Process terminated without output." } else { err_msg.trim() }
            ));
        }

        let mut width = 0;
        let mut height = 0;
        let mut bit_depth = 8;
        let mut detected_transfer = TransferCurve::Sdr;

        for token in header.split_whitespace() {
            if let Some(w) = token.strip_prefix('W') {
                width = w.parse().unwrap_or(0);
            } else if let Some(h) = token.strip_prefix('H') {
                height = h.parse().unwrap_or(0);
            } else if let Some(c) = token.strip_prefix('C') {
                if c.contains("p10") || c.contains("10") {
                    bit_depth = 10;
                } else if c.contains("p12") || c.contains("12") {
                    bit_depth = 12;
                } else if c.contains("p16") || c.contains("16") {
                    bit_depth = 16;
                }
            } else if let Some(x) = token.strip_prefix("XCOLORSPACE=") {
                let upper = x.to_uppercase();
                if upper.contains("PQ") || upper.contains("2084") {
                    detected_transfer = TransferCurve::Pq;
                } else if upper.contains("HLG") || upper.contains("B67") {
                    detected_transfer = TransferCurve::Hlg;
                } else if upper.contains("BT2020") && bit_depth >= 10 {
                    detected_transfer = TransferCurve::Pq;
                }
            }
        }

        if width == 0 || height == 0 {
            return Err("Video dimensions not detected in Y4M header".to_string());
        }

        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        let luma_size = width * height * bytes_per_sample;
        let chroma_size = (width / 2) * (height / 2) * 2 * bytes_per_sample;

        Ok(Self {
            reader,
            width,
            height,
            current_frame: 0,
            bit_depth,
            bytes_per_sample,
            luma_size,
            chroma_size,
            detected_transfer,
        })
    }

    fn read_frame_header(&mut self) -> Result<(), ()> {
        let mut line = Vec::new();
        match self.reader.read_until(b'\n', &mut line) {
            Ok(n) if n > 0 && line.starts_with(b"FRAME") => Ok(()),
            _ => Err(()),
        }
    }

    /// Reads the next luma plane and normalizes 8..16-bit samples into 0.0..255.0 floats.
    fn read_next_luma_normalized(&mut self) -> Option<Vec<f32>> {
        if self.read_frame_header().is_err() {
            return None;
        }

        let mut raw_luma = vec![0u8; self.luma_size];
        if self.reader.read_exact(&mut raw_luma).is_err() {
            return None;
        }

        if skip_exact_bytes(&mut self.reader, self.chroma_size).is_err() {
            return None;
        }

        self.current_frame += 1;

        let total_pixels = self.width * self.height;
        let mut normalized = Vec::with_capacity(total_pixels);

        if self.bytes_per_sample == 1 {
            for &p in &raw_luma {
                normalized.push(p as f32);
            }
        } else {
            let max_val = ((1u32 << self.bit_depth) - 1) as f32;
            for chunk in raw_luma.chunks_exact(2) {
                let val = u16::from_le_bytes([chunk[0], chunk[1]]) as f32;
                normalized.push((val / max_val) * 255.0);
            }
        }

        Some(normalized)
    }

    /// Advances the stream forward by discarding frame payloads without buffer reallocations.
    fn skip_frames(&mut self, count: usize) -> bool {
        let frame_data_bytes = self.luma_size + self.chroma_size;
        for _ in 0..count {
            if self.read_frame_header().is_err() {
                return false;
            }
            if skip_exact_bytes(&mut self.reader, frame_data_bytes).is_err() {
                return false;
            }
            self.current_frame += 1;
        }
        true
    }
}

fn skip_exact_bytes<R: BufRead>(reader: &mut R, mut bytes_to_skip: usize) -> std::io::Result<()> {
    while bytes_to_skip > 0 {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF"));
        }
        let consume_len = buffer.len().min(bytes_to_skip);
        reader.consume(consume_len);
        bytes_to_skip -= consume_len;
    }
    Ok(())
}

// -------------------------------------------------------------------------
// MUTAGEN MODULE: SCHARR, PLANAR DETRENDING, IMMERKAER MAD, BLOCKINESS & NLM
// -------------------------------------------------------------------------

mod mutagen {
    use super::*;

    struct BlockSample {
        bin_center: u8,
        sigma_detrend: f32,
        sigma_imm: f32,
        blockiness: f32,
        rho: f32,
    }

    pub struct MutagenEngine {
        pub no_noise_bias: bool,
        pub digital_mode: Option<u8>,
    }

    impl MutagenEngine {
        pub fn new(no_noise_bias: bool, digital_mode: Option<u8>) -> Self {
            Self {
                no_noise_bias,
                digital_mode,
            }
        }

        /// Analyzes a luma frame using Scharr edge rejection, 2D planar detrending,
        /// Immerkaer 3x3 Laplacian MAD, 8x8 DCT blockiness detection, and NLM patch consensus.
        pub fn analyze_frame(
            &self,
            luma: &[f32],
            width: usize,
            height: usize,
            ctype: ContentType,
        ) -> GrainProfile {
            let block_size = 16usize;
            let blocks_x = width / block_size;
            let blocks_y = height / block_size;

            let max_edge = match ctype {
                ContentType::TwoD => 2.8f32,
                ContentType::ThreeD => 4.2f32,
                ContentType::LiveAction => 5.5f32,
            };

            let total_pixels = (width * height).max(1) as f32;
            let frame_mean_luma = luma.iter().sum::<f32>() / total_pixels;

            let block_samples: Vec<BlockSample> = (0..blocks_y)
                .into_par_iter()
                .flat_map(|by| {
                    let mut local = Vec::new();
                    for bx in (0..blocks_x).step_by(2) {
                        let y_start = by * block_size;
                        let x_start = bx * block_size;

                        // 1. Isotropic 3x3 Scharr edge detection + Immerkaer Laplacian high-pass noise
                        let mut scharr_acc = 0.0f32;
                        let mut scharr_max = 0.0f32;
                        let mut imm_abs = Vec::with_capacity(14 * 14);

                        for y in 1..block_size - 1 {
                            for x in 1..block_size - 1 {
                                let row0 = (y_start + y - 1) * width + x_start;
                                let row1 = (y_start + y) * width + x_start;
                                let row2 = (y_start + y + 1) * width + x_start;

                                let p00 = luma[row0 + x - 1];
                                let p01 = luma[row0 + x];
                                let p02 = luma[row0 + x + 1];
                                let p10 = luma[row1 + x - 1];
                                let p11 = luma[row1 + x];
                                let p12 = luma[row1 + x + 1];
                                let p20 = luma[row2 + x - 1];
                                let p21 = luma[row2 + x];
                                let p22 = luma[row2 + x + 1];

                                let gx = -3.0 * p00 + 3.0 * p02 - 10.0 * p10 + 10.0 * p12 - 3.0 * p20 + 3.0 * p22;
                                let gy = -3.0 * p00 - 10.0 * p01 - 3.0 * p02 + 3.0 * p20 + 10.0 * p21 + 3.0 * p22;
                                let mag = (gx.abs() + gy.abs()) / 32.0;
                                scharr_acc += mag;
                                if mag > scharr_max {
                                    scharr_max = mag;
                                }

                                // Immerkaer mask: [1 -2 1; -2 4 -2; 1 -2 1] annihilates linear ramps
                                let imm = (p00 - 2.0 * p01 + p02 - 2.0 * p10 + 4.0 * p11 - 2.0 * p12
                                    + p20 - 2.0 * p21 + p22)
                                    .abs();
                                imm_abs.push(imm);
                            }
                        }

                        let avg_scharr = scharr_acc / (14.0 * 14.0);
                        if avg_scharr > max_edge || scharr_max > max_edge * 4.5 {
                            continue;
                        }

                        imm_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let sigma_imm = (imm_abs[imm_abs.len() / 2] * 1.4826) / 6.0;

                        // 2. 8x8 DCT grid blockiness index (Wang / Muijs boundary ratio)
                        let mut bound_sum = 0.0f32;
                        let mut inner_sum = 0.0f32;
                        let mut inner_count = 0.0f32;

                        for y in 0..block_size {
                            let row = (y_start + y) * width + x_start;
                            bound_sum += (luma[row + 7] - luma[row + 8]).abs();
                            for x in 2..6 {
                                inner_sum += (luma[row + x] - luma[row + x + 1]).abs();
                                inner_count += 1.0;
                            }
                            for x in 10..14 {
                                inner_sum += (luma[row + x] - luma[row + x + 1]).abs();
                                inner_count += 1.0;
                            }
                        }
                        for x in 0..block_size {
                            bound_sum += (luma[(y_start + 7) * width + x_start + x]
                                - luma[(y_start + 8) * width + x_start + x])
                                .abs();
                            for y in 2..6 {
                                inner_sum += (luma[(y_start + y) * width + x_start + x]
                                    - luma[(y_start + y + 1) * width + x_start + x])
                                    .abs();
                                inner_count += 1.0;
                            }
                            for y in 10..14 {
                                inner_sum += (luma[(y_start + y) * width + x_start + x]
                                    - luma[(y_start + y + 1) * width + x_start + x])
                                    .abs();
                                inner_count += 1.0;
                            }
                        }

                        let avg_bound = bound_sum / 32.0;
                        let avg_inner = (inner_sum / inner_count.max(1.0)).max(0.25);
                        let blockiness = avg_bound / avg_inner;

                        // 3. 2D planar linear-trend subtraction to isolate grain from smooth sky gradients
                        let mut sum = 0.0f32;
                        let mut sum_x = 0.0f32;
                        let mut sum_y = 0.0f32;

                        for y in 0..block_size {
                            let dy = (y as f32) - 7.5;
                            let row = (y_start + y) * width + x_start;
                            for x in 0..block_size {
                                let p = luma[row + x];
                                let dx = (x as f32) - 7.5;
                                sum += p;
                                sum_x += dx * p;
                                sum_y += dy * p;
                            }
                        }

                        let mean = sum / 256.0;
                        let slope_x = sum_x / 5440.0;
                        let slope_y = sum_y / 5440.0;

                        let mut detrended = [0.0f32; 256];
                        let mut abs_detrended = Vec::with_capacity(256);
                        let mut sq_diff = 0.0f32;

                        for y in 0..block_size {
                            let dy = (y as f32) - 7.5;
                            let row = (y_start + y) * width + x_start;
                            for x in 0..block_size {
                                let dx = (x as f32) - 7.5;
                                let trend = mean + slope_x * dx + slope_y * dy;
                                let r = luma[row + x] - trend;
                                detrended[y * block_size + x] = r;
                                abs_detrended.push(r.abs());
                                sq_diff += r * r;
                            }
                        }

                        // 2D spatial autocorrelation (horizontal + vertical)
                        let mut autocorr_acc = 0.0f32;
                        for y in 0..block_size {
                            for x in 0..block_size {
                                let r = detrended[y * block_size + x];
                                if x + 1 < block_size {
                                    autocorr_acc += r * detrended[y * block_size + x + 1];
                                }
                                if y + 1 < block_size {
                                    autocorr_acc += r * detrended[(y + 1) * block_size + x];
                                }
                            }
                        }

                        abs_detrended.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let sigma_detrend = abs_detrended[128] * 1.4826;
                        let rho = if sq_diff > 1.0 {
                            (autocorr_acc / (2.0 * sq_diff)).clamp(-0.5, 0.99)
                        } else {
                            0.0
                        };

                        let nearest_bin = BIN_CENTERS
                            .iter()
                            .min_by_key(|&&c| ((c as f32) - mean).abs() as i32)
                            .copied()
                            .unwrap_or(128);

                        local.push(BlockSample {
                            bin_center: nearest_bin,
                            sigma_detrend,
                            sigma_imm,
                            blockiness,
                            rho,
                        });
                    }
                    local
                })
                .collect();

            // Organize samples into 8 luminance bins and track unwanted noise vs authentic grain
            let mut bin_samples: [Vec<&BlockSample>; 8] = Default::default();
            let mut all_blockiness = Vec::new();
            let mut all_unwanted = Vec::new();

            for s in &block_samples {
                all_blockiness.push(s.blockiness);

                // Quantify compression/uncorrelated digital noise component
                let hf_ratio = s.sigma_imm / s.sigma_detrend.max(0.05);
                let is_blocky = s.blockiness > 1.75;
                if is_blocky || hf_ratio > 0.85 {
                    all_unwanted.push(s.sigma_imm);
                } else {
                    // Minor ambient floor noise even in clean grain
                    all_unwanted.push(s.sigma_imm * 0.35);
                }

                // Filter out macroblocked outliers from authentic grain bins when no-noise-bias=1
                if self.no_noise_bias && is_blocky {
                    continue;
                }

                if let Some(pos) = BIN_CENTERS.iter().position(|&c| c == s.bin_center) {
                    bin_samples[pos].push(s);
                }
            }

            let mut rhos = Vec::new();
            let mut profile_sy = Vec::new();
            let mut valid_sigmas = Vec::new();
            let mut hf_ratios = Vec::new();

            for (i, &center) in BIN_CENTERS.iter().enumerate() {
                let samples = &bin_samples[i];
                if samples.is_empty() {
                    profile_sy.push((center, 1));
                    continue;
                }

                let mut detrend_vals: Vec<f32> = samples.iter().map(|s| s.sigma_detrend).collect();
                detrend_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let med_detrend = detrend_vals[detrend_vals.len() / 2];

                // Fast patch-similarity NLM consensus weighting to suppress isolated deterministic micro-textures
                let h_param = (med_detrend * 0.5).max(0.5);
                let mut w_sum = 0.0f32;
                let mut val_sum = 0.0f32;

                for s in samples {
                    let diff = s.sigma_detrend - med_detrend;
                    let w = (-(diff * diff) / (2.0 * h_param * h_param)).exp();
                    w_sum += w;
                    val_sum += w * s.sigma_detrend;
                    if s.rho > 0.0 {
                        rhos.push(s.rho);
                    }
                    hf_ratios.push(s.sigma_imm / s.sigma_detrend.max(0.05));
                }

                let nlm_sigma = if w_sum > 0.0 { val_sum / w_sum } else { med_detrend };

                if nlm_sigma > 0.3 {
                    valid_sigmas.push(nlm_sigma);
                }

                let sy_val = match self.digital_mode {
                    Some(1) => (nlm_sigma * 2.8).round().clamp(1.0, 255.0) as u8,
                    Some(2) => {
                        let celluloid_base = match center {
                            16 | 240 => 1.0,
                            48 | 208 => 2.5,
                            80 | 176 => 4.0,
                            _ => 5.0,
                        };
                        (nlm_sigma * 1.4 + celluloid_base * 0.6).round().clamp(1.0, 255.0) as u8
                    }
                    _ => {
                        if nlm_sigma > 0.5 {
                            (nlm_sigma * 2.3).round().clamp(1.0, 255.0) as u8
                        } else {
                            1
                        }
                    }
                };

                profile_sy.push((center, sy_val));
            }

            let avg_rho = if !rhos.is_empty() {
                rhos.iter().sum::<f32>() / rhos.len() as f32
            } else {
                0.20
            };

            let avg_hf_ratio = if !hf_ratios.is_empty() {
                hf_ratios.iter().sum::<f32>() / hf_ratios.len() as f32
            } else {
                0.60
            };

            // Combine 2D autocorrelation and Immerkaer HF ratio to determine physical particle lag
            let lag: u8 = if let Some(1) = self.digital_mode {
                if avg_hf_ratio > 0.90 { 0 } else { 1 }
            } else if avg_rho > 0.36 {
                1
            } else if avg_rho > 0.20 || avg_hf_ratio > 0.78 {
                2
            } else {
                3
            };

            let authentic_grain_sigma = if !valid_sigmas.is_empty() {
                valid_sigmas.iter().sum::<f32>() / valid_sigmas.len() as f32
            } else {
                0.0
            };

            let unwanted_noise_sigma = if !all_unwanted.is_empty() {
                all_unwanted.iter().sum::<f32>() / all_unwanted.len() as f32
            } else {
                0.0
            };

            let avg_blockiness = if !all_blockiness.is_empty() {
                all_blockiness.iter().sum::<f32>() / all_blockiness.len() as f32
            } else {
                1.0
            };

            GrainProfile {
                lag,
                profile_sy,
                avg_sigma: authentic_grain_sigma,
                unwanted_noise_sigma,
                blockiness: avg_blockiness,
                mean_luma: frame_mean_luma,
            }
        }
    }
}

// -------------------------------------------------------------------------
// FALLBACK ANALYZER (WHEN MUTAGEN IS OFF)
// -------------------------------------------------------------------------

fn fallback_analyze_frame(luma: &[f32], width: usize, height: usize) -> GrainProfile {
    let block_size = 16;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;
    let total_pixels = (width * height).max(1) as f32;
    let frame_mean_luma = luma.iter().sum::<f32>() / total_pixels;

    let mut sigmas = Vec::new();
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let y_start = by * block_size;
            let x_start = bx * block_size;
            let mut sum = 0.0f32;
            for y in 0..block_size {
                for x in 0..block_size {
                    sum += luma[(y_start + y) * width + (x_start + x)];
                }
            }
            let mean = sum / 256.0;
            let mut sq_diff = 0.0f32;
            for y in 0..block_size {
                for x in 0..block_size {
                    let r = luma[(y_start + y) * width + (x_start + x)] - mean;
                    sq_diff += r * r;
                }
            }
            sigmas.push((sq_diff / 256.0).sqrt());
        }
    }

    sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let avg_sigma = if !sigmas.is_empty() { sigmas[sigmas.len() / 2] } else { 0.0 };

    let profile_sy = BIN_CENTERS
        .iter()
        .map(|&c| (c, (avg_sigma * 2.0).round().clamp(1.0, 255.0) as u8))
        .collect();

    GrainProfile {
        lag: 2,
        profile_sy,
        avg_sigma,
        unwanted_noise_sigma: avg_sigma * 0.5,
        blockiness: 1.0,
        mean_luma: frame_mean_luma,
    }
}

#[derive(Clone, Debug)]
struct GrainProfile {
    lag: u8,
    profile_sy: Vec<(u8, u8)>,
    avg_sigma: f32,
    unwanted_noise_sigma: f32,
    blockiness: f32,
    mean_luma: f32,
}

fn classify_frame_auto(luma: &[f32], width: usize, height: usize) -> ContentType {
    let block_size = 16;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    let flat_blocks = (0..blocks_y)
        .into_par_iter()
        .map(|by| {
            let mut count = 0;
            for bx in 0..blocks_x {
                let y_start = by * block_size;
                let x_start = bx * block_size;
                let mut sum = 0.0f32;
                let mut sq_sum = 0.0f32;

                for y in 0..block_size {
                    for x in 0..block_size {
                        let p = luma[(y_start + y) * width + (x_start + x)];
                        sum += p;
                        sq_sum += p * p;
                    }
                }
                let mean = sum / 256.0;
                let var = (sq_sum / 256.0) - (mean * mean);
                if var < 7.0 {
                    count += 1;
                }
            }
            count
        })
        .sum::<usize>();

    let total = (blocks_x * blocks_y).max(1);
    let flat_ratio = (flat_blocks as f32) / (total as f32);

    if flat_ratio > 0.38 {
        ContentType::TwoD
    } else if flat_ratio > 0.18 {
        ContentType::ThreeD
    } else {
        ContentType::LiveAction
    }
}

// -------------------------------------------------------------------------
// ISO-BASED PHOTON NOISE MODELING (BT.1886, ST 2084 PQ, HLG)
// -------------------------------------------------------------------------

/// Converts normalized limited-range luma [0.0..1.0] to relative linear light via EOTF.
fn eotf_to_linear(v: f32, transfer: TransferCurve) -> f32 {
    let vc = v.clamp(0.0, 1.0);
    match transfer {
        TransferCurve::Pq => {
            // SMPTE ST 2084 constants
            let m1 = 0.1593017578125f32;
            let m2 = 78.84375f32;
            let c1 = 0.8359375f32;
            let c2 = 18.8515625f32;
            let c3 = 18.6875f32;
            let vp = vc.powf(1.0 / m2);
            let num = (vp - c1).max(0.0);
            let den = (c2 - c3 * vp).max(1e-6);
            // Normalized to 203 nits diffuse reference white
            (num / den).powf(1.0 / m1) * (10000.0 / 203.0)
        }
        TransferCurve::Hlg => {
            let a = 0.17883277f32;
            let b = 0.28466892f32;
            let c = 0.55991073f32;
            if vc <= 0.5 {
                (vc * vc) / 3.0
            } else {
                (((vc - c) / a).exp() + b) / 12.0
            }
        }
        TransferCurve::Sdr | TransferCurve::Auto => vc.powf(2.4),
    }
}

/// Generates an ISO-calibrated photon-shot scaling curve aligned with Av1an's --photon-noise
/// while supporting clean 70mm spatial correlation or pure sensor noise.
fn generate_iso_photon_profile(
    iso: u32,
    width: usize,
    height: usize,
    transfer: TransferCurve,
    pure_digital: bool,
) -> (u8, Vec<(u8, u8)>) {
    // Standard 14-point luma distribution matching reference photon noise tables
    let luma_pts: [u8; 14] = [16, 19, 38, 58, 77, 96, 116, 135, 155, 174, 193, 213, 232, 235];

    let iso_scale = ((iso as f32) / 200.0).sqrt();
    // Compensate grain visibility across resolutions relative to 1080p
    let res_scale = ((1920.0 * 1080.0) / ((width * height).max(1) as f32))
        .powf(0.15)
        .clamp(0.80, 1.30);

    let mut sy_points = Vec::with_capacity(14);
    for &luma in &luma_pts {
        if luma <= 16 || luma >= 235 {
            sy_points.push((luma, 1));
            continue;
        }

        let norm_v = ((luma as f32) - 16.0) / (235.0 - 16.0);
        let lin_light = eotf_to_linear(norm_v, transfer);

        // Model Poisson shot variance + dark read-noise toe at black boundary
        let toe_boost = if luma <= 24 { 0.85 } else { 0.0 };
        let shot_curve = match transfer {
            TransferCurve::Pq => 1.8 + 1.6 * lin_light.clamp(0.0, 2.5).powf(0.35),
            TransferCurve::Hlg => 1.9 + 1.8 * lin_light.clamp(0.0, 1.5).sqrt(),
            _ => 1.7 + 2.2 * lin_light.powf(0.25),
        };

        let val = ((shot_curve + toe_boost) * iso_scale * res_scale)
            .round()
            .clamp(1.0, 255.0) as u8;
        sy_points.push((luma, val));
    }

    let lag = if pure_digital {
        0 // Uncorrelated sensor shot noise (Lag 0)
    } else if height >= 2160 {
        3 // 70mm high-frequency fine celluloid spatial correlation
    } else {
        2
    };

    (lag, sy_points)
}

fn get_subtle_70mm_profile(ctype: ContentType, scene_mean_luma: f32) -> (u8, Vec<(u8, u8)>) {
    let lag = match ctype {
        ContentType::TwoD => 2,
        ContentType::ThreeD => 2,
        ContentType::LiveAction => 3,
    };

    let luma_weight = (scene_mean_luma / 128.0).clamp(0.6, 1.4);

    let raw_sy: Vec<(u8, u8)> = match ctype {
        ContentType::TwoD => {
            vec![
                (16, 1),
                (32, 1),
                (64, (2.0 * luma_weight).round() as u8),
                (96, (3.0 * luma_weight).round() as u8),
                (128, (4.0 * luma_weight).round() as u8),
                (160, (3.0 * luma_weight).round() as u8),
                (192, 2),
                (224, 1),
                (235, 1),
            ]
        }
        ContentType::ThreeD => {
            vec![
                (16, 1),
                (32, 2),
                (64, (3.0 * luma_weight).round() as u8),
                (96, (4.0 * luma_weight).round() as u8),
                (128, (5.0 * luma_weight).round() as u8),
                (160, (4.0 * luma_weight).round() as u8),
                (192, 3),
                (224, 1),
                (240, 1),
            ]
        }
        ContentType::LiveAction => {
            vec![
                (16, 1),
                (28, 2),
                (48, (3.0 * luma_weight).round() as u8),
                (72, (5.0 * luma_weight).round() as u8),
                (96, (6.0 * luma_weight).round() as u8),
                (120, (7.0 * luma_weight).round() as u8),
                (144, (6.0 * luma_weight).round() as u8),
                (168, (5.0 * luma_weight).round() as u8),
                (192, 4),
                (216, 2),
                (232, 1),
                (240, 1),
            ]
        }
    };

    let cleaned_sy = raw_sy.into_iter().map(|(l, v)| (l, v.clamp(1, 255))).collect();
    (lag, cleaned_sy)
}

/// Computes the FGS scaling factor required to complement retained encoded grain.
fn compute_fgs_retention_scale(grain_retain_pct: f32, balance_mode: FgsBalanceMode) -> f32 {
    let r = (grain_retain_pct / 100.0).clamp(0.0, 1.0);
    match balance_mode {
        FgsBalanceMode::Full => 1.0,
        FgsBalanceMode::Linear => (1.0 - r).max(0.0),
        FgsBalanceMode::Complement => (1.0 - r * r).max(0.0).sqrt(),
    }
}

// -------------------------------------------------------------------------
// STRICT FILMGRN1 WRITER
// -------------------------------------------------------------------------

fn write_fgs_entry(
    out: &mut File,
    start_f: usize,
    end_f: usize,
    lag: u8,
    points_sy: &[(u8, u8)],
    intensity: f32,
) -> std::io::Result<()> {
    let c_y: Vec<i32> = match lag {
        3 => vec![2, 1, -1, -3, 1, 4, 6, 4, 1, -2, 2, 8, 14, 8, 2, -2, 1, 4, 6, 4, 1, -3, -1, 1],
        2 => vec![3, 5, 3, -2, 6, 16, 6, -2, 3, 5, 3, -1],
        1 => vec![8, 20, 8, 4],
        _ => vec![], // Lag 0 requires empty cY coefficients
    };

    let num_pos_luma = 2 * (lag as usize) * ((lag as usize) + 1);
    let num_pos_chroma = num_pos_luma + 1;

    let c_cb_cr = vec!["0"; num_pos_chroma].join(" ");
    let c_y_str = c_y.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");

    let scaled_points: Vec<String> = points_sy
        .iter()
        .map(|&(l, v)| {
            let scaled_v = ((v as f32) * intensity).round().clamp(1.0, 255.0) as u8;
            format!("{l} {scaled_v}")
        })
        .collect();

    let sy_str = format!("{}  {}", scaled_points.len(), scaled_points.join(" "));
    let seed: u16 = (rand_seed(start_f) % 64000 + 1000) as u16;

    writeln!(out, "E {start_f} {end_f} 1 {seed} 1")?;
    writeln!(out, "\tp {lag} 6 0 8 1 1 0 0 0 0 0 0")?;
    writeln!(out, "\tsY {sy_str}")?;
    writeln!(out, "\tsCb 0")?;
    writeln!(out, "\tsCr 0")?;
    if c_y_str.is_empty() {
        writeln!(out, "\tcY")?;
    } else {
        writeln!(out, "\tcY {c_y_str}")?;
    }
    writeln!(out, "\tcCb {c_cb_cr}")?;
    writeln!(out, "\tcCr {c_cb_cr}")?;

    Ok(())
}

fn rand_seed(salt: usize) -> usize {
    let mut x = salt.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn parse_av1an_scenes(path: &Path) -> Result<Vec<SceneSegment>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to open scenes file: {e}"))?;
    let mut raw_segments = Vec::new();

    if content.contains("\"scenes\"") || content.trim_start().starts_with('{') {
        let mut current_start: Option<usize> = None;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.contains("\"start_frame\"") {
                if let Some(val_str) = trimmed.split(':').nth(1) {
                    let clean = val_str.trim().trim_end_matches(',').trim();
                    current_start = clean.parse::<usize>().ok();
                }
            } else if trimmed.contains("\"end_frame\"") {
                if let Some(val_str) = trimmed.split(':').nth(1) {
                    let clean = val_str.trim().trim_end_matches(',').trim();
                    if let (Some(s), Ok(e)) = (current_start, clean.parse::<usize>()) {
                        raw_segments.push(SceneSegment {
                            start_frame: s,
                            end_frame: e,
                        });
                        current_start = None;
                    }
                }
            }
        }
    } else {
        for line in content.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = if l.contains(',') {
                l.split(',').collect()
            } else {
                l.split_whitespace().collect()
            };

            if parts.len() >= 2 {
                if let (Ok(s), Ok(e)) = (parts[0].trim().parse::<usize>(), parts[1].trim().parse::<usize>()) {
                    raw_segments.push(SceneSegment {
                        start_frame: s,
                        end_frame: e,
                    });
                }
            }
        }
    }

    if raw_segments.is_empty() {
        return Err("No valid scene cuts found in scenes file".to_string());
    }

    let mut cleaned_segments: Vec<SceneSegment> = Vec::new();
    let mut highest_end = 0;

    for seg in raw_segments {
        if seg.start_frame >= highest_end && seg.end_frame > seg.start_frame {
            highest_end = seg.end_frame;
            cleaned_segments.push(seg);
        }
    }

    Ok(cleaned_segments)
}

// -------------------------------------------------------------------------
// VAPOURSYNTH PLUGIN MODULE GENERATOR & TELEMETRY EXPORTER
// -------------------------------------------------------------------------

/// Exports per-scene noise/grain telemetry as JSON for consumption by ct_fgs_vs.py.
fn write_vs_telemetry_json(
    path: &Path,
    grain_retain: f32,
    fgs_balance: FgsBalanceMode,
    telemetry: &[VsSceneTelemetry],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let balance_str = match fgs_balance {
        FgsBalanceMode::Complement => "complement",
        FgsBalanceMode::Linear => "linear",
        FgsBalanceMode::Full => "full",
    };

    writeln!(f, "{{")?;
    writeln!(f, "  \"engine\": \"CT-AV1-FGS-ENGINE\",")?;
    writeln!(f, "  \"version\": \"1.3.0\",")?;
    writeln!(f, "  \"grain_retain_pct\": {:.2},", grain_retain)?;
    writeln!(f, "  \"fgs_balance\": \"{}\",", balance_str)?;
    writeln!(f, "  \"scenes\": [")?;

    for (i, t) in telemetry.iter().enumerate() {
        let comma = if i + 1 < telemetry.len() { "," } else { "" };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"start_frame\": {},", t.start_frame)?;
        writeln!(f, "      \"end_frame\": {},", t.end_frame)?;
        writeln!(f, "      \"unwanted_noise_sigma\": {:.3},", t.unwanted_noise_sigma)?;
        writeln!(f, "      \"authentic_grain_sigma\": {:.3},", t.authentic_grain_sigma)?;
        writeln!(f, "      \"blockiness\": {:.3},", t.blockiness)?;
        writeln!(f, "      \"mean_luma\": {:.2},", t.mean_luma)?;
        writeln!(f, "      \"denoise_thsad\": {},", t.denoise_thsad)?;
        writeln!(f, "      \"degrain_thsad\": {}", t.degrain_thsad)?;
        writeln!(f, "    }}{}", comma)?;
    }

    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

/// Emits the self-contained VapourSynth plugin module (ct_fgs_vs.py) implementing
/// automatic unwanted-noise removal and user-controlled film grain retention.
fn write_vs_plugin_module(path: &Path) -> std::io::Result<()> {
    let plugin_code = r#"""
CT-AV1-FGS-ENGINE VapourSynth Auto-Denoise & Auto-Degrain Plugin Module
Licensed under the Apache License, Version 2.0

Provides automated compression/digital noise suppression while retaining a
user-defined percentage of authentic film grain synchronized with FGS tables.
Requires: vapoursynth, mvtools (mv), and removegrain (rgvs).
"""

import json
import bisect
import vapoursynth as vs
from vapoursynth import core


def _load_telemetry(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    starts = [s["start_frame"] for s in scenes]
    return data, scenes, starts


def auto_degrain(
    clip: vs.VideoNode,
    profile: str = None,
    grain_retain: float = None,
    protect_lines: bool = True,
    blksize: int = 8,
    overlap: int = 4,
    thsad_denoise_override: int = None,
    thsad_degrain_override: int = None,
) -> vs.VideoNode:
    """
    Automatically removes compression/mosquito/digital noise while preserving
    a configurable percentage of authentic film grain in the output video.

    Args:
        clip: Input VapourSynth VideoNode.
        profile: Optional path to JSON telemetry exported via --export-vs-profile.
        grain_retain: Percentage (0.0 to 100.0) of authentic film grain to keep.
                      If None, inherits value from profile JSON (defaults to 0.0).
        protect_lines: Builds a Prewitt edge mask to protect sharp 2D line art.
        blksize: MVTools analysis block size (8 recommended for 1080p, 16 for 4K).
        overlap: MVTools block overlap.
        thsad_denoise_override: Manual override for unwanted-noise removal strength.
        thsad_degrain_override: Manual override for full grain removal strength.
    """
    orig_format = clip.format
    if orig_format.bits_per_sample < 16:
        clip16 = core.resize.Point(clip, format=clip.format.replace(bits_per_sample=16))
    else:
        clip16 = clip

    telemetry_scenes = []
    scene_starts = []
    retain_pct = 0.0 if grain_retain is None else float(grain_retain)

    if profile is not None:
        data, telemetry_scenes, scene_starts = _load_telemetry(profile)
        if grain_retain is None:
            retain_pct = float(data.get("grain_retain_pct", 0.0))

    retain_pct = max(0.0, min(100.0, retain_pct))
    retain_weight = retain_pct / 100.0

    # Stage 1: Non-destructive spatial mosquito and impulse spike cleanup (RemoveGrain Mode 1)
    clean_spatial = core.rgvs.RemoveGrain(clip16, mode=1)

    # Stage 2: Optimized 3x3 Gaussian prefilter for stable motion vector estimation
    pref = core.rgvs.RemoveGrain(clean_spatial, mode=11)

    super_render = core.mv.Super(clean_spatial, pel=2, sharp=2)
    super_search = core.mv.Super(pref, pel=2, sharp=2)

    bv2 = core.mv.Analyse(super_search, isb=True, delta=2, overlap=overlap, blksize=blksize, truemotion=False)
    bv1 = core.mv.Analyse(super_search, isb=True, delta=1, overlap=overlap, blksize=blksize, truemotion=False)
    fv1 = core.mv.Analyse(super_search, isb=False, delta=1, overlap=overlap, blksize=blksize, truemotion=False)
    fv2 = core.mv.Analyse(super_search, isb=False, delta=2, overlap=overlap, blksize=blksize, truemotion=False)

    # Edge protection mask for sharp line art
    edge_mask = None
    if protect_lines:
        luma_pref = core.std.ShufflePlanes(pref, planes=0, colorfamily=vs.GRAY)
        edges = core.std.Prewitt(luma_pref)
        edge_mask = core.std.Expr(edges, expr="x 2200 > 65535 0 ?").rgvs.RemoveGrain(mode=11)

    if telemetry_scenes:
        # Per-scene dynamic evaluation driven by CT-AV1-FGS-ENGINE Mutagen telemetry
        # Pre-generate calibrated levels or evaluate smoothly per scene bracket
        unique_pairs = sorted(set(
            (
                thsad_denoise_override if thsad_denoise_override is not None else int(s["denoise_thsad"]),
                thsad_degrain_override if thsad_degrain_override is not None else int(s["degrain_thsad"]),
            )
            for s in telemetry_scenes
        ))

        # Quantize thsad to discrete buckets (step=20) to keep graph instantiation lightweight
        def quantize_thsad(val: int) -> int:
            return max(40, min(400, ((val + 10) // 20) * 20))

        clip_cache = {}

        def get_degrained_pair(d_noise: int, d_grain: int):
            key = (d_noise, d_grain)
            if key not in clip_cache:
                if retain_weight >= 0.999:
                    c_out = core.mv.Degrain2(
                        clip=clean_spatial, super=super_render,
                        mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                        thsad=d_noise, thscd1=200, thscd2=90
                    )
                elif retain_weight <= 0.001:
                    c_out = core.mv.Degrain2(
                        clip=clean_spatial, super=super_render,
                        mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                        thsad=d_grain, thscd1=200, thscd2=90
                    )
                else:
                    c_noise_free = core.mv.Degrain2(
                        clip=clean_spatial, super=super_render,
                        mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                        thsad=d_noise, thscd1=200, thscd2=90
                    )
                    c_fully_clean = core.mv.Degrain2(
                        clip=clean_spatial, super=super_render,
                        mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                        thsad=d_grain, thscd1=200, thscd2=90
                    )
                    c_out = core.std.Merge(c_fully_clean, c_noise_free, weight=retain_weight)

                if edge_mask is not None:
                    c_out = core.std.MaskedMerge(c_out, clean_spatial, edge_mask)
                clip_cache[key] = c_out
            return clip_cache[key]

        for dn, dg in unique_pairs:
            get_degrained_pair(quantize_thsad(dn), quantize_thsad(dg))

        def selector(n):
            idx = bisect.bisect_right(scene_starts, n) - 1
            if idx < 0:
                idx = 0
            s = telemetry_scenes[idx]
            dn = quantize_thsad(thsad_denoise_override if thsad_denoise_override is not None else int(s["denoise_thsad"]))
            dg = quantize_thsad(thsad_degrain_override if thsad_degrain_override is not None else int(s["degrain_thsad"]))
            return get_degrained_pair(dn, dg)

        processed = core.std.FrameEval(clean_spatial, selector)
    else:
        # Standalone automatic fallback using local spatial noise estimation if no JSON profile is supplied
        d_noise = thsad_denoise_override if thsad_denoise_override is not None else 95
        d_grain = thsad_degrain_override if thsad_degrain_override is not None else 200

        if retain_weight >= 0.999:
            processed = core.mv.Degrain2(
                clip=clean_spatial, super=super_render,
                mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                thsad=d_noise, thscd1=200, thscd2=90
            )
        elif retain_weight <= 0.001:
            processed = core.mv.Degrain2(
                clip=clean_spatial, super=super_render,
                mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                thsad=d_grain, thscd1=200, thscd2=90
            )
        else:
            c_noise_free = core.mv.Degrain2(
                clip=clean_spatial, super=super_render,
                mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                thsad=d_noise, thscd1=200, thscd2=90
            )
            c_fully_clean = core.mv.Degrain2(
                clip=clean_spatial, super=super_render,
                mvbw=bv1, mvbw2=bv2, mvfw=fv1, mvfw2=fv2,
                thsad=d_grain, thscd1=200, thscd2=90
            )
            processed = core.std.Merge(c_fully_clean, c_noise_free, weight=retain_weight)

        if edge_mask is not None:
            processed = core.std.MaskedMerge(processed, clean_spatial, edge_mask)

    if orig_format.bits_per_sample < 16:
        return core.resize.Point(processed, format=orig_format, dither_type="error_diffusion")
    return processed
"#;

    let mut f = File::create(path)?;
    f.write_all(plugin_code.as_bytes())?;
    Ok(())
}

// -------------------------------------------------------------------------
// ENTRYPOINT
// -------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Emit VapourSynth plugin module if requested
    if let Some(ref plugin_path) = args.emit_vs_plugin {
        write_vs_plugin_module(plugin_path)?;
        println!("[✓] VapourSynth plugin module written to: {}", plugin_path.display());
        if args.input.is_none() {
            return Ok(());
        }
    }

    let input_path = args.input.as_ref().ok_or("Input path (-i / --input) is required for analysis")?;

    // Validate mutually exclusive parameters and grain retention range
    if args.grain_retain < 0.0 || args.grain_retain > 100.0 {
        eprintln!("[-] Error: --grain-retain must be between 0.0 and 100.0.");
        std::process::exit(1);
    }

    if args.no_noise_bias == 1 && args.tune_grain_noise_digital.is_some() {
        eprintln!("[-] Error: --tune-grain-noise-digital can ONLY be used when --no-noise-bias is 0.");
        std::process::exit(1);
    }

    let digital_noise_mode = if args.no_noise_bias == 0 {
        Some(args.tune_grain_noise_digital.unwrap_or(2))
    } else {
        None
    };

    if args.lookahead.is_some() && !args.fg_search_full && !args.forced_fg_search_full {
        eprintln!("[-] Error: --lookahead requires either --fg-search-full or --forced-fg-search-full.");
        std::process::exit(1);
    }

    if args.fg_search_full && !args.tune_grain {
        eprintln!("[-] Error: --fg-search-full requires --tune-grain.");
        std::process::exit(1);
    }

    let target_iso: Option<u32> = args.iso.or_else(|| args.photon_noise.map(|n| (n as u32) * 100));

    println!("[*] Initializing CT-AV1-FGS-ENGINE (Rust Engine v1.3.0 | Apache-2.0)...");
    let mut stream = Y4mStream::from_input(input_path)?;

    let effective_transfer = match args.transfer {
        TransferCurve::Auto => stream.detected_transfer,
        other => other,
    };

    println!(
        "    - Stream: {}x{} | {}-bit depth ({} bytes/sample) | Transfer: {:?}",
        stream.width, stream.height, stream.bit_depth, stream.bytes_per_sample, effective_transfer
    );

    if let Some(iso_val) = target_iso {
        println!("    - ISO Photon-Noise Scaling Active: ISO {}", iso_val);
    }

    let retention_scale = compute_fgs_retention_scale(args.grain_retain, args.fgs_balance);
    if args.grain_retain > 0.0 || args.export_vs_profile.is_some() {
        println!(
            "    - VapourSynth Grain Retention: {:.1}% | FGS Balance Mode: {:?} (FGS Factor: {:.3})",
            args.grain_retain, args.fgs_balance, retention_scale
        );
    }

    let mutagen_engine = mutagen::MutagenEngine::new(args.no_noise_bias == 1, digital_noise_mode);
    if args.mutagen == MutagenState::Off {
        eprintln!("[!] WARNING: Mutagen module is OFF. Using fallback analysis (higher false-positive risk).");
    }

    let mut out_file = File::create(&args.output)?;
    writeln!(out_file, "filmgrn1")?;

    let first_frame = stream.read_next_luma_normalized().ok_or("Failed to read initial luma frame")?;
    let content_type = if let Some(forced) = args.force_type {
        println!("[*] Content type forced by user: {:?}", forced);
        forced
    } else {
        let auto = classify_frame_auto(&first_frame, stream.width, stream.height);
        println!("[*] Content type auto-detected: {:?}", auto);
        auto
    };

    let is_tuning_active = args.tune_grain || digital_noise_mode.is_some();

    let (analysis_ctype, effective_tune_intensity) = if is_tuning_active && args.force_bypass >= 1 {
        let eff_int = if args.force_bypass == 1 {
            if (args.intensity - 1.0).abs() > 0.01 {
                println!("    [!] --force-bypass 1 active: Overriding --intensity {:.2} -> 1.0 for pure 1:1 reproduction.", args.intensity);
            } else {
                println!("    [*] --force-bypass 1 active: Bypassing content-type bias and locking --intensity to 1.0.");
            }
            1.0f32
        } else {
            println!("    [*] --force-bypass 2 active: Bypassing content-type bias; applying --intensity {:.2}.", args.intensity);
            args.intensity
        };
        (ContentType::LiveAction, eff_int)
    } else {
        (content_type, args.intensity)
    };

    let lookahead_window = args.lookahead.unwrap_or(0);
    let mut vs_telemetry: Vec<VsSceneTelemetry> = Vec::new();

    if args.fg_search_full || args.forced_fg_search_full {
        let mode_desc = if args.fg_search_full {
            "Full source grain search & replication (--fg-search-full)"
        } else {
            "Scene-adaptive subtle 70mm grain search (--forced-fg-search-full)"
        };
        println!("[*] Active temporal mode: {}", mode_desc);

        let segments = if let Some(ref sc_path) = args.scenes {
            let segs = parse_av1an_scenes(sc_path)?;
            println!("    [+] Loading exact scene cuts from: {}", sc_path.display());
            println!("    [+] Clean chronological scene segments to process: {}", segs.len());
            segs
        } else {
            println!("    [!] No scenes file provided. Fallback to 120-frame intervals...");
            (0..50000)
                .step_by(120)
                .map(|start| SceneSegment {
                    start_frame: start,
                    end_frame: start + 120,
                })
                .collect()
        };

        let mut previous_profiles: Vec<GrainProfile> = Vec::new();

        for (idx, seg) in segments.iter().enumerate() {
            let mid_point = (seg.start_frame + seg.end_frame) / 2;

            if stream.current_frame < mid_point {
                let to_skip = mid_point - stream.current_frame;
                if !stream.skip_frames(to_skip) {
                    break;
                }
            }

            if let Some(frame) = stream.read_next_luma_normalized() {
                let current_prof = if args.mutagen == MutagenState::On {
                    mutagen_engine.analyze_frame(&frame, stream.width, stream.height, analysis_ctype)
                } else {
                    fallback_analyze_frame(&frame, stream.width, stream.height)
                };

                let smoothed_profile = if lookahead_window > 0 {
                    previous_profiles.push(current_prof.clone());
                    if previous_profiles.len() > lookahead_window {
                        previous_profiles.remove(0);
                    }
                    let len_f = previous_profiles.len() as f32;
                    GrainProfile {
                        lag: current_prof.lag,
                        profile_sy: current_prof.profile_sy.clone(),
                        avg_sigma: previous_profiles.iter().map(|p| p.avg_sigma).sum::<f32>() / len_f,
                        unwanted_noise_sigma: previous_profiles.iter().map(|p| p.unwanted_noise_sigma).sum::<f32>() / len_f,
                        blockiness: previous_profiles.iter().map(|p| p.blockiness).sum::<f32>() / len_f,
                        mean_luma: previous_profiles.iter().map(|p| p.mean_luma).sum::<f32>() / len_f,
                    }
                } else {
                    current_prof
                };

                // Record VapourSynth auto-denoise / auto-degrain telemetry for this scene
                let denoise_thsad = (smoothed_profile.unwanted_noise_sigma * 75.0
                    + (smoothed_profile.blockiness - 1.0).max(0.0) * 55.0)
                    .round()
                    .clamp(40.0, 220.0) as u32;
                let degrain_thsad = ((smoothed_profile.unwanted_noise_sigma + smoothed_profile.avg_sigma) * 75.0)
                    .round()
                    .clamp(denoise_thsad as f32, 360.0) as u32;

                vs_telemetry.push(VsSceneTelemetry {
                    start_frame: seg.start_frame,
                    end_frame: seg.end_frame,
                    unwanted_noise_sigma: smoothed_profile.unwanted_noise_sigma,
                    authentic_grain_sigma: smoothed_profile.avg_sigma,
                    blockiness: smoothed_profile.blockiness,
                    mean_luma: smoothed_profile.mean_luma,
                    denoise_thsad,
                    degrain_thsad,
                });

                // Determine output scaling curve for this segment
                if args.fg_search_full {
                    if smoothed_profile.avg_sigma >= 1.15 || digital_noise_mode.is_some() {
                        let iso_boost = if args.force_bypass == 1 {
                            1.0
                        } else {
                            target_iso.map(|i| (i as f32 / 100.0).sqrt()).unwrap_or(1.0)
                        };
                        let final_scale = effective_tune_intensity * retention_scale * iso_boost;
                        write_fgs_entry(&mut out_file, seg.start_frame, seg.end_frame, smoothed_profile.lag, &smoothed_profile.profile_sy, final_scale)?;
                    } else if let Some(iso_val) = target_iso {
                        let pure_digital = digital_noise_mode == Some(1);
                        let (lag, iso_sy) = generate_iso_photon_profile(iso_val, stream.width, stream.height, effective_transfer, pure_digital);
                        write_fgs_entry(&mut out_file, seg.start_frame, seg.end_frame, lag, &iso_sy, args.intensity * retention_scale)?;
                    } else {
                        let (lag, subtle_sy) = get_subtle_70mm_profile(content_type, smoothed_profile.mean_luma);
                        write_fgs_entry(&mut out_file, seg.start_frame, seg.end_frame, lag, &subtle_sy, args.intensity * retention_scale)?;
                    }
                } else if let Some(iso_val) = target_iso {
                    let pure_digital = digital_noise_mode == Some(1);
                    let (lag, iso_sy) = generate_iso_photon_profile(iso_val, stream.width, stream.height, effective_transfer, pure_digital);
                    write_fgs_entry(&mut out_file, seg.start_frame, seg.end_frame, lag, &iso_sy, args.intensity * retention_scale)?;
                } else {
                    let (lag, dynamic_sy) = get_subtle_70mm_profile(content_type, smoothed_profile.mean_luma);
                    write_fgs_entry(&mut out_file, seg.start_frame, seg.end_frame, lag, &dynamic_sy, args.intensity * retention_scale)?;
                }

                if idx % 10 == 0 || idx == segments.len() - 1 {
                    println!(
                        "    - Segment {}/{} [{}..{}]: Luma={:.1} | Grain={:.2} | Noise={:.2} | Block={:.2}",
                        idx + 1,
                        segments.len(),
                        seg.start_frame,
                        seg.end_frame,
                        smoothed_profile.mean_luma,
                        smoothed_profile.avg_sigma,
                        smoothed_profile.unwanted_noise_sigma,
                        smoothed_profile.blockiness
                    );
                }
            } else {
                break;
            }
        }
    } else {
        println!("[*] Generating static global grain table...");
        let prof = if args.mutagen == MutagenState::On {
            mutagen_engine.analyze_frame(&first_frame, stream.width, stream.height, analysis_ctype)
        } else {
            fallback_analyze_frame(&first_frame, stream.width, stream.height)
        };

        let denoise_thsad = (prof.unwanted_noise_sigma * 75.0 + (prof.blockiness - 1.0).max(0.0) * 55.0)
            .round()
            .clamp(40.0, 220.0) as u32;
        let degrain_thsad = ((prof.unwanted_noise_sigma + prof.avg_sigma) * 75.0)
            .round()
            .clamp(denoise_thsad as f32, 360.0) as u32;

        vs_telemetry.push(VsSceneTelemetry {
            start_frame: 0,
            end_frame: 999999999,
            unwanted_noise_sigma: prof.unwanted_noise_sigma,
            authentic_grain_sigma: prof.avg_sigma,
            blockiness: prof.blockiness,
            mean_luma: prof.mean_luma,
            denoise_thsad,
            degrain_thsad,
        });

        if (args.tune_grain && prof.avg_sigma >= 1.15) || digital_noise_mode.is_some() {
            println!("    [+] Grain/noise detected (Grain Sigma: {:.2}, Unwanted Noise: {:.2}). Synthesizing...", prof.avg_sigma, prof.unwanted_noise_sigma);
            let iso_boost = if args.force_bypass == 1 {
                1.0
            } else {
                target_iso.map(|i| (i as f32 / 100.0).sqrt()).unwrap_or(1.0)
            };
            let final_scale = effective_tune_intensity * retention_scale * iso_boost;
            write_fgs_entry(&mut out_file, 0, 18446744073709551615, prof.lag, &prof.profile_sy, final_scale)?;
        } else if let Some(iso_val) = target_iso {
            println!("    [*] Generating ISO {} photon-shot profile...", iso_val);
            let pure_digital = digital_noise_mode == Some(1);
            let (lag, iso_sy) = generate_iso_photon_profile(iso_val, stream.width, stream.height, effective_transfer, pure_digital);
            write_fgs_entry(&mut out_file, 0, 18446744073709551615, lag, &iso_sy, args.intensity * retention_scale)?;
        } else {
            println!("    [*] Applying subtle 70mm baseline default profile...");
            let (lag, subtle_sy) = get_subtle_70mm_profile(content_type, prof.mean_luma);
            write_fgs_entry(&mut out_file, 0, 18446744073709551615, lag, &subtle_sy, args.intensity * retention_scale)?;
        }
    }

    if let Some(ref vs_prof_path) = args.export_vs_profile {
        write_vs_telemetry_json(vs_prof_path, args.grain_retain, args.fgs_balance, &vs_telemetry)?;
        println!("[✓] VapourSynth telemetry profile exported to: {}", vs_prof_path.display());
    }

    println!("[✓] FGS table successfully written to: {}", args.output.display());
    Ok(())
}
