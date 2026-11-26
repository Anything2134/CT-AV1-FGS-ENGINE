mod denoiser;
mod fgs_out;
mod motion;
mod mutagen;
mod scene;
mod types;
mod y4m;

use clap::Parser;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use denoiser::denoise_and_degrain_frame;
use fgs_out::{
    compute_fgs_retention_scale, rand_seed, write_av1an_chunks_and_zones, write_fgs_events_jsonl,
    write_filmgrn1_table, write_svt_c_bridge, write_vs_plugin_module, write_vs_telemetry_json,
};
use motion::estimate_motion_field;
use mutagen::{
    classify_frame_auto, fallback_analyze_frame, generate_iso_photon_profile,
    get_subtle_70mm_profile, MutagenEngine,
};
use scene::{parse_av1an_scenes, SceneDetector};
use types::{
    Av1anEncoder, ColorRange, ContentType, FgsBalanceMode, FgsTableEntry, GrainProfile,
    MutagenState, TimeUnitMode, TransferCurve, VsSceneTelemetry, YuvFrame,
};
use y4m::{Y4mStream, Y4mWriter};

#[derive(Parser, Debug)]
#[command(name = "CT-AV1-FGS-ENGINE")]
#[command(version = "2.3.0")]
#[command(about = "Modular AV1 Film Grain Synthesis, Motion-Compensated Denoiser, and SVT-AV1 Event Engine")]
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

    #[arg(long, help = "Temporal rolling lookahead window in scenes (Only valid with full search modes)")]
    lookahead: Option<usize>,

    #[arg(long, help = "Path to Av1an scenes file (JSON or CSV); omit when VapourSynth alters frame count via Trim/IVTC")]
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

    #[arg(long, value_enum, default_value = "auto", help = "Transfer curve for ISO photon noise (auto, sdr, pq, hlg)")]
    transfer: TransferCurve,

    #[arg(long, value_enum, default_value = "auto", help = "Input color range (auto, limited, full)")]
    color_range: ColorRange,

    #[arg(long, default_value_t = 0.0, value_parser = clap::value_parser!(f32), help = "Percentage (0.0..100.0) of film grain retained after auto-degraining")]
    grain_retain: f32,

    #[arg(long, value_enum, default_value = "complement", help = "FGS balance mode (complement: RMS variance match, linear: 1-r, full: 100% detected)")]
    fgs_balance: FgsBalanceMode,

    #[arg(long, help = "Native zero-dependency Rust auto-denoised/degrained Y4M output path (use '-' for stdout pipe)")]
    output_y4m: Option<PathBuf>,

    #[arg(long, default_value_t = 1.0, help = "MCTF + Bilateral denoiser strength multiplier (1.0 = full automatic, 0.5 = gentle, 0.0 = off)")]
    denoise_strength: f32,

    #[arg(long, default_value_t = 1.0, help = "Scharr edge & fine texture protection scale in native denoiser (higher preserves more detail)")]
    denoise_detail_protect: f32,

    #[arg(long, help = "Disable 3x3 spatial mosquito/impulse clamp in native denoiser")]
    no_spatial_clamp: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help = "Enable Scharr line-art edge protection in native Rust denoiser")]
    protect_lines: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help = "Enable 4x4 Bayer dithering when encoding denoised Y4M output")]
    dither: bool,

    #[arg(long, value_enum, default_value = "100ns", help = "Timestamp unit in filmgrn1 table (100ns: SVT-AV1 4.3.0+ standard, frames, global)")]
    time_unit: TimeUnitMode,

    #[arg(long, help = "Directory to emit merged t=0 zone/chunk .tbl files for Av1an")]
    emit_chunk_tables: Option<PathBuf>,

    #[arg(long, help = "Generate ready-to-use Av1an --zones file pointing to merged per-zone .tbl files")]
    emit_av1an_zones: Option<PathBuf>,

    #[arg(long, value_enum, default_value = "svt-av1", help = "Target encoder syntax inside Av1an zones file (svt-av1, aom, rav1e)")]
    av1an_encoder: Av1anEncoder,

    #[arg(long, default_value_t = 1.0, help = "Merge consecutive scenes into one Av1an zone if mean sY delta <= threshold")]
    zone_merge_threshold: f32,

    #[arg(long, help = "Emit per-scene FILM_GRAIN_EVENT JSONL stream for SVT-AV1 4.3.0+")]
    emit_fgs_events: Option<PathBuf>,

    #[arg(long, help = "Directory to generate ct_svt_fgs_bridge.h C integration header for libSvtAv1Enc 4.3.0+")]
    emit_svt_c_bridge: Option<PathBuf>,

    #[arg(long, help = "Export per-scene telemetry JSON for VapourSynth plugin")]
    export_vs_profile: Option<PathBuf>,

    #[arg(long, help = "Generate the VapourSynth Python plugin module (e.g., ct_fgs_vs.py)")]
    emit_vs_plugin: Option<PathBuf>,
}

fn build_entry_for_profile(
    start_f: usize,
    end_f: usize,
    prof: &GrainProfile,
    args: &Args,
    content_type: ContentType,
    digital_noise_mode: Option<u8>,
    target_iso: Option<u32>,
    effective_transfer: TransferCurve,
    effective_range: ColorRange,
    effective_tune_intensity: f32,
    retention_scale: f32,
    width: usize,
    height: usize,
) -> FgsTableEntry {
    let seed = rand_seed(start_f);

    if args.fg_search_full || (args.tune_grain && !args.forced_fg_search_full) {
        if prof.avg_sigma >= 1.15 || digital_noise_mode.is_some() {
            let iso_boost = if args.force_bypass == 1 {
                1.0
            } else {
                target_iso.map(|i| (i as f32 / 100.0).sqrt()).unwrap_or(1.0)
            };
            return FgsTableEntry {
                start_frame: start_f,
                end_frame: end_f,
                lag: prof.lag,
                ar_coeffs_y: prof.ar_coeffs_y.clone(),
                points_sy: prof.profile_sy.clone(),
                scale: effective_tune_intensity * retention_scale * iso_boost,
                seed,
            };
        }
    }

    if let Some(iso_val) = target_iso {
        let pure_digital = digital_noise_mode == Some(1);
        let (lag, ar_coeffs, iso_sy) = generate_iso_photon_profile(
            iso_val,
            width,
            height,
            effective_transfer,
            effective_range,
            pure_digital,
        );
        return FgsTableEntry {
            start_frame: start_f,
            end_frame: end_f,
            lag,
            ar_coeffs_y: ar_coeffs,
            points_sy: iso_sy,
            scale: args.intensity * retention_scale,
            seed,
        };
    }

    let (lag, ar_coeffs, subtle_sy) = get_subtle_70mm_profile(content_type, prof.mean_luma, effective_range);
    FgsTableEntry {
        start_frame: start_f,
        end_frame: end_f,
        lag,
        ar_coeffs_y: ar_coeffs,
        points_sy: subtle_sy,
        scale: args.intensity * retention_scale,
        seed,
    }
}

fn make_vs_telemetry(start_f: usize, end_f: usize, prof: &GrainProfile, denoise_strength: f32) -> VsSceneTelemetry {
    let s = denoise_strength.clamp(0.1, 3.0);
    let denoise_thsad = ((prof.unwanted_noise_sigma * 75.0
        + (prof.blockiness - 1.0).max(0.0) * 55.0)
        * s)
        .round()
        .clamp(40.0, 260.0) as u32;
    let degrain_thsad = (((prof.unwanted_noise_sigma + prof.avg_sigma) * 75.0) * s)
        .round()
        .clamp(denoise_thsad as f32, 380.0) as u32;

    VsSceneTelemetry {
        start_frame: start_f,
        end_frame: end_f,
        unwanted_noise_sigma: prof.unwanted_noise_sigma,
        authentic_grain_sigma: prof.avg_sigma,
        blockiness: prof.blockiness,
        mean_luma: prof.mean_luma,
        denoise_thsad,
        degrain_thsad,
    }
}

fn smooth_profile(window: &mut VecDeque<GrainProfile>, current: GrainProfile, max_len: usize) -> GrainProfile {
    if max_len == 0 {
        return current;
    }
    window.push_back(current.clone());
    while window.len() > max_len {
        window.pop_front();
    }
    let len_f = window.len() as f32;
    GrainProfile {
        lag: current.lag,
        ar_coeffs_y: current.ar_coeffs_y,
        profile_sy: current.profile_sy,
        avg_sigma: window.iter().map(|p| p.avg_sigma).sum::<f32>() / len_f,
        unwanted_noise_sigma: window.iter().map(|p| p.unwanted_noise_sigma).sum::<f32>() / len_f,
        blockiness: window.iter().map(|p| p.blockiness).sum::<f32>() / len_f,
        mean_luma: window.iter().map(|p| p.mean_luma).sum::<f32>() / len_f,
        texture_complexity: window.iter().map(|p| p.texture_complexity).sum::<f32>() / len_f,
        hf_ratio: window.iter().map(|p| p.hf_ratio).sum::<f32>() / len_f,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Some(ref plugin_path) = args.emit_vs_plugin {
        write_vs_plugin_module(plugin_path)?;
        eprintln!("[✓] VapourSynth plugin module written to: {}", plugin_path.display());
        if args.input.is_none() {
            return Ok(());
        }
    }

    let input_path = args.input.as_ref().ok_or("Input path (-i / --input) is required")?;

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

    eprintln!("[*] Initializing CT-AV1-FGS-ENGINE v2.3.0 (Modular Apache-2.0)...");
    let mut stream = Y4mStream::from_input(input_path)?;

    let effective_transfer = match args.transfer {
        TransferCurve::Auto => stream.detected_transfer,
        other => other,
    };
    let effective_range = match args.color_range {
        ColorRange::Auto => stream.detected_range,
        other => other,
    };

    eprintln!(
        "    - Stream: {}x{} | {}-bit | {:?} | FPS: {}/{} | Transfer: {:?} | Range: {:?}",
        stream.width,
        stream.height,
        stream.bit_depth,
        stream.subsampling,
        stream.fps_num,
        stream.fps_den,
        effective_transfer,
        effective_range
    );

    if let Some(iso_val) = target_iso {
        eprintln!("    - ISO Photon-Noise Active: ISO {}", iso_val);
    }

    let retention_scale = compute_fgs_retention_scale(args.grain_retain, args.fgs_balance);
    if args.grain_retain > 0.0 || args.output_y4m.is_some() || args.export_vs_profile.is_some() {
        eprintln!(
            "    - Grain Retention: {:.1}% | FGS Balance: {:?} (FGS Scale: {:.3}) | Denoise Strength: {:.2}x",
            args.grain_retain, args.fgs_balance, retention_scale, args.denoise_strength
        );
    }

    let mutagen_engine = MutagenEngine::new(args.no_noise_bias == 1, digital_noise_mode);
    if args.mutagen == MutagenState::Off {
        eprintln!("[!] WARNING: Mutagen module is OFF. Using fallback analysis.");
    }

    let is_tuning_active = args.tune_grain || digital_noise_mode.is_some();
    let mut content_type_opt = args.force_type;
    let lookahead_window = args.lookahead.unwrap_or(0);

    let mut entries: Vec<FgsTableEntry> = Vec::new();
    let mut vs_telemetry: Vec<VsSceneTelemetry> = Vec::new();
    let mut lookahead_queue: VecDeque<GrainProfile> = VecDeque::new();

    if let Some(ref y4m_out_path) = args.output_y4m {
        eprintln!(
            "[*] Native Rust MCTF + Bilateral Denoiser (Strength: {:.2}x, Detail Protect: {:.2}x, Dither: {}) -> {}",
            args.denoise_strength, args.denoise_detail_protect, args.dither, y4m_out_path.display()
        );
        let mut y4m_writer = Y4mWriter::create(y4m_out_path, &stream.raw_header_line, stream.bit_depth, args.dither)?;

        let provided_scenes = if let Some(ref sc_path) = args.scenes {
            Some(parse_av1an_scenes(sc_path)?)
        } else {
            None
        };

        let mut scene_detector = SceneDetector::new(12, 300);
        let mut ring: VecDeque<(usize, YuvFrame, usize)> = VecDeque::new();
        let mut scene_bounds: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut scene_profiles: HashMap<usize, GrainProfile> = HashMap::new();
        let mut current_scene_id = 0usize;
        let mut eof_reached = false;
        let mut next_frame_idx = 0usize;

        while ring.len() < 3 && !eof_reached {
            if let Some(frame) = stream.read_next_yuv_frame() {
                let ctype = *content_type_opt.get_or_insert_with(|| {
                    let auto = classify_frame_auto(&frame.y, stream.width, stream.height);
                    eprintln!("[*] Content type auto-detected: {:?}", auto);
                    auto
                });

                let is_cut = if let Some(ref scenes) = provided_scenes {
                    scenes.iter().any(|s| s.start_frame == next_frame_idx && next_frame_idx > 0)
                } else {
                    scene_detector.check_scenecut(&frame.y, stream.width, stream.height, next_frame_idx)
                };

                if is_cut {
                    if let Some(bounds) = scene_bounds.get_mut(&current_scene_id) {
                        bounds.1 = next_frame_idx;
                    }
                    current_scene_id += 1;
                }

                scene_bounds
                    .entry(current_scene_id)
                    .and_modify(|b| b.1 = next_frame_idx + 1)
                    .or_insert((next_frame_idx, next_frame_idx + 1));

                if !scene_profiles.contains_key(&current_scene_id) {
                    let analysis_ctype = if is_tuning_active && args.force_bypass >= 1 {
                        ContentType::LiveAction
                    } else {
                        ctype
                    };
                    let prof = if args.mutagen == MutagenState::On {
                        mutagen_engine.analyze_frame(&frame.y, None, None, stream.width, stream.height, analysis_ctype)
                    } else {
                        fallback_analyze_frame(&frame.y, stream.width, stream.height)
                    };
                    scene_profiles.insert(current_scene_id, smooth_profile(&mut lookahead_queue, prof, lookahead_window));
                }

                ring.push_back((next_frame_idx, frame, current_scene_id));
                next_frame_idx += 1;
            } else {
                eof_reached = true;
            }
        }

        let content_type = content_type_opt.unwrap_or(ContentType::LiveAction);
        let (analysis_ctype, effective_tune_intensity) = if is_tuning_active && args.force_bypass >= 1 {
            let eff_int = if args.force_bypass == 1 { 1.0 } else { args.intensity };
            (ContentType::LiveAction, eff_int)
        } else {
            (content_type, args.intensity)
        };

        let mut center_pos = 0usize;

        while center_pos < ring.len() {
            while ring.len() - center_pos <= 2 && !eof_reached {
                if let Some(frame) = stream.read_next_yuv_frame() {
                    let is_cut = if let Some(ref scenes) = provided_scenes {
                        scenes.iter().any(|s| s.start_frame == next_frame_idx && next_frame_idx > 0)
                    } else {
                        scene_detector.check_scenecut(&frame.y, stream.width, stream.height, next_frame_idx)
                    };

                    if is_cut {
                        if let Some(bounds) = scene_bounds.get_mut(&current_scene_id) {
                            bounds.1 = next_frame_idx;
                        }
                        current_scene_id += 1;
                    }

                    scene_bounds
                        .entry(current_scene_id)
                        .and_modify(|b| b.1 = next_frame_idx + 1)
                        .or_insert((next_frame_idx, next_frame_idx + 1));

                    if !scene_profiles.contains_key(&current_scene_id) {
                        let prev_y = ring.back().filter(|r| r.2 == current_scene_id).map(|r| r.1.y.as_slice());
                        let mf = prev_y.map(|p| estimate_motion_field(&frame.y, p, stream.width, stream.height, 16));
                        let prof = if args.mutagen == MutagenState::On {
                            mutagen_engine.analyze_frame(&frame.y, prev_y, mf.as_ref(), stream.width, stream.height, analysis_ctype)
                        } else {
                            fallback_analyze_frame(&frame.y, stream.width, stream.height)
                        };
                        scene_profiles.insert(current_scene_id, smooth_profile(&mut lookahead_queue, prof, lookahead_window));
                    }

                    ring.push_back((next_frame_idx, frame, current_scene_id));
                    next_frame_idx += 1;
                } else {
                    eof_reached = true;
                }
            }

            let (c_idx, ref c_frame, c_sid) = ring[center_pos];
            let get_same_scene = |offset: isize| -> Option<&YuvFrame> {
                let pos = center_pos as isize + offset;
                if pos >= 0 && (pos as usize) < ring.len() && ring[pos as usize].2 == c_sid {
                    Some(&ring[pos as usize].1)
                } else {
                    None
                }
            };

            let neighbors: [Option<&YuvFrame>; 5] = [
                get_same_scene(-2),
                get_same_scene(-1),
                Some(c_frame),
                get_same_scene(1),
                get_same_scene(2),
            ];

            let prof_ref = &scene_profiles[&c_sid];
            let cleaned_frame = denoise_and_degrain_frame(
                &neighbors,
                prof_ref,
                args.grain_retain,
                args.protect_lines,
                args.denoise_strength,
                args.denoise_detail_protect,
                !args.no_spatial_clamp,
            );
            y4m_writer.write_frame(&cleaned_frame)?;

            if c_idx % 120 == 0 {
                eprintln!("    - MCTF Denoised & Emitted Frame {} (Scene {})", c_idx, c_sid + 1);
            }

            if center_pos >= 2 {
                ring.pop_front();
            } else {
                center_pos += 1;
            }
        }

        for sid in 0..=current_scene_id {
            if let (Some(&(s_f, e_f)), Some(prof)) = (scene_bounds.get(&sid), scene_profiles.get(&sid)) {
                entries.push(build_entry_for_profile(
                    s_f,
                    e_f,
                    prof,
                    &args,
                    content_type,
                    digital_noise_mode,
                    target_iso,
                    effective_transfer,
                    effective_range,
                    effective_tune_intensity,
                    retention_scale,
                    stream.width,
                    stream.height,
                ));
                vs_telemetry.push(make_vs_telemetry(s_f, e_f, prof, args.denoise_strength));
            }
        }

        y4m_writer.flush()?;
    } else if args.fg_search_full || args.forced_fg_search_full {
        let first_frame = stream.read_next_luma_only().ok_or("Failed to read initial luma frame")?;
        let content_type = if let Some(forced) = args.force_type {
            eprintln!("[*] Content type forced by user: {:?}", forced);
            forced
        } else {
            let auto = classify_frame_auto(&first_frame, stream.width, stream.height);
            eprintln!("[*] Content type auto-detected: {:?}", auto);
            auto
        };

        let (analysis_ctype, effective_tune_intensity) = if is_tuning_active && args.force_bypass >= 1 {
            let eff_int = if args.force_bypass == 1 {
                eprintln!("    [*] --force-bypass 1 active: Locking --intensity to 1.0 & bypassing content bias.");
                1.0f32
            } else {
                eprintln!("    [*] --force-bypass 2 active: Bypassing content bias; applying --intensity {:.2}.", args.intensity);
                args.intensity
            };
            (ContentType::LiveAction, eff_int)
        } else {
            (content_type, args.intensity)
        };

        if let Some(ref sc_path) = args.scenes {
            let segments = parse_av1an_scenes(sc_path)?;
            eprintln!("    [+] Loaded {} scenes from: {}", segments.len(), sc_path.display());

            for (idx, seg) in segments.iter().enumerate() {
                let mid_point = (seg.start_frame + seg.end_frame) / 2;
                let target_prev = mid_point.saturating_sub(1);
                if stream.current_frame < target_prev {
                    let to_skip = target_prev - stream.current_frame;
                    if !stream.skip_frames(to_skip) {
                        let tail = stream.drain_stderr_tail();
                        if !tail.trim().is_empty() {
                            eprintln!("    [!] Stream ended early at frame {}. Process log:\n{}", stream.current_frame, tail.trim());
                        }
                        break;
                    }
                }

                let prev_opt = if target_prev == stream.current_frame && mid_point > 0 {
                    stream.read_next_luma_only()
                } else {
                    None
                };

                let curr_frame = if stream.current_frame == 0 && mid_point == 0 {
                    first_frame.clone()
                } else if let Some(f) = stream.read_next_luma_only() {
                    f
                } else {
                    break;
                };

                let mf = prev_opt.as_ref().map(|p| estimate_motion_field(&curr_frame, p, stream.width, stream.height, 16));
                let raw_prof = if args.mutagen == MutagenState::On {
                    mutagen_engine.analyze_frame(&curr_frame, prev_opt.as_deref(), mf.as_ref(), stream.width, stream.height, analysis_ctype)
                } else {
                    fallback_analyze_frame(&curr_frame, stream.width, stream.height)
                };

                let prof = smooth_profile(&mut lookahead_queue, raw_prof, lookahead_window);
                entries.push(build_entry_for_profile(
                    seg.start_frame,
                    seg.end_frame,
                    &prof,
                    &args,
                    content_type,
                    digital_noise_mode,
                    target_iso,
                    effective_transfer,
                    effective_range,
                    effective_tune_intensity,
                    retention_scale,
                    stream.width,
                    stream.height,
                ));
                vs_telemetry.push(make_vs_telemetry(seg.start_frame, seg.end_frame, &prof, args.denoise_strength));

                if idx % 10 == 0 || idx == segments.len() - 1 {
                    eprintln!(
                        "    - Segment {}/{} [{}..{}]: Luma={:.1} | Grain={:.2} | Noise={:.2} | Lag={}",
                        idx + 1,
                        segments.len(),
                        seg.start_frame,
                        seg.end_frame,
                        prof.mean_luma,
                        prof.avg_sigma,
                        prof.unwanted_noise_sigma,
                        prof.lag
                    );
                }
            }
        } else {
            eprintln!("[*] Running single-pass native scene detector (no --scenes provided)...");
            let mut scene_detector = SceneDetector::new(12, 300);
            scene_detector.check_scenecut(&first_frame, stream.width, stream.height, 0);

            let mut scene_start = 0usize;
            let mut prev_luma = first_frame.clone();
            let mut scene_mid_pair: Option<(Vec<f32>, Vec<f32>)> = Some((first_frame.clone(), first_frame));
            let mut frame_idx = 1usize;

            while let Some(curr_luma) = stream.read_next_luma_only() {
                let is_cut = scene_detector.check_scenecut(&curr_luma, stream.width, stream.height, frame_idx);

                if frame_idx.saturating_sub(scene_start) == 24 || scene_mid_pair.is_none() {
                    scene_mid_pair = Some((prev_luma.clone(), curr_luma.clone()));
                }

                if is_cut {
                    let (ref p_frame, ref c_frame) = scene_mid_pair.as_ref().unwrap();
                    let mf = estimate_motion_field(c_frame, p_frame, stream.width, stream.height, 16);
                    let raw_prof = if args.mutagen == MutagenState::On {
                        mutagen_engine.analyze_frame(c_frame, Some(p_frame.as_slice()), Some(&mf), stream.width, stream.height, analysis_ctype)
                    } else {
                        fallback_analyze_frame(c_frame, stream.width, stream.height)
                    };

                    let prof = smooth_profile(&mut lookahead_queue, raw_prof, lookahead_window);
                    entries.push(build_entry_for_profile(
                        scene_start,
                        frame_idx,
                        &prof,
                        &args,
                        content_type,
                        digital_noise_mode,
                        target_iso,
                        effective_transfer,
                        effective_range,
                        effective_tune_intensity,
                        retention_scale,
                        stream.width,
                        stream.height,
                    ));
                    vs_telemetry.push(make_vs_telemetry(scene_start, frame_idx, &prof, args.denoise_strength));

                    eprintln!(
                        "    - Native Scene {} [{}..{}]: Luma={:.1} | Grain={:.2} | Lag={}",
                        entries.len(),
                        scene_start,
                        frame_idx,
                        prof.mean_luma,
                        prof.avg_sigma,
                        prof.lag
                    );

                    scene_start = frame_idx;
                    scene_mid_pair = None;
                }

                prev_luma = curr_luma;
                frame_idx += 1;
            }

            if scene_start < frame_idx {
                let (ref p_frame, ref c_frame) = scene_mid_pair.unwrap_or_else(|| (prev_luma.clone(), prev_luma));
                let mf = estimate_motion_field(c_frame, p_frame, stream.width, stream.height, 16);
                let raw_prof = if args.mutagen == MutagenState::On {
                    mutagen_engine.analyze_frame(c_frame, Some(p_frame.as_slice()), Some(&mf), stream.width, stream.height, analysis_ctype)
                } else {
                    fallback_analyze_frame(c_frame, stream.width, stream.height)
                };
                let prof = smooth_profile(&mut lookahead_queue, raw_prof, lookahead_window);
                entries.push(build_entry_for_profile(
                    scene_start,
                    frame_idx,
                    &prof,
                    &args,
                    content_type,
                    digital_noise_mode,
                    target_iso,
                    effective_transfer,
                    effective_range,
                    effective_tune_intensity,
                    retention_scale,
                    stream.width,
                    stream.height,
                ));
                vs_telemetry.push(make_vs_telemetry(scene_start, frame_idx, &prof, args.denoise_strength));
            }
        }
    } else {
        eprintln!("[*] Generating fast static global FGS table...");
        let frame0 = stream.read_next_luma_only().ok_or("Failed to read frame 0")?;
        let frame1_opt = stream.read_next_luma_only();

        let content_type = if let Some(forced) = args.force_type {
            eprintln!("[*] Content type forced by user: {:?}", forced);
            forced
        } else {
            let auto = classify_frame_auto(&frame0, stream.width, stream.height);
            eprintln!("[*] Content type auto-detected: {:?}", auto);
            auto
        };

        let (analysis_ctype, effective_tune_intensity) = if is_tuning_active && args.force_bypass >= 1 {
            let eff_int = if args.force_bypass == 1 { 1.0 } else { args.intensity };
            (ContentType::LiveAction, eff_int)
        } else {
            (content_type, args.intensity)
        };

        let mf = frame1_opt.as_ref().map(|f1| estimate_motion_field(f1, &frame0, stream.width, stream.height, 16));
        let prof = if args.mutagen == MutagenState::On {
            if let Some(ref f1) = frame1_opt {
                mutagen_engine.analyze_frame(f1, Some(&frame0), mf.as_ref(), stream.width, stream.height, analysis_ctype)
            } else {
                mutagen_engine.analyze_frame(&frame0, None, None, stream.width, stream.height, analysis_ctype)
            }
        } else {
            fallback_analyze_frame(&frame0, stream.width, stream.height)
        };

        let entry = build_entry_for_profile(
            0,
            usize::MAX,
            &prof,
            &args,
            content_type,
            digital_noise_mode,
            target_iso,
            effective_transfer,
            effective_range,
            effective_tune_intensity,
            retention_scale,
            stream.width,
            stream.height,
        );
        vs_telemetry.push(make_vs_telemetry(0, 999999999, &prof, args.denoise_strength));
        entries.push(entry);
    }

    let effective_time_unit = if entries.len() == 1 && entries[0].end_frame == usize::MAX {
        TimeUnitMode::Global
    } else {
        args.time_unit
    };

    write_filmgrn1_table(&args.output, &entries, effective_time_unit, stream.fps_num, stream.fps_den)?;
    eprintln!("[✓] FGS table ({:?}) written to: {}", effective_time_unit, args.output.display());

    if args.emit_chunk_tables.is_some() || args.emit_av1an_zones.is_some() {
        let chunk_dir = args
            .emit_chunk_tables
            .clone()
            .unwrap_or_else(|| PathBuf::from("ct_av1an_zones_tbl"));

        let zone_count = write_av1an_chunks_and_zones(
            &chunk_dir,
            args.emit_av1an_zones.as_deref(),
            args.av1an_encoder,
            &entries,
            args.zone_merge_threshold,
            stream.fps_num,
            stream.fps_den,
        )?;

        eprintln!(
            "[✓] Emitted {} merged t=0 zone tables into: {}",
            zone_count,
            chunk_dir.display()
        );
        if let Some(ref zp) = args.emit_av1an_zones {
            eprintln!(
                "[✓] Av1an zones file written to: {} (Pass via `av1an --zones {}` WITHOUT --fgs-table inside -v)",
                zp.display(),
                zp.display()
            );
        }
    }

    if let Some(ref events_path) = args.emit_fgs_events {
        write_fgs_events_jsonl(events_path, &entries, stream.bit_depth)?;
        eprintln!("[✓] SVT-AV1 FILM_GRAIN_EVENT JSONL written to: {}", events_path.display());
    }

    if let Some(ref bridge_dir) = args.emit_svt_c_bridge {
        write_svt_c_bridge(bridge_dir, &entries, stream.bit_depth)?;
        eprintln!("[✓] SVT-AV1 4.3.0+ C Bridge header written to: {}/ct_svt_fgs_bridge.h", bridge_dir.display());
    }

    if let Some(ref vs_prof_path) = args.export_vs_profile {
        write_vs_telemetry_json(vs_prof_path, args.grain_retain, args.fgs_balance, &vs_telemetry)?;
        eprintln!("[✓] VapourSynth telemetry exported to: {}", vs_prof_path.display());
    }

    Ok(())
}
