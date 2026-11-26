use crate::mutagen::default_70mm_ar_coeffs;
use crate::types::{Av1anEncoder, FgsBalanceMode, FgsTableEntry, TimeUnitMode, VsSceneTelemetry};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub fn rand_seed(salt: usize) -> u16 {
    let mut x = salt.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((x ^ (x >> 31)) % 64000 + 1000) as u16
}

pub fn frame_to_100ns(frame: usize, fps_num: u64, fps_den: u64) -> u64 {
    let num = (frame as u128) * 10_000_000u128 * (fps_den as u128);
    let den = fps_num.max(1) as u128;
    ((num + den / 2) / den) as u64
}

pub fn compute_fgs_retention_scale(grain_retain_pct: f32, balance_mode: FgsBalanceMode) -> f32 {
    let r = (grain_retain_pct / 100.0).clamp(0.0, 1.0);
    match balance_mode {
        FgsBalanceMode::Full => 1.0,
        FgsBalanceMode::Linear => (1.0 - r).max(0.0),
        FgsBalanceMode::Complement => (1.0 - r * r).max(0.0).sqrt(),
    }
}

/// Computes a duration-weighted global FGS entry across all analyzed scenes so
/// global-mode tables accurately represent the entire feature rather than Scene 0 alone.
fn aggregate_global_entry(entries: &[FgsTableEntry]) -> FgsTableEntry {
    if entries.len() == 1 {
        return entries[0].clone();
    }

    let mut total_weight = 0.0f64;
    let mut lag_votes = [0.0f64; 4];

    for e in entries {
        let dur = e.end_frame.saturating_sub(e.start_frame).max(1) as f64;
        let avg_v: f64 = e.points_sy.iter().map(|&(_, v)| v as f64).sum::<f64>()
            / (e.points_sy.len().max(1) as f64);
        // Down-weight pure black fade/logo scenes when richer scenes exist
        let w = if avg_v <= 1.1 { dur * 0.1 } else { dur };
        total_weight += w;
        let l = (e.lag as usize).min(3);
        lag_votes[l] += w;
    }

    let dominant_lag = lag_votes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u8)
        .unwrap_or(2);

    let ref_entry = entries
        .iter()
        .find(|e| e.lag == dominant_lag)
        .unwrap_or(&entries[0]);

    let num_pts = ref_entry.points_sy.len();
    let mut agg_pts = Vec::with_capacity(num_pts);
    for pi in 0..num_pts {
        let luma = ref_entry.points_sy[pi].0;
        let mut acc_v = 0.0f64;
        let mut w_acc = 0.0f64;
        for e in entries {
            if e.points_sy.len() == num_pts {
                let dur = e.end_frame.saturating_sub(e.start_frame).max(1) as f64;
                let avg_v: f64 = e.points_sy.iter().map(|&(_, v)| v as f64).sum::<f64>()
                    / (e.points_sy.len().max(1) as f64);
                let w = if avg_v <= 1.1 { dur * 0.1 } else { dur };
                acc_v += w * ((e.points_sy[pi].1 as f64) * (e.scale as f64));
                w_acc += w;
            }
        }
        let v = if w_acc > 0.0 {
            (acc_v / w_acc).round().clamp(1.0, 255.0) as u8
        } else {
            ref_entry.points_sy[pi].1
        };
        agg_pts.push((luma, v));
    }

    let ar_len = 2 * (dominant_lag as usize) * ((dominant_lag as usize) + 1);
    let mut agg_ar = vec![0.0f64; ar_len];
    let mut ar_w = 0.0f64;

    for e in entries {
        if e.lag == dominant_lag && e.ar_coeffs_y.len() == ar_len {
            let dur = e.end_frame.saturating_sub(e.start_frame).max(1) as f64;
            for (k, &c) in e.ar_coeffs_y.iter().enumerate() {
                agg_ar[k] += dur * (c as f64);
            }
            ar_w += dur;
        }
    }

    let final_ar: Vec<i32> = if ar_w > 0.0 {
        agg_ar.into_iter().map(|v| (v / ar_w).round().clamp(-128.0, 127.0) as i32).collect()
    } else {
        default_70mm_ar_coeffs(dominant_lag)
    };

    let _ = total_weight;
    FgsTableEntry {
        start_frame: 0,
        end_frame: usize::MAX,
        lag: dominant_lag,
        ar_coeffs_y: final_ar,
        points_sy: agg_pts,
        scale: 1.0,
        seed: entries[0].seed,
    }
}

pub fn write_filmgrn1_table(
    path: &Path,
    entries: &[FgsTableEntry],
    time_unit: TimeUnitMode,
    fps_num: u64,
    fps_den: u64,
) -> std::io::Result<()> {
    let mut out = File::create(path)?;
    writeln!(out, "filmgrn1")?;

    if time_unit == TimeUnitMode::Global && !entries.is_empty() {
        let agg = aggregate_global_entry(entries);
        write_single_entry(&mut out, 0, u64::MAX, &agg)?;
        return Ok(());
    }

    for e in entries {
        let (start_ts, end_ts) = match time_unit {
            TimeUnitMode::Ns100 => (
                frame_to_100ns(e.start_frame, fps_num, fps_den),
                if e.end_frame == usize::MAX {
                    u64::MAX
                } else {
                    frame_to_100ns(e.end_frame, fps_num, fps_den)
                },
            ),
            TimeUnitMode::Frames | TimeUnitMode::Global => (
                e.start_frame as u64,
                if e.end_frame == usize::MAX {
                    u64::MAX
                } else {
                    e.end_frame as u64
                },
            ),
        };
        write_single_entry(&mut out, start_ts, end_ts, e)?;
    }

    Ok(())
}

/// Clusters consecutive entries with similar grain characteristics to prevent redundant mini-chunks
/// in Av1an, writes t=0 chunk tables, and optionally emits a ready-to-use Av1an zones file.
pub fn write_av1an_chunks_and_zones(
    chunk_dir: &Path,
    zones_path: Option<&Path>,
    encoder: Av1anEncoder,
    entries: &[FgsTableEntry],
    merge_threshold: f32,
    fps_num: u64,
    fps_den: u64,
) -> std::io::Result<usize> {
    fs::create_dir_all(chunk_dir)?;
    let abs_dir = fs::canonicalize(chunk_dir).unwrap_or_else(|_| chunk_dir.to_path_buf());

    // Merge consecutive scenes whose lag matches and mean scaled sY delta <= merge_threshold
    let mut merged: Vec<FgsTableEntry> = Vec::new();
    for e in entries {
        if let Some(last) = merged.last_mut() {
            if last.lag == e.lag && last.points_sy.len() == e.points_sy.len() {
                let mean_delta: f32 = last
                    .points_sy
                    .iter()
                    .zip(e.points_sy.iter())
                    .map(|((_, a), (_, b))| {
                        ((*a as f32) * last.scale - (*b as f32) * e.scale).abs()
                    })
                    .sum::<f32>()
                    / (last.points_sy.len().max(1) as f32);

                if mean_delta <= merge_threshold {
                    last.end_frame = e.end_frame;
                    continue;
                }
            }
        }
        merged.push(e.clone());
    }

    let mut zones_file = if let Some(zp) = zones_path {
        Some(File::create(zp)?)
    } else {
        None
    };

    let (enc_name, flag_fmt) = match encoder {
        Av1anEncoder::SvtAv1 => ("svt-av1", "--fgs-table"),
        Av1anEncoder::Aom => ("aom", "--film-grain-table="),
        Av1anEncoder::Rav1e => ("rav1e", "--photon-noise-table"),
    };

    for (idx, e) in merged.iter().enumerate() {
        let chunk_file = abs_dir.join(format!("zone_{:05}.tbl", idx));
        let mut out = File::create(&chunk_file)?;
        writeln!(out, "filmgrn1")?;
        let duration_frames = e.end_frame.saturating_sub(e.start_frame).max(1);
        let _end_100ns = frame_to_100ns(duration_frames, fps_num, fps_den);
        // Write u64::MAX end timestamp per t=0 zone table so chunks never drop grain early
        write_single_entry(&mut out, 0, u64::MAX, e)?;

        if let Some(ref mut zf) = zones_file {
            if encoder == Av1anEncoder::Aom {
                writeln!(
                    zf,
                    "{} {} {} {}{}",
                    e.start_frame,
                    e.end_frame,
                    enc_name,
                    flag_fmt,
                    chunk_file.display()
                )?;
            } else {
                writeln!(
                    zf,
                    "{} {} {} {} {}",
                    e.start_frame,
                    e.end_frame,
                    enc_name,
                    flag_fmt,
                    chunk_file.display()
                )?;
            }
        }
    }

    Ok(merged.len())
}

fn write_single_entry(
    out: &mut File,
    start_ts: u64,
    end_ts: u64,
    e: &FgsTableEntry,
) -> std::io::Result<()> {
    let lag = e.lag.min(3);
    let expected_luma_coeffs = 2 * (lag as usize) * ((lag as usize) + 1);
    let ar_coeffs = if e.ar_coeffs_y.len() == expected_luma_coeffs {
        e.ar_coeffs_y.clone()
    } else {
        default_70mm_ar_coeffs(lag)
    };

    let num_pos_chroma = expected_luma_coeffs + 1;
    let c_cb_cr = vec!["0"; num_pos_chroma].join(" ");
    let c_y_str = ar_coeffs
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    let scaled_points: Vec<String> = e
        .points_sy
        .iter()
        .map(|&(l, v)| {
            let scaled_v = ((v as f32) * e.scale).round().clamp(1.0, 255.0) as u8;
            format!("{l} {scaled_v}")
        })
        .collect();

    let sy_str = format!("{}  {}", scaled_points.len(), scaled_points.join(" "));

    writeln!(out, "E {start_ts} {end_ts} 1 {} 1", e.seed)?;
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

pub fn write_fgs_events_jsonl(
    path: &Path,
    entries: &[FgsTableEntry],
    bit_depth: usize,
) -> std::io::Result<()> {
    let mut out = File::create(path)?;
    for e in entries {
        let lag = e.lag.min(3);
        let expected_luma_coeffs = 2 * (lag as usize) * ((lag as usize) + 1);
        let ar_coeffs = if e.ar_coeffs_y.len() == expected_luma_coeffs {
            e.ar_coeffs_y.clone()
        } else {
            default_70mm_ar_coeffs(lag)
        };

        let scaled_pts: Vec<(u8, u8)> = e
            .points_sy
            .iter()
            .map(|&(l, v)| {
                let sv = ((v as f32) * e.scale).round().clamp(1.0, 255.0) as u8;
                (l, sv)
            })
            .collect();

        let pts_json: Vec<String> = scaled_pts
            .iter()
            .map(|(l, v)| format!("[{l},{v}]"))
            .collect();
        let coeffs_json: Vec<String> = ar_coeffs.iter().map(|c| c.to_string()).collect();

        writeln!(
            out,
            "{{\"event\":\"FILM_GRAIN_EVENT\",\"start_frame\":{},\"end_frame\":{},\"apply_grain\":1,\"update_grain\":1,\"bit_depth\":{},\"ar_coeff_lag\":{},\"ar_coeff_shift\":6,\"grain_scale_shift\":0,\"scaling_shift\":8,\"overlap_flag\":1,\"chroma_scaling_from_luma\":1,\"random_seed\":{},\"scaling_points_y\":[{}],\"ar_coeffs_y\":[{}]}}",
            e.start_frame,
            e.end_frame,
            bit_depth,
            lag,
            e.seed,
            pts_json.join(","),
            coeffs_json.join(",")
        )?;
    }
    Ok(())
}

pub fn write_svt_c_bridge(
    dir: &Path,
    entries: &[FgsTableEntry],
    bit_depth: usize,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let header_path = dir.join("ct_svt_fgs_bridge.h");
    let mut h = File::create(&header_path)?;

    writeln!(h, "/* Auto-generated by CT-AV1-FGS-ENGINE v2.3.0 (Apache-2.0) */")?;
    writeln!(h, "/* Provides static AomFilmGrain models for SVT-AV1 4.3.0+ FILM_GRAIN_EVENT API */")?;
    writeln!(h, "#ifndef CT_SVT_FGS_BRIDGE_H")?;
    writeln!(h, "#define CT_SVT_FGS_BRIDGE_H")?;
    writeln!(h)?;
    writeln!(h, "#include <stdint.h>")?;
    writeln!(h, "#include <stddef.h>")?;
    writeln!(h, "#include <string.h>")?;
    writeln!(h, "#include \"EbSvtAv1Enc.h\"")?;
    writeln!(h, "#include \"EbSvtAv1Formats.h\"")?;
    writeln!(h)?;
    writeln!(h, "typedef struct {{")?;
    writeln!(h, "    uint64_t start_frame;")?;
    writeln!(h, "    uint64_t end_frame;")?;
    writeln!(h, "    AomFilmGrain grain;")?;
    writeln!(h, "}} CtFgsSceneEvent;")?;
    writeln!(h)?;
    writeln!(h, "static CtFgsSceneEvent CT_FGS_EVENTS[] = {{")?;

    for e in entries {
        let lag = e.lag.min(3);
        let expected_luma_coeffs = 2 * (lag as usize) * ((lag as usize) + 1);
        let ar_coeffs = if e.ar_coeffs_y.len() == expected_luma_coeffs {
            e.ar_coeffs_y.clone()
        } else {
            default_70mm_ar_coeffs(lag)
        };

        let mut pts_arr = vec!["{0, 0}".to_string(); 14];
        for (i, &(l, v)) in e.points_sy.iter().take(14).enumerate() {
            let sv = ((v as f32) * e.scale).round().clamp(1.0, 255.0) as u8;
            pts_arr[i] = format!("{{{l}, {sv}}}");
        }

        let mut coeffs_arr = vec!["0".to_string(); 24];
        for (i, &c) in ar_coeffs.iter().take(24).enumerate() {
            coeffs_arr[i] = c.to_string();
        }

        let end_u64 = if e.end_frame == usize::MAX {
            u64::MAX
        } else {
            e.end_frame as u64
        };

        writeln!(h, "    {{")?;
        writeln!(h, "        .start_frame = {}ULL,", e.start_frame)?;
        writeln!(h, "        .end_frame = {}ULL,", end_u64)?;
        writeln!(h, "        .grain = {{")?;
        writeln!(h, "            .apply_grain = 1,")?;
        writeln!(h, "            .update_grain = 1,")?;
        writeln!(h, "            .random_seed = {},", e.seed)?;
        writeln!(h, "            .num_y_points = {},", e.points_sy.len().min(14))?;
        writeln!(h, "            .scaling_points_y = {{{}}},", pts_arr.join(", "))?;
        writeln!(h, "            .num_cb_points = 0,")?;
        writeln!(h, "            .num_cr_points = 0,")?;
        writeln!(h, "            .scaling_shift = 8,")?;
        writeln!(h, "            .ar_coeff_lag = {},", lag)?;
        writeln!(h, "            .ar_coeffs_y = {{{}}},", coeffs_arr.join(", "))?;
        writeln!(h, "            .ar_coeff_shift = 6,")?;
        writeln!(h, "            .grain_scale_shift = 0,")?;
        writeln!(h, "            .chroma_scaling_from_luma = 1,")?;
        writeln!(h, "            .overlap_flag = 1,")?;
        writeln!(h, "            .bit_depth = {}", bit_depth)?;
        writeln!(h, "        }}")?;
        writeln!(h, "    }},")?;
    }

    writeln!(h, "}};")?;
    writeln!(h)?;
    writeln!(h, "static const size_t CT_FGS_EVENT_COUNT = {};", entries.len())?;
    writeln!(h)?;
    writeln!(h, "static inline void ct_attach_film_grain_event(")?;
    writeln!(h, "    EbBufferHeaderType *pic_hdr,")?;
    writeln!(h, "    EbPrivDataNode *node,")?;
    writeln!(h, "    uint64_t frame_index")?;
    writeln!(h, ") {{")?;
    writeln!(h, "    for (size_t i = 0; i < CT_FGS_EVENT_COUNT; ++i) {{")?;
    writeln!(h, "        if (frame_index >= CT_FGS_EVENTS[i].start_frame && frame_index < CT_FGS_EVENTS[i].end_frame) {{")?;
    writeln!(h, "            memset(node, 0, sizeof(EbPrivDataNode));")?;
    writeln!(h, "            node->node_type = FILM_GRAIN_EVENT;")?;
    writeln!(h, "            node->data = &CT_FGS_EVENTS[i].grain;")?;
    writeln!(h, "            node->size = sizeof(AomFilmGrain);")?;
    writeln!(h, "            node->next = pic_hdr->pic_priv_data;")?;
    writeln!(h, "            pic_hdr->pic_priv_data = node;")?;
    writeln!(h, "            return;")?;
    writeln!(h, "        }}")?;
    writeln!(h, "    }}")?;
    writeln!(h, "}}")?;
    writeln!(h)?;
    writeln!(h, "#endif /* CT_SVT_FGS_BRIDGE_H */")?;

    Ok(())
}

pub fn write_vs_telemetry_json(
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
    writeln!(f, "  \"version\": \"2.3.0\",")?;
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

pub fn write_vs_plugin_module(path: &Path) -> std::io::Result<()> {
    let plugin_code = r####"# -*- coding: utf-8 -*-
"""
CT-AV1-FGS-ENGINE VapourSynth Auto-Denoise & Auto-Degrain Plugin Module
Licensed under the Apache License, Version 2.0

Supports both MVTools + RemoveGrain (when installed) and a built-in pure
VapourSynth core.std fallback with zero external plugins required.
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


def _pure_core_auto_degrain(clip16, retain_weight, protect_lines):
    med = core.std.Median(clip16)
    clean_spatial = core.std.Expr([clip16, med], expr="x y - abs 1800 > y x ?")

    temp_light = core.std.AverageFrames(clean_spatial, weights=[1, 4, 1], scenechange=True)
    temp_strong = core.std.AverageFrames(clean_spatial, weights=[1, 2, 4, 2, 1], scenechange=True)

    if retain_weight >= 0.999:
        out = temp_light
    elif retain_weight <= 0.001:
        out = temp_strong
    else:
        out = core.std.Merge(temp_strong, temp_light, weight=retain_weight)

    if protect_lines:
        luma = core.std.ShufflePlanes(clean_spatial, planes=0, colorfamily=vs.GRAY)
        edges = core.std.Prewitt(luma)
        mask = core.std.Expr(edges, expr="x 2200 > 65535 0 ?")
        out = core.std.MaskedMerge(out, clean_spatial, mask)
    return out


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

    has_mv_rgvs = hasattr(core, "mv") and hasattr(core, "rgvs")
    if not has_mv_rgvs:
        processed = _pure_core_auto_degrain(clip16, retain_weight, protect_lines)
        if orig_format.bits_per_sample < 16:
            return core.resize.Point(processed, format=orig_format, dither_type="error_diffusion")
        return processed

    clean_spatial = core.rgvs.RemoveGrain(clip16, mode=1)
    pref = core.rgvs.RemoveGrain(clean_spatial, mode=11)

    super_render = core.mv.Super(clean_spatial, pel=2, sharp=2)
    super_search = core.mv.Super(pref, pel=2, sharp=2)

    bv2 = core.mv.Analyse(super_search, isb=True, delta=2, overlap=overlap, blksize=blksize, truemotion=False)
    bv1 = core.mv.Analyse(super_search, isb=True, delta=1, overlap=overlap, blksize=blksize, truemotion=False)
    fv1 = core.mv.Analyse(super_search, isb=False, delta=1, overlap=overlap, blksize=blksize, truemotion=False)
    fv2 = core.mv.Analyse(super_search, isb=False, delta=2, overlap=overlap, blksize=blksize, truemotion=False)

    edge_mask = None
    if protect_lines:
        luma_pref = core.std.ShufflePlanes(pref, planes=0, colorfamily=vs.GRAY)
        edges = core.std.Prewitt(luma_pref)
        edge_mask = core.std.Expr(edges, expr="x 2200 > 65535 0 ?").rgvs.RemoveGrain(mode=11)

    if telemetry_scenes:
        unique_pairs = sorted(set(
            (
                thsad_denoise_override if thsad_denoise_override is not None else int(s["denoise_thsad"]),
                thsad_degrain_override if thsad_degrain_override is not None else int(s["degrain_thsad"]),
            )
            for s in telemetry_scenes
        ))

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
"####;

    let mut f = File::create(path)?;
    f.write_all(plugin_code.as_bytes())?;
    Ok(())
}
