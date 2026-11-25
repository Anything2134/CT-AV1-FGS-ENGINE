use crate::motion::{downsample_2x, estimate_motion_field, satd_4x4};
use crate::types::SceneSegment;
use rayon::prelude::*;
use std::path::Path;

/// Evaluates directional spatial intra complexity via 4x4 Hadamard SATD (av-scenechange style).
pub fn estimate_intra_cost(plane: &[f32], width: usize, height: usize) -> f64 {
    let bx_count = width / 8;
    let by_count = height / 8;
    if bx_count == 0 || by_count == 0 {
        return 1.0;
    }

    let total_cost: f64 = (1..by_count)
        .into_par_iter()
        .map(|by| {
            let mut row_cost = 0.0f64;
            for bx in 1..bx_count {
                let y0 = by * 8;
                let x0 = bx * 8;

                // Evaluate simple DC & directional predictors per 4x4 sub-block
                for sy in 0..2 {
                    for sx in 0..2 {
                        let py = y0 + sy * 4;
                        let px = x0 + sx * 4;

                        // DC prediction from top and left boundary pixels
                        let mut dc_acc = 0.0f32;
                        for k in 0..4 {
                            dc_acc += plane[(py - 1) * width + px + k];
                            dc_acc += plane[(py + k) * width + px - 1];
                        }
                        let dc = dc_acc * 0.125;

                        let mut dc_pred = [0.0f32; 16];
                        let mut h_pred = [0.0f32; 16];
                        let mut v_pred = [0.0f32; 16];

                        for r in 0..4 {
                            let left = plane[(py + r) * width + px - 1];
                            for c in 0..4 {
                                let top = plane[(py - 1) * width + px + c];
                                dc_pred[r * 4 + c] = dc;
                                h_pred[r * 4 + c] = left;
                                v_pred[r * 4 + c] = top;
                            }
                        }

                        let c_dc = satd_4x4(&plane[py * width + px..], width, &dc_pred, 4);
                        let c_h = satd_4x4(&plane[py * width + px..], width, &h_pred, 4);
                        let c_v = satd_4x4(&plane[py * width + px..], width, &v_pred, 4);
                        row_cost += c_dc.min(c_h).min(c_v) as f64;
                    }
                }
            }
            row_cost
        })
        .sum();

    let count = ((bx_count - 1) * (by_count - 1)).max(1) as f64;
    total_cost / count
}

/// Evaluates structural importance block difference between consecutive frames.
pub fn estimate_importance_block_diff(curr: &[f32], prev: &[f32], width: usize, height: usize) -> f64 {
    let bx_count = width / 8;
    let by_count = height / 8;
    if bx_count == 0 || by_count == 0 {
        return 0.0;
    }

    let total_diff: f64 = (0..by_count)
        .into_par_iter()
        .map(|by| {
            let mut row_diff = 0.0f64;
            for bx in 0..bx_count {
                let y0 = by * 8;
                let x0 = bx * 8;

                let mut sum_c = 0.0f32;
                let mut sum_p = 0.0f32;
                let mut sq_c = 0.0f32;
                let mut sad = 0.0f32;

                for y in 0..8 {
                    let row = (y0 + y) * width + x0;
                    for x in 0..8 {
                        let pc = curr[row + x];
                        let pp = prev[row + x];
                        sum_c += pc;
                        sum_p += pp;
                        sq_c += pc * pc;
                        sad += (pc - pp).abs();
                    }
                }

                let mean_c = sum_c / 64.0;
                let var_c = ((sq_c / 64.0) - mean_c * mean_c).max(0.0);
                let mean_delta = (sum_c - sum_p).abs() / 64.0;

                // Weight temporal difference by spatial block variance importance
                let imp_weight = (1.0 + var_c.sqrt() * 0.05) as f64;
                row_diff += ((sad / 64.0) as f64 + mean_delta as f64) * imp_weight;
            }
            row_diff
        })
        .sum();

    total_diff / ((bx_count * by_count).max(1) as f64)
}

/// Native scene cut detector combining av-scenechange's Intra/Inter SATD and Importance metrics.
pub struct SceneDetector {
    pub min_scenecut: usize,
    pub max_scenecut: usize,
    prev_ds: Option<Vec<f32>>,
    ds_width: usize,
    ds_height: usize,
    last_cut_frame: usize,
    prev_inter_cost: f64,
}

impl SceneDetector {
    pub fn new(min_scenecut: usize, max_scenecut: usize) -> Self {
        Self {
            min_scenecut,
            max_scenecut,
            prev_ds: None,
            ds_width: 0,
            ds_height: 0,
            last_cut_frame: 0,
            prev_inter_cost: 0.0,
        }
    }

    /// Evaluates if `frame_idx` is an optimal scene cut boundary.
    pub fn check_scenecut(&mut self, luma: &[f32], width: usize, height: usize, frame_idx: usize) -> bool {
        let (curr_ds, dw, dh) = downsample_2x(luma, width, height);
        self.ds_width = dw;
        self.ds_height = dh;

        let Some(ref prev_ds) = self.prev_ds else {
            self.prev_ds = Some(curr_ds);
            return false;
        };

        let dist = frame_idx.saturating_sub(self.last_cut_frame);
        if dist >= self.max_scenecut {
            self.last_cut_frame = frame_idx;
            self.prev_ds = Some(curr_ds);
            return true;
        }

        if dist < self.min_scenecut {
            self.prev_ds = Some(curr_ds);
            return false;
        }

        // Compute Intra SATD, Motion-Compensated Inter SATD, and Importance Block Diff in parallel
        let (intra_cost, (inter_cost, imp_diff)) = rayon::join(
            || estimate_intra_cost(&curr_ds, dw, dh),
            || {
                rayon::join(
                    || {
                        let mf = estimate_motion_field(&curr_ds, prev_ds, dw, dh, 16);
                        if mf.vectors.is_empty() {
                            0.0
                        } else {
                            mf.vectors.iter().map(|v| v.satd as f64).sum::<f64>() / (mf.vectors.len() as f64)
                        }
                    },
                    || estimate_importance_block_diff(&curr_ds, prev_ds, dw, dh),
                )
            },
        );

        const BIAS: f64 = 0.70;
        let threshold = (intra_cost / 16.0) * (1.0 - BIAS);

        // Flash protection: avoid triggering on single-frame brightness spikes that immediately settle
        let is_cut = (inter_cost > threshold.max(8.5) && imp_diff > 10.0)
            || (inter_cost > self.prev_inter_cost * 3.2 && imp_diff > 14.0);

        self.prev_inter_cost = inter_cost;
        self.prev_ds = Some(curr_ds);

        if is_cut {
            self.last_cut_frame = frame_idx;
            true
        } else {
            false
        }
    }
}

/// Parses Av1an JSON or CSV scenes files with strict chronological deduplication.
pub fn parse_av1an_scenes(path: &Path) -> Result<Vec<SceneSegment>, String> {
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
