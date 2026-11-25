use crate::motion::{downsample_2x, estimate_motion_field, satd_4x4};
use crate::types::SceneSegment;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::Path;

/// Evaluates directional spatial intra complexity via 4x4 Hadamard SATD.
pub fn estimate_intra_cost(plane: &[f32], width: usize, height: usize) -> f64 {
    let bx_count = width / 8;
    let by_count = height / 8;
    if bx_count <= 1 || by_count <= 1 {
        return 1.0;
    }

    let total_cost: f64 = (1..by_count)
        .into_par_iter()
        .map(|by| {
            let mut row_cost = 0.0f64;
            for bx in 1..bx_count {
                let y0 = by * 8;
                let x0 = bx * 8;

                for sy in 0..2 {
                    for sx in 0..2 {
                        let py = y0 + sy * 4;
                        let px = x0 + sx * 4;

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

/// Computes normalized 32-bin luma histogram distance (Bhattacharyya-like metric).
fn compute_hist_distance(curr: &[f32], prev: &[f32]) -> f64 {
    let mut h_curr = [0.0f64; 32];
    let mut h_prev = [0.0f64; 32];

    for &p in curr {
        let bin = ((p / 8.0) as usize).min(31);
        h_curr[bin] += 1.0;
    }
    for &p in prev {
        let bin = ((p / 8.0) as usize).min(31);
        h_prev[bin] += 1.0;
    }

    let n_curr = curr.len().max(1) as f64;
    let n_prev = prev.len().max(1) as f64;

    let mut b_coeff = 0.0f64;
    for i in 0..32 {
        b_coeff += ((h_curr[i] / n_curr) * (h_prev[i] / n_prev)).sqrt();
    }
    (1.0 - b_coeff).clamp(0.0, 1.0)
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
                let imp_weight = (1.0 + var_c.sqrt() * 0.05) as f64;
                row_diff += ((sad / 64.0) as f64 + mean_delta as f64) * imp_weight;
            }
            row_diff
        })
        .sum();

    total_diff / ((bx_count * by_count).max(1) as f64)
}

/// Native scene cut detector combining av-scenechange Intra/Inter SATD, Importance Diff,
/// Luma Histogram shift, and rolling window outlier detection.
pub struct SceneDetector {
    pub min_scenecut: usize,
    pub max_scenecut: usize,
    prev_ds: Option<Vec<f32>>,
    last_cut_frame: usize,
    inter_history: VecDeque<f64>,
}

impl SceneDetector {
    pub fn new(min_scenecut: usize, max_scenecut: usize) -> Self {
        Self {
            min_scenecut,
            max_scenecut,
            prev_ds: None,
            last_cut_frame: 0,
            inter_history: VecDeque::with_capacity(12),
        }
    }

    pub fn check_scenecut(&mut self, luma: &[f32], width: usize, height: usize, frame_idx: usize) -> bool {
        let (curr_ds, dw, dh) = downsample_2x(luma, width, height);

        let Some(ref prev_ds) = self.prev_ds else {
            self.prev_ds = Some(curr_ds);
            return false;
        };

        let dist = frame_idx.saturating_sub(self.last_cut_frame);
        if dist >= self.max_scenecut {
            self.last_cut_frame = frame_idx;
            self.prev_ds = Some(curr_ds);
            self.inter_history.clear();
            return true;
        }

        if dist < self.min_scenecut {
            self.prev_ds = Some(curr_ds);
            return false;
        }

        let (intra_cost, (inter_cost, (imp_diff, hist_dist))) = rayon::join(
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
                    || {
                        rayon::join(
                            || estimate_importance_block_diff(&curr_ds, prev_ds, dw, dh),
                            || compute_hist_distance(&curr_ds, prev_ds),
                        )
                    },
                )
            },
        );

        let avg_hist_inter = if self.inter_history.is_empty() {
            inter_cost.max(1.0)
        } else {
            self.inter_history.iter().sum::<f64>() / (self.inter_history.len() as f64)
        };

        const BIAS: f64 = 0.70;
        let threshold = (intra_cost / 16.0) * (1.0 - BIAS);

        // Require both structural SATD shift and global histogram/importance corroboration
        let strong_satd_cut = inter_cost > threshold.max(7.5) && imp_diff > 9.5 && hist_dist > 0.045;
        let relative_spike_cut =
            inter_cost > avg_hist_inter * 2.8 && imp_diff > 12.0 && hist_dist > 0.07;

        let is_cut = strong_satd_cut || relative_spike_cut;

        self.inter_history.push_back(inter_cost);
        if self.inter_history.len() > 10 {
            self.inter_history.pop_front();
        }
        self.prev_ds = Some(curr_ds);

        if is_cut {
            self.last_cut_frame = frame_idx;
            self.inter_history.clear();
            true
        } else {
            false
        }
    }
}

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
