use crate::motion::{estimate_motion_field, sample_bilinear};
use crate::types::{ChromaSubsampling, GrainProfile, MotionField, YuvFrame};
use rayon::prelude::*;

/// Non-destructive 3x3 impulse/mosquito spike clamp (clean-room RemoveGrain mode 1 equivalent).
pub fn spatial_impulse_clamp(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = plane.to_vec();
    if width < 3 || height < 3 {
        return out;
    }

    out.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
        if y == 0 || y == height - 1 {
            return;
        }
        let r0 = (y - 1) * width;
        let r1 = y * width;
        let r2 = (y + 1) * width;

        for x in 1..width - 1 {
            let n0 = plane[r0 + x - 1];
            let n1 = plane[r0 + x];
            let n2 = plane[r0 + x + 1];
            let n3 = plane[r1 + x - 1];
            let center = plane[r1 + x];
            let n4 = plane[r1 + x + 1];
            let n5 = plane[r2 + x - 1];
            let n6 = plane[r2 + x];
            let n7 = plane[r2 + x + 1];

            let min_n = n0.min(n1).min(n2).min(n3).min(n4).min(n5).min(n6).min(n7);
            let max_n = n0.max(n1).max(n2).max(n3).max(n4).max(n5).max(n6).max(n7);
            row_out[x] = center.clamp(min_n, max_n);
        }
    });

    out
}

/// Computes per-pixel Scharr edge & local texture protection mask [0.0..1.0].
fn compute_detail_protection_mask(
    plane: &[f32],
    width: usize,
    height: usize,
    detail_protect: f32,
) -> Vec<f32> {
    let mut mask = vec![0.0f32; width * height];
    if width < 3 || height < 3 || detail_protect <= 0.0 {
        return mask;
    }

    let edge_floor = (5.0 / detail_protect.max(0.1)).clamp(1.5, 15.0);
    let edge_span = (8.0 / detail_protect.max(0.1)).clamp(3.0, 20.0);

    mask.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
        if y == 0 || y == height - 1 {
            return;
        }
        let r0 = (y - 1) * width;
        let r1 = y * width;
        let r2 = (y + 1) * width;

        for x in 1..width - 1 {
            let p00 = plane[r0 + x - 1];
            let p01 = plane[r0 + x];
            let p02 = plane[r0 + x + 1];
            let p10 = plane[r1 + x - 1];
            let p12 = plane[r1 + x + 1];
            let p20 = plane[r2 + x - 1];
            let p21 = plane[r2 + x];
            let p22 = plane[r2 + x + 1];

            let gx = -3.0 * p00 + 3.0 * p02 - 10.0 * p10 + 10.0 * p12 - 3.0 * p20 + 3.0 * p22;
            let gy = -3.0 * p00 - 10.0 * p01 - 3.0 * p02 + 3.0 * p20 + 10.0 * p21 + 3.0 * p22;
            let mag = (gx.abs() + gy.abs()) / 32.0;

            row_out[x] = ((mag - edge_floor) / edge_span).clamp(0.0, 1.0);
        }
    });

    mask
}

/// Native Rust Motion-Compensated Temporal Filter (MCTF) and Auto-Degrainer.
/// Integrates ergonomic user strength & detail protection controls on top of Mutagen auto-tuning.
pub fn denoise_and_degrain_frame(
    neighbors: &[Option<&YuvFrame>; 5],
    profile: &GrainProfile,
    grain_retain_pct: f32,
    protect_lines: bool,
    denoise_strength: f32,
    detail_protect: f32,
    spatial_clamp: bool,
) -> YuvFrame {
    let curr = neighbors[2].expect("Center frame t must exist");
    let w = curr.width;
    let h = curr.height;
    let bs = 16usize;

    let y_spatial = if spatial_clamp && denoise_strength > 0.0 {
        spatial_impulse_clamp(&curr.y, w, h)
    } else {
        curr.y.clone()
    };

    if denoise_strength <= 0.001 {
        return YuvFrame {
            width: w,
            height: h,
            chroma_width: curr.chroma_width,
            chroma_height: curr.chroma_height,
            bit_depth: curr.bit_depth,
            subsampling: curr.subsampling,
            y: y_spatial,
            u: curr.u.clone(),
            v: curr.v.clone(),
        };
    }

    let mvs: Vec<Option<MotionField>> = [0usize, 1, 3, 4]
        .par_iter()
        .map(|&slot| {
            neighbors[slot].map(|ref_frame| estimate_motion_field(&y_spatial, &ref_frame.y, w, h, bs))
        })
        .collect();

    let edge_mask = if protect_lines && detail_protect > 0.0 {
        Some(compute_detail_protection_mask(&y_spatial, w, h, detail_protect))
    } else {
        None
    };

    // Scale thresholds by user-configurable --denoise-strength (1.0 = full automatic)
    let strength = denoise_strength.clamp(0.05, 3.0);
    let h_noise = ((profile.unwanted_noise_sigma * 2.4 + (profile.blockiness - 1.0).max(0.0) * 1.8)
        * strength)
        .clamp(0.8, 12.0);
    let h_total = (((profile.unwanted_noise_sigma + profile.avg_sigma) * 2.6) * strength)
        .clamp(h_noise, 24.0);

    let inv_2h_noise = 1.0 / (2.0 * h_noise * h_noise);
    let inv_2h_total = 1.0 / (2.0 * h_total * h_total);
    let retain_weight = (grain_retain_pct / 100.0).clamp(0.0, 1.0);

    let mut y_out = vec![0.0f32; w * h];
    let neighbor_slots = [0usize, 1, 3, 4];

    y_out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let by = (y / bs).min((h / bs).saturating_sub(1));
        let row_off = y * w;

        for x in 0..w {
            let c_val = y_spatial[row_off + x];
            let bx = (x / bs).min((w / bs).saturating_sub(1));

            let mut w_noise_sum = 1.0f32;
            let mut acc_noise = c_val;

            let mut w_total_sum = 1.0f32;
            let mut acc_total = c_val;

            for (mi, &slot) in neighbor_slots.iter().enumerate() {
                if let (Some(ref_frame), Some(ref mf)) = (neighbors[slot], &mvs[mi]) {
                    let mv_idx = by * mf.cols + bx;
                    if mv_idx < mf.vectors.len() {
                        let mv = mf.vectors[mv_idx];
                        if mv.confidence > 0.15 {
                            let sample = sample_bilinear(&ref_frame.y, w, h, (x as f32) + mv.dx, (y as f32) + mv.dy);
                            let diff = sample - c_val;
                            let d2 = diff * diff;
                            let temp_decay = if slot == 0 || slot == 4 { 0.70 } else { 1.0 };

                            let wn = mv.confidence * temp_decay * (-d2 * inv_2h_noise).exp();
                            w_noise_sum += wn;
                            acc_noise += wn * sample;

                            let wt = mv.confidence * temp_decay * (-d2 * inv_2h_total).exp();
                            w_total_sum += wt;
                            acc_total += wt * sample;
                        }
                    }
                }
            }

            let val_noise_clean = acc_noise / w_noise_sum;
            let val_fully_clean = acc_total / w_total_sum;
            let mut blended = val_fully_clean * (1.0 - retain_weight) + val_noise_clean * retain_weight;

            if let Some(ref mask) = edge_mask {
                let em = mask[row_off + x];
                blended = blended * (1.0 - em) + c_val * em;
            }

            row_out[x] = blended;
        }
    });

    let cw = curr.chroma_width;
    let ch = curr.chroma_height;
    let (scale_x, scale_y) = match curr.subsampling {
        ChromaSubsampling::Yuv420 => (0.5f32, 0.5f32),
        ChromaSubsampling::Yuv422 => (0.5f32, 1.0f32),
        ChromaSubsampling::Yuv444 => (1.0f32, 1.0f32),
    };

    let clean_chroma = |curr_c: &[f32], get_plane: fn(&YuvFrame) -> &[f32]| -> Vec<f32> {
        let mut c_out = vec![0.0f32; cw * ch];
        let h_chroma = (h_noise * 1.1).clamp(1.2, 8.0);
        let inv_2hc = 1.0 / (2.0 * h_chroma * h_chroma);

        c_out.par_chunks_mut(cw).enumerate().for_each(|(cy, row_out)| {
            let full_y = ((cy as f32) / scale_y) as usize;
            let by = (full_y / bs).min((h / bs).saturating_sub(1));
            let row_off = cy * cw;

            for cx in 0..cw {
                let c_val = curr_c[row_off + cx];
                let full_x = ((cx as f32) / scale_x) as usize;
                let bx = (full_x / bs).min((w / bs).saturating_sub(1));

                let mut w_sum = 1.0f32;
                let mut acc = c_val;

                for (mi, &slot) in neighbor_slots.iter().enumerate() {
                    if let (Some(ref_frame), Some(ref mf)) = (neighbors[slot], &mvs[mi]) {
                        let mv_idx = by * mf.cols + bx;
                        if mv_idx < mf.vectors.len() {
                            let mv = mf.vectors[mv_idx];
                            if mv.confidence > 0.20 {
                                let ref_c = get_plane(ref_frame);
                                let sample = sample_bilinear(
                                    ref_c,
                                    cw,
                                    ch,
                                    (cx as f32) + mv.dx * scale_x,
                                    (cy as f32) + mv.dy * scale_y,
                                );
                                let diff = sample - c_val;
                                let wt = mv.confidence * (-(diff * diff) * inv_2hc).exp();
                                w_sum += wt;
                                acc += wt * sample;
                            }
                        }
                    }
                }
                row_out[cx] = acc / w_sum;
            }
        });
        c_out
    };

    let (u_out, v_out) = rayon::join(
        || clean_chroma(&curr.u, |f| &f.u),
        || clean_chroma(&curr.v, |f| &f.v),
    );

    YuvFrame {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth: curr.bit_depth,
        subsampling: curr.subsampling,
        y: y_out,
        u: u_out,
        v: v_out,
    }
}
