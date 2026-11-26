use crate::motion::sample_bilinear;
use crate::types::{ColorRange, ContentType, GrainProfile, MotionField, TransferCurve};
use rayon::prelude::*;
use std::cmp::Ordering;

const BIN_CENTERS: [u8; 8] = [16, 48, 80, 112, 144, 176, 208, 240];

struct BlockSample {
    bin_center: u8,
    sigma_eff: f32,
    sigma_imm: f32,
    blockiness: f32,
    rho: f32,
    detrended: [f32; 256],
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

    pub fn analyze_frame(
        &self,
        luma: &[f32],
        prev_luma: Option<&[f32]>,
        motion: Option<&MotionField>,
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

        let block_results: (Vec<BlockSample>, Vec<f32>) = {
            let raw: Vec<(Option<BlockSample>, f32)> = (0..blocks_y)
                .into_par_iter()
                .flat_map(|by| {
                    let mut local = Vec::new();
                    for bx in (0..blocks_x).step_by(2) {
                        let y_start = by * block_size;
                        let x_start = bx * block_size;

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

                                let imm = (p00 - 2.0 * p01 + p02 - 2.0 * p10 + 4.0 * p11 - 2.0 * p12
                                    + p20 - 2.0 * p21 + p22)
                                    .abs();
                                imm_abs.push(imm);
                            }
                        }

                        let avg_scharr = scharr_acc / (14.0 * 14.0);
                        if avg_scharr > max_edge || scharr_max > max_edge * 4.5 {
                            local.push((None, avg_scharr));
                            continue;
                        }

                        imm_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let sigma_imm = (imm_abs[imm_abs.len() / 2] * 1.4826) / 6.0;

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

                        let blockiness = (bound_sum / 32.0) / (inner_sum / inner_count.max(1.0)).max(0.25);

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
                                let r = luma[row + x] - (mean + slope_x * dx + slope_y * dy);
                                detrended[y * block_size + x] = r;
                                abs_detrended.push(r.abs());
                                sq_diff += r * r;
                            }
                        }

                        abs_detrended.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let sigma_detrend = abs_detrended[128] * 1.4826;

                        let mut sigma_eff = sigma_detrend;
                        if let (Some(prev), Some(mf)) = (prev_luma, motion) {
                            let mv_idx = by * mf.cols + bx;
                            if mv_idx < mf.vectors.len() {
                                let mv = mf.vectors[mv_idx];
                                if mv.confidence > 0.65 {
                                    let mut temp_abs = Vec::with_capacity(256);
                                    for y in 0..block_size {
                                        let cy = y_start + y;
                                        let row = cy * width + x_start;
                                        for x in 0..block_size {
                                            let cx = x_start + x;
                                            let pred = sample_bilinear(
                                                prev,
                                                width,
                                                height,
                                                (cx as f32) + mv.dx,
                                                (cy as f32) + mv.dy,
                                            );
                                            temp_abs.push((luma[row + x] - pred).abs());
                                        }
                                    }
                                    temp_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                                    let sigma_temp = (temp_abs[128] * 1.4826) / std::f32::consts::SQRT_2;
                                    sigma_eff = sigma_detrend.min(sigma_temp * 1.15);
                                }
                            }
                        }

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

                        local.push((
                            Some(BlockSample {
                                bin_center: nearest_bin,
                                sigma_eff,
                                sigma_imm,
                                blockiness,
                                rho,
                                detrended,
                            }),
                            avg_scharr,
                        ));
                    }
                    local
                })
                .collect();

            let mut samples = Vec::new();
            let mut textures = Vec::new();
            for (opt_s, scharr) in raw {
                textures.push(scharr);
                if let Some(s) = opt_s {
                    samples.push(s);
                }
            }
            (samples, textures)
        };

        let texture_complexity = if !block_results.1.is_empty() {
            block_results.1.iter().sum::<f32>() / (block_results.1.len() as f32)
        } else {
            1.0
        };

        let mut bin_samples: [Vec<&BlockSample>; 8] = Default::default();
        let mut all_blockiness = Vec::new();
        let mut all_unwanted = Vec::new();
        let mut ar_valid_blocks: Vec<&[f32; 256]> = Vec::new();

        for s in &block_results.0 {
            all_blockiness.push(s.blockiness);
            let hf_ratio = s.sigma_imm / s.sigma_eff.max(0.05);
            let is_blocky = s.blockiness > 1.75;

            if is_blocky || hf_ratio > 0.85 {
                all_unwanted.push(s.sigma_imm);
            } else {
                all_unwanted.push(s.sigma_imm * 0.35);
            }

            if self.no_noise_bias && is_blocky {
                continue;
            }

            if s.sigma_eff >= 0.95 {
                ar_valid_blocks.push(&s.detrended);
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

            let mut eff_vals: Vec<f32> = samples.iter().map(|s| s.sigma_eff).collect();
            eff_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let med_eff = eff_vals[eff_vals.len() / 2];

            let h_param = (med_eff * 0.5).max(0.5);
            let mut w_sum = 0.0f32;
            let mut val_sum = 0.0f32;

            for s in samples {
                let diff = s.sigma_eff - med_eff;
                let w = (-(diff * diff) / (2.0 * h_param * h_param)).exp();
                w_sum += w;
                val_sum += w * s.sigma_eff;
                if s.rho > 0.0 {
                    rhos.push(s.rho);
                }
                hf_ratios.push(s.sigma_imm / s.sigma_eff.max(0.05));
            }

            let nlm_sigma = if w_sum > 0.0 { val_sum / w_sum } else { med_eff };
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

        let ar_coeffs_y = if lag == 0 {
            vec![]
        } else if ar_valid_blocks.len() >= 6 && authentic_grain_sigma >= 1.15 {
            solve_ar_yule_walker(&ar_valid_blocks, lag)
        } else {
            default_70mm_ar_coeffs(lag)
        };

        GrainProfile {
            lag,
            ar_coeffs_y,
            profile_sy,
            avg_sigma: authentic_grain_sigma,
            unwanted_noise_sigma,
            blockiness: avg_blockiness,
            mean_luma: frame_mean_luma,
            texture_complexity,
            hf_ratio: avg_hf_ratio,
        }
    }
}

pub fn default_70mm_ar_coeffs(lag: u8) -> Vec<i32> {
    match lag {
        3 => vec![2, 1, -1, -3, 1, 4, 6, 4, 1, -2, 2, 8, 14, 8, 2, -2, 1, 4, 6, 4, 1, -3, -1, 1],
        2 => vec![3, 5, 3, -2, 6, 16, 6, -2, 3, 5, 3, -1],
        1 => vec![8, 20, 8, 4],
        _ => vec![],
    }
}

pub fn solve_ar_yule_walker(blocks: &[&[f32; 256]], lag: u8) -> Vec<i32> {
    let l = lag as isize;
    let mut offsets: Vec<(isize, isize)> = Vec::new();
    for dy in -l..=0 {
        for dx in -l..=l {
            if dy == 0 && dx >= 0 {
                break;
            }
            offsets.push((dx, dy));
        }
    }

    let n = offsets.len();
    if n == 0 {
        return vec![];
    }

    let max_shift = (2 * l) as usize;
    let span = 2 * max_shift + 1;
    let mut cov_sum = vec![0.0f64; span * span];
    let mut cov_cnt = vec![0u64; span * span];

    for blk in blocks {
        for y in 0..16isize {
            for x in 0..16isize {
                let c = blk[(y * 16 + x) as usize] as f64;
                for dy in -(2 * l)..=(2 * l) {
                    let ny = y + dy;
                    if !(0..16).contains(&ny) {
                        continue;
                    }
                    for dx in -(2 * l)..=(2 * l) {
                        let nx = x + dx;
                        if !(0..16).contains(&nx) {
                            continue;
                        }
                        let idx = ((dy + 2 * l) as usize) * span + ((dx + 2 * l) as usize);
                        cov_sum[idx] += c * (blk[(ny * 16 + nx) as usize] as f64);
                        cov_cnt[idx] += 1;
                    }
                }
            }
        }
    }

    let gamma = |dx: isize, dy: isize| -> f64 {
        let idx = ((dy + 2 * l) as usize) * span + ((dx + 2 * l) as usize);
        if cov_cnt[idx] > 0 {
            cov_sum[idx] / (cov_cnt[idx] as f64)
        } else {
            0.0
        }
    };

    let var0 = gamma(0, 0).max(1e-6);

    let mut mat_r = vec![0.0f64; n * n];
    let mut vec_r = vec![0.0f64; n];

    for i in 0..n {
        let (xi, yi) = offsets[i];
        vec_r[i] = gamma(xi, yi);
        for j in 0..n {
            let (xj, yj) = offsets[j];
            mat_r[i * n + j] = gamma(xi - xj, yi - yj);
        }
        mat_r[i * n + i] += 0.025 * var0;
    }

    let mut chol = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat_r[i * n + j];
            for k in 0..j {
                sum -= chol[i * n + k] * chol[j * n + k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return default_70mm_ar_coeffs(lag);
                }
                chol[i * n + j] = sum.sqrt();
            } else {
                chol[i * n + j] = sum / chol[j * n + j];
            }
        }
    }

    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = vec_r[i];
        for k in 0..i {
            sum -= chol[i * n + k] * y[k];
        }
        y[i] = sum / chol[i * n + i];
    }

    let mut a = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= chol[k * n + i] * a[k];
        }
        a[i] = sum / chol[i * n + i];
    }

    let l1_norm: f64 = a.iter().map(|v| v.abs()).sum();
    let damp = if l1_norm > 0.85 { 0.85 / l1_norm } else { 1.0 };

    a.into_iter()
        .map(|v| (v * damp * 64.0).round().clamp(-128.0, 127.0) as i32)
        .collect()
}

pub fn fallback_analyze_frame(luma: &[f32], width: usize, height: usize) -> GrainProfile {
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
        ar_coeffs_y: default_70mm_ar_coeffs(2),
        profile_sy,
        avg_sigma,
        unwanted_noise_sigma: avg_sigma * 0.5,
        blockiness: 1.0,
        mean_luma: frame_mean_luma,
        texture_complexity: 1.5,
        hf_ratio: 0.60,
    }
}

pub fn classify_frame_auto(luma: &[f32], width: usize, height: usize) -> ContentType {
    let block_size = 16;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    let stats: Vec<(usize, usize, usize)> = (0..blocks_y)
        .into_par_iter()
        .map(|by| {
            let mut active = 0usize;
            let mut cel_flat = 0usize;
            let mut cgi_smooth = 0usize;

            for bx in 0..blocks_x {
                let y_start = by * block_size;
                let x_start = bx * block_size;
                let mut sum = 0.0f32;
                let mut sq_sum = 0.0f32;
                let mut identical_pairs = 0usize;

                for y in 0..block_size {
                    let row = (y_start + y) * width + x_start;
                    for x in 0..block_size {
                        let p = luma[row + x];
                        sum += p;
                        sq_sum += p * p;
                        if x + 1 < block_size && (p - luma[row + x + 1]).abs() < 0.15 {
                            identical_pairs += 1;
                        }
                    }
                }

                let mean = sum / 256.0;
                if mean <= 24.0 || mean >= 232.0 {
                    continue;
                }

                active += 1;
                let var = ((sq_sum / 256.0) - (mean * mean)).max(0.0);
                let zero_diff_ratio = (identical_pairs as f32) / 240.0;

                if var < 1.6 && zero_diff_ratio > 0.55 {
                    cel_flat += 1;
                } else if var < 8.5 && zero_diff_ratio > 0.42 {
                    cgi_smooth += 1;
                }
            }
            (active, cel_flat, cgi_smooth)
        })
        .collect();

    let total_active: usize = stats.iter().map(|s| s.0).sum();
    if total_active < 16 {
        return ContentType::LiveAction;
    }

    let total_cel: usize = stats.iter().map(|s| s.1).sum();
    let total_cgi: usize = stats.iter().map(|s| s.2).sum();

    let cel_ratio = (total_cel as f32) / (total_active as f32);
    let smooth_ratio = ((total_cel + total_cgi) as f32) / (total_active as f32);

    if cel_ratio > 0.34 {
        ContentType::TwoD
    } else if smooth_ratio > 0.48 && cel_ratio > 0.18 {
        ContentType::ThreeD
    } else {
        ContentType::LiveAction
    }
}

pub fn eotf_to_linear(v: f32, transfer: TransferCurve) -> f32 {
    let vc = v.clamp(0.0, 1.0);
    match transfer {
        TransferCurve::Pq => {
            let m1 = 0.1593017578125f32;
            let m2 = 78.84375f32;
            let c1 = 0.8359375f32;
            let c2 = 18.8515625f32;
            let c3 = 18.6875f32;
            let vp = vc.powf(1.0 / m2);
            let num = (vp - c1).max(0.0);
            let den = (c2 - c3 * vp).max(1e-6);
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

pub fn generate_iso_photon_profile(
    iso: u32,
    width: usize,
    height: usize,
    transfer: TransferCurve,
    color_range: ColorRange,
    pure_digital: bool,
) -> (u8, Vec<i32>, Vec<(u8, u8)>) {
    let luma_pts: [u8; 14] = if color_range == ColorRange::Full {
        [0, 12, 32, 54, 76, 98, 120, 142, 164, 186, 208, 230, 250, 255]
    } else {
        [16, 19, 38, 58, 77, 96, 116, 135, 155, 174, 193, 213, 232, 235]
    };

    let (min_l, max_l) = if color_range == ColorRange::Full {
        (0.0f32, 255.0f32)
    } else {
        (16.0f32, 235.0f32)
    };

    let iso_scale = ((iso as f32) / 200.0).sqrt();
    let res_scale = ((1920.0 * 1080.0) / ((width * height).max(1) as f32))
        .powf(0.15)
        .clamp(0.80, 1.30);

    let mut sy_points = Vec::with_capacity(14);
    for (idx, &luma) in luma_pts.iter().enumerate() {
        if idx == 0 || idx == luma_pts.len() - 1 {
            sy_points.push((luma, 1));
            continue;
        }

        let norm_v = (((luma as f32) - min_l) / (max_l - min_l)).clamp(0.0, 1.0);
        let lin_light = eotf_to_linear(norm_v, transfer);
        let toe_boost = if norm_v <= 0.04 { 0.85 } else { 0.0 };

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
        0
    } else if height >= 2160 {
        3
    } else {
        2
    };

    (lag, default_70mm_ar_coeffs(lag), sy_points)
}

pub fn get_subtle_70mm_profile(
    ctype: ContentType,
    scene_mean_luma: f32,
    color_range: ColorRange,
) -> (u8, Vec<i32>, Vec<(u8, u8)>) {
    let lag = match ctype {
        ContentType::TwoD => 2,
        ContentType::ThreeD => 2,
        ContentType::LiveAction => 3,
    };

    let luma_weight = (scene_mean_luma / 128.0).clamp(0.6, 1.4);

    let mut raw_sy: Vec<(u8, u8)> = match ctype {
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

    if color_range == ColorRange::Full {
        raw_sy[0].0 = 0;
        let last = raw_sy.len() - 1;
        raw_sy[last].0 = 255;
    }

    let cleaned_sy = raw_sy.into_iter().map(|(l, v)| (l, v.clamp(1, 255))).collect();
    (lag, default_70mm_ar_coeffs(lag), cleaned_sy)
}
