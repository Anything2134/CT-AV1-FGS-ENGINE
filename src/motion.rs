use crate::types::{MotionField, MotionVector};
use rayon::prelude::*;

/// Computes 4x4 Walsh-Hadamard Sum of Absolute Transformed Differences (SATD).
/// Classical public-domain frequency-domain residual metric for motion & intra cost.
#[inline]
pub fn satd_4x4(a: &[f32], a_stride: usize, b: &[f32], b_stride: usize) -> f32 {
    let mut m = [[0.0f32; 4]; 4];
    for r in 0..4 {
        let ra = r * a_stride;
        let rb = r * b_stride;
        let d0 = a[ra] - b[rb];
        let d1 = a[ra + 1] - b[rb + 1];
        let d2 = a[ra + 2] - b[rb + 2];
        let d3 = a[ra + 3] - b[rb + 3];

        let t0 = d0 + d1;
        let t1 = d0 - d1;
        let t2 = d2 + d3;
        let t3 = d2 - d3;

        m[r][0] = t0 + t2;
        m[r][1] = t1 + t3;
        m[r][2] = t0 - t2;
        m[r][3] = t1 - t3;
    }

    let mut satd = 0.0f32;
    for c in 0..4 {
        let t0 = m[0][c] + m[1][c];
        let t1 = m[0][c] - m[1][c];
        let t2 = m[2][c] + m[3][c];
        let t3 = m[2][c] - m[3][c];

        satd += (t0 + t2).abs() + (t1 + t3).abs() + (t0 - t2).abs() + (t1 - t3).abs();
    }

    satd * 0.25
}

/// Computes 8x8 SATD by summing four 4x4 Hadamard transforms.
#[inline]
pub fn satd_8x8(a: &[f32], a_stride: usize, b: &[f32], b_stride: usize) -> f32 {
    satd_4x4(a, a_stride, b, b_stride)
        + satd_4x4(&a[4..], a_stride, &b[4..], b_stride)
        + satd_4x4(&a[4 * a_stride..], a_stride, &b[4 * b_stride..], b_stride)
        + satd_4x4(&a[4 * a_stride + 4..], a_stride, &b[4 * b_stride + 4..], b_stride)
}

/// Computes Sum of Absolute Differences (SAD) for an arbitrary block size inside bounded coordinates.
#[inline]
pub fn sad_block(
    curr: &[f32],
    prev: &[f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    dx: isize,
    dy: isize,
    block_size: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for y in 0..block_size {
        let cy = by + y;
        let py = (cy as isize + dy).clamp(0, (height - 1) as isize) as usize;
        let crow = cy * width;
        let prow = py * width;
        for x in 0..block_size {
            let cx = bx + x;
            let px = (cx as isize + dx).clamp(0, (width - 1) as isize) as usize;
            sum += (curr[crow + cx] - prev[prow + px]).abs();
        }
    }
    sum
}

/// Evaluates bilinear-interpolated pixel sample at floating-point coordinates.
#[inline]
pub fn sample_bilinear(plane: &[f32], width: usize, height: usize, fx: f32, fy: f32) -> f32 {
    let cx = fx.clamp(0.0, (width - 1) as f32);
    let cy = fy.clamp(0.0, (height - 1) as f32);

    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let wx = cx - (x0 as f32);
    let wy = cy - (y0 as f32);

    let p00 = plane[y0 * width + x0];
    let p10 = plane[y0 * width + x1];
    let p01 = plane[y1 * width + x0];
    let p11 = plane[y1 * width + x1];

    let top = p00 + (p10 - p00) * wx;
    let bot = p01 + (p11 - p01) * wx;
    top + (bot - top) * wy
}

/// Computes a 2x downsampled plane for fast hierarchical motion search.
pub fn downsample_2x(plane: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let dw = (width / 2).max(1);
    let dh = (height / 2).max(1);
    let mut out = vec![0.0f32; dw * dh];

    out.par_chunks_mut(dw).enumerate().for_each(|(dy, row)| {
        let sy = (dy * 2).min(height - 2);
        for dx in 0..dw {
            let sx = (dx * 2).min(width - 2);
            row[dx] = 0.25
                * (plane[sy * width + sx]
                    + plane[sy * width + sx + 1]
                    + plane[(sy + 1) * width + sx]
                    + plane[(sy + 1) * width + sx + 1]);
        }
    });

    (out, dw, dh)
}

/// Estimates hierarchical coarse-to-fine Diamond Search motion vectors with half-pixel refinement.
pub fn estimate_motion_field(
    curr: &[f32],
    prev: &[f32],
    width: usize,
    height: usize,
    block_size: usize,
) -> MotionField {
    let (curr_ds, dw, dh) = downsample_2x(curr, width, height);
    let (prev_ds, _, _) = downsample_2x(prev, width, height);

    let cols = width / block_size;
    let rows = height / block_size;
    let ds_block = (block_size / 2).max(4);

    let vectors: Vec<MotionVector> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_vecs = Vec::with_capacity(cols);
            let by = row * block_size;

            for col in 0..cols {
                let bx = col * block_size;

                // Stage 1: Coarse Diamond Search on 2x downsampled image (up to +-32 full-res range)
                let dbx = (bx / 2).min(dw.saturating_sub(ds_block));
                let dby = (by / 2).min(dh.saturating_sub(ds_block));

                let mut cdx: isize = 0;
                let mut cdy: isize = 0;
                let mut best_ds_sad = sad_block(&curr_ds, &prev_ds, dw, dh, dbx, dby, 0, 0, ds_block);

                for step in [8isize, 4, 2, 1] {
                    let candidates = [
                        (cdx + step, cdy),
                        (cdx - step, cdy),
                        (cdx, cdy + step),
                        (cdx, cdy - step),
                    ];
                    let mut improved = false;
                    let (mut next_x, mut next_y) = (cdx, cdy);

                    for (tx, ty) in candidates {
                        if tx.abs() > 16 || ty.abs() > 16 {
                            continue;
                        }
                        let s = sad_block(&curr_ds, &prev_ds, dw, dh, dbx, dby, tx, ty, ds_block);
                        if s < best_ds_sad {
                            best_ds_sad = s;
                            next_x = tx;
                            next_y = ty;
                            improved = true;
                        }
                    }
                    if improved {
                        cdx = next_x;
                        cdy = next_y;
                    }
                }

                // Stage 2: Fine full-resolution Diamond Search comparing coarse projection vs zero-MV
                let mut fdx = cdx * 2;
                let mut fdy = cdy * 2;
                let mut best_sad = sad_block(curr, prev, width, height, bx, by, fdx, fdy, block_size);

                let zero_sad = sad_block(curr, prev, width, height, bx, by, 0, 0, block_size);
                if zero_sad <= best_sad {
                    fdx = 0;
                    fdy = 0;
                    best_sad = zero_sad;
                }

                for step in [2isize, 1] {
                    let candidates = [
                        (fdx + step, fdy),
                        (fdx - step, fdy),
                        (fdx, fdy + step),
                        (fdx, fdy - step),
                    ];
                    let (mut next_x, mut next_y) = (fdx, fdy);
                    for (tx, ty) in candidates {
                        if tx.abs() > 36 || ty.abs() > 36 {
                            continue;
                        }
                        let s = sad_block(curr, prev, width, height, bx, by, tx, ty, block_size);
                        if s < best_sad {
                            best_sad = s;
                            next_x = tx;
                            next_y = ty;
                        }
                    }
                    fdx = next_x;
                    fdy = next_y;
                }

                // Stage 3: Half-pixel bilinear refinement on best integer candidate
                let mut best_fx = fdx as f32;
                let mut best_fy = fdy as f32;
                let mut best_sub_sad = best_sad;

                for &(ox, oy) in &[
                    (-0.5f32, 0.0f32),
                    (0.5, 0.0),
                    (0.0, -0.5),
                    (0.0, 0.5),
                ] {
                    let tx = (fdx as f32) + ox;
                    let ty = (fdy as f32) + oy;
                    let mut sub_sad = 0.0f32;
                    for y in 0..block_size {
                        let cy = by + y;
                        let crow = cy * width;
                        for x in 0..block_size {
                            let cx = bx + x;
                            let pred = sample_bilinear(prev, width, height, (cx as f32) + tx, (cy as f32) + ty);
                            sub_sad += (curr[crow + cx] - pred).abs();
                        }
                    }
                    if sub_sad < best_sub_sad {
                        best_sub_sad = sub_sad;
                        best_fx = tx;
                        best_fy = ty;
                    }
                }

                // Stage 4: Compute Hadamard SATD and block match confidence
                let mut pred_patch = [0.0f32; 256];
                let mut curr_patch = [0.0f32; 256];
                let bs_clamped = block_size.min(16);

                for y in 0..bs_clamped {
                    let cy = by + y;
                    let crow = cy * width;
                    for x in 0..bs_clamped {
                        let cx = bx + x;
                        curr_patch[y * bs_clamped + x] = curr[crow + cx];
                        pred_patch[y * bs_clamped + x] =
                            sample_bilinear(prev, width, height, (cx as f32) + best_fx, (cy as f32) + best_fy);
                    }
                }

                let mut total_satd = 0.0f32;
                let sub_blocks = bs_clamped / 8;
                for sy in 0..sub_blocks {
                    for sx in 0..sub_blocks {
                        let off = sy * 8 * bs_clamped + sx * 8;
                        total_satd += satd_8x8(&curr_patch[off..], bs_clamped, &pred_patch[off..], bs_clamped);
                    }
                }

                let mean_sad = best_sub_sad / ((block_size * block_size) as f32);
                let confidence = (1.0 - (mean_sad / 18.0)).clamp(0.0, 1.0);

                row_vecs.push(MotionVector {
                    dx: best_fx,
                    dy: best_fy,
                    sad: mean_sad,
                    satd: total_satd / ((block_size * block_size) as f32),
                    confidence,
                });
            }
            row_vecs
        })
        .collect();

    MotionField {
        cols,
        rows,
        block_size,
        vectors,
    }
}
