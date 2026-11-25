use crate::types::{ChromaSubsampling, TransferCurve, YuvFrame};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;

pub struct Y4mStream {
    child: Option<Child>,
    _stderr_thread: Option<JoinHandle<String>>,
    reader: BufReader<Box<dyn Read + Send>>,
    pub width: usize,
    pub height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub current_frame: usize,
    pub bit_depth: usize,
    pub bytes_per_sample: usize,
    pub subsampling: ChromaSubsampling,
    pub fps_num: u64,
    pub fps_den: u64,
    pub raw_header_line: String,
    pub detected_transfer: TransferCurve,
    luma_bytes: usize,
    chroma_plane_bytes: usize,
}

impl Drop for Y4mStream {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Y4mStream {
    /// Opens a Y4M pipe from VapourSynth (.vpy), FFmpeg (media containers), or raw .y4m files.
    pub fn from_input(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("File does not exist at path: '{}'.", path.display()));
        }

        let is_vpy = path.extension().and_then(|s| s.to_str()) == Some("vpy");
        let is_y4m = path.extension().and_then(|s| s.to_str()) == Some("y4m");

        let (mut child_opt, mut stderr_thread, stdout): (
            Option<Child>,
            Option<JoinHandle<String>>,
            Box<dyn Read + Send>,
        ) = if is_y4m {
            let file = File::open(path).map_err(|e| format!("Failed to open Y4M file: {e}"))?;
            (None, None, Box::new(file))
        } else {
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
                // Convert only unsupported formats (e.g. yuvj420p, nv12, rgb24) while passing
                // through native 8/10/12/16-bit planar YUV formats untouched.
                Command::new("ffmpeg")
                    .arg("-nostdin")
                    .arg("-i")
                    .arg(path)
                    .arg("-map")
                    .arg("0:v:0")
                    .arg("-vf")
                    .arg("format=yuv420p|yuv422p|yuv444p|yuv420p10le|yuv422p10le|yuv444p10le|yuv420p12le|yuv422p12le|yuv444p12le|yuv420p16le|yuv422p16le|yuv444p16le")
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

            let out = child.stdout.take().ok_or("Failed to capture child stdout")?;
            let err_handle = child.stderr.take().map(|mut err| {
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    while let Ok(n) = err.read(&mut chunk) {
                        if n == 0 {
                            break;
                        }
                        if buf.len() < 32768 {
                            let take_n = n.min(32768 - buf.len());
                            buf.extend_from_slice(&chunk[..take_n]);
                        }
                    }
                    String::from_utf8_lossy(&buf).into_owned()
                })
            });

            (Some(child), err_handle, Box::new(out))
        };

        let mut reader = BufReader::with_capacity(2 * 1024 * 1024, stdout);

        let mut header = String::new();
        let read_res = reader.read_line(&mut header);

        if read_res.is_err() || !header.starts_with("YUV4MPEG2") {
            if let Some(mut c) = child_opt.take() {
                let _ = c.wait();
            }
            let stderr_msg = stderr_thread
                .take()
                .and_then(|h| h.join().ok())
                .unwrap_or_default();

            return Err(format!(
                "Invalid Y4M header received: '{}'.\nProcess stderr:\n{}",
                header.trim(),
                if stderr_msg.trim().is_empty() {
                    "No stderr output (verify input path and ffmpeg/vspipe installation)."
                } else {
                    stderr_msg.trim()
                }
            ));
        }

        let mut width = 0usize;
        let mut height = 0usize;
        let mut bit_depth = 8usize;
        let mut subsampling = ChromaSubsampling::Yuv420;
        let mut fps_num = 24000u64;
        let mut fps_den = 1001u64;
        let mut detected_transfer = TransferCurve::Sdr;

        for token in header.split_whitespace() {
            if let Some(w) = token.strip_prefix('W') {
                width = w.parse().unwrap_or(0);
            } else if let Some(h) = token.strip_prefix('H') {
                height = h.parse().unwrap_or(0);
            } else if let Some(f) = token.strip_prefix('F') {
                let parts: Vec<&str> = f.split(':').collect();
                if parts.len() == 2 {
                    let n = parts[0].parse::<u64>().unwrap_or(24000);
                    let d = parts[1].parse::<u64>().unwrap_or(1001);
                    if n > 0 && d > 0 {
                        fps_num = n;
                        fps_den = d;
                    }
                }
            } else if let Some(c) = token.strip_prefix('C') {
                if c.contains("444") {
                    subsampling = ChromaSubsampling::Yuv444;
                } else if c.contains("422") {
                    subsampling = ChromaSubsampling::Yuv422;
                } else {
                    subsampling = ChromaSubsampling::Yuv420;
                }

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
            return Err("Video dimensions missing from Y4M header".to_string());
        }

        let (chroma_width, chroma_height) = match subsampling {
            ChromaSubsampling::Yuv420 => (width / 2, height / 2),
            ChromaSubsampling::Yuv422 => (width / 2, height),
            ChromaSubsampling::Yuv444 => (width, height),
        };

        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        let luma_bytes = width * height * bytes_per_sample;
        let chroma_plane_bytes = chroma_width * chroma_height * bytes_per_sample;

        Ok(Self {
            child: child_opt,
            _stderr_thread: stderr_thread,
            reader,
            width,
            height,
            chroma_width,
            chroma_height,
            current_frame: 0,
            bit_depth,
            bytes_per_sample,
            subsampling,
            fps_num,
            fps_den,
            raw_header_line: header,
            detected_transfer,
            luma_bytes,
            chroma_plane_bytes,
        })
    }

    fn read_frame_header(&mut self) -> Result<(), ()> {
        let mut line = Vec::new();
        match self.reader.read_until(b'\n', &mut line) {
            Ok(n) if n > 0 && line.starts_with(b"FRAME") => Ok(()),
            _ => Err(()),
        }
    }

    fn decode_plane(&self, raw: &[u8]) -> Vec<f32> {
        if self.bytes_per_sample == 1 {
            raw.iter().map(|&p| p as f32).collect()
        } else {
            let max_val = ((1u32 << self.bit_depth) - 1) as f32;
            raw.chunks_exact(2)
                .map(|c| (u16::from_le_bytes([c[0], c[1]]) as f32 / max_val) * 255.0)
                .collect()
        }
    }

    pub fn read_next_luma_only(&mut self) -> Option<Vec<f32>> {
        if self.read_frame_header().is_err() {
            return None;
        }
        let mut raw_y = vec![0u8; self.luma_bytes];
        if self.reader.read_exact(&mut raw_y).is_err() {
            return None;
        }
        if skip_exact_bytes(&mut self.reader, self.chroma_plane_bytes * 2).is_err() {
            return None;
        }
        self.current_frame += 1;
        Some(self.decode_plane(&raw_y))
    }

    pub fn read_next_yuv_frame(&mut self) -> Option<YuvFrame> {
        if self.read_frame_header().is_err() {
            return None;
        }

        let mut raw_y = vec![0u8; self.luma_bytes];
        let mut raw_u = vec![0u8; self.chroma_plane_bytes];
        let mut raw_v = vec![0u8; self.chroma_plane_bytes];

        if self.reader.read_exact(&mut raw_y).is_err()
            || self.reader.read_exact(&mut raw_u).is_err()
            || self.reader.read_exact(&mut raw_v).is_err()
        {
            return None;
        }

        self.current_frame += 1;

        Some(YuvFrame {
            width: self.width,
            height: self.height,
            chroma_width: self.chroma_width,
            chroma_height: self.chroma_height,
            bit_depth: self.bit_depth,
            subsampling: self.subsampling,
            y: self.decode_plane(&raw_y),
            u: self.decode_plane(&raw_u),
            v: self.decode_plane(&raw_v),
        })
    }

    pub fn skip_frames(&mut self, count: usize) -> bool {
        let payload = self.luma_bytes + self.chroma_plane_bytes * 2;
        for _ in 0..count {
            if self.read_frame_header().is_err() {
                return false;
            }
            if skip_exact_bytes(&mut self.reader, payload).is_err() {
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

pub struct Y4mWriter {
    writer: BufWriter<Box<dyn Write>>,
    bit_depth: usize,
    bytes_per_sample: usize,
}

impl Y4mWriter {
    pub fn create(target: &Path, header_line: &str, bit_depth: usize) -> std::io::Result<Self> {
        let dest: Box<dyn Write> = if target.as_os_str() == "-" {
            Box::new(std::io::stdout())
        } else {
            Box::new(File::create(target)?)
        };

        let mut writer = BufWriter::with_capacity(2 * 1024 * 1024, dest);
        writer.write_all(header_line.as_bytes())?;
        if !header_line.ends_with('\n') {
            writer.write_all(b"\n")?;
        }

        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        Ok(Self {
            writer,
            bit_depth,
            bytes_per_sample,
        })
    }

    fn encode_plane(&self, plane: &[f32], out_buf: &mut Vec<u8>) {
        out_buf.clear();
        if self.bytes_per_sample == 1 {
            for &p in plane {
                out_buf.push(p.round().clamp(0.0, 255.0) as u8);
            }
        } else {
            let max_val = ((1u32 << self.bit_depth) - 1) as f32;
            let scale = max_val / 255.0;
            for &p in plane {
                let q = (p * scale).round().clamp(0.0, max_val) as u16;
                out_buf.extend_from_slice(&q.to_le_bytes());
            }
        }
    }

    pub fn write_frame(&mut self, frame: &YuvFrame) -> std::io::Result<()> {
        self.writer.write_all(b"FRAME\n")?;
        let mut buf = Vec::with_capacity(frame.y.len() * self.bytes_per_sample);

        self.encode_plane(&frame.y, &mut buf);
        self.writer.write_all(&buf)?;

        self.encode_plane(&frame.u, &mut buf);
        self.writer.write_all(&buf)?;

        self.encode_plane(&frame.v, &mut buf);
        self.writer.write_all(&buf)?;

        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
