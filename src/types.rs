use clap::ValueEnum;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum ContentType {
    #[value(name = "2d")]
    TwoD,
    #[value(name = "3d")]
    ThreeD,
    #[value(name = "live_action", alias = "live-action")]
    LiveAction,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum MutagenState {
    #[value(name = "on")]
    On,
    #[value(name = "off")]
    Off,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum TransferCurve {
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
pub enum FgsBalanceMode {
    #[value(name = "complement")]
    Complement,
    #[value(name = "linear")]
    Linear,
    #[value(name = "full")]
    Full,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum TimeUnitMode {
    #[value(name = "100ns")]
    Ns100,
    #[value(name = "frames")]
    Frames,
    #[value(name = "global")]
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChromaSubsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

#[derive(Clone, Debug)]
pub struct YuvFrame {
    pub width: usize,
    pub height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub bit_depth: usize,
    pub subsampling: ChromaSubsampling,
    pub y: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SceneSegment {
    pub start_frame: usize,
    pub end_frame: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MotionVector {
    pub dx: f32,
    pub dy: f32,
    pub sad: f32,
    pub satd: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct MotionField {
    pub cols: usize,
    pub rows: usize,
    pub block_size: usize,
    pub vectors: Vec<MotionVector>,
}

#[derive(Clone, Debug)]
pub struct GrainProfile {
    pub lag: u8,
    pub ar_coeffs_y: Vec<i32>,
    pub profile_sy: Vec<(u8, u8)>,
    pub avg_sigma: f32,
    pub unwanted_noise_sigma: f32,
    pub blockiness: f32,
    pub mean_luma: f32,
}

#[derive(Clone, Debug)]
pub struct FgsTableEntry {
    pub start_frame: usize,
    pub end_frame: usize,
    pub lag: u8,
    pub ar_coeffs_y: Vec<i32>,
    pub points_sy: Vec<(u8, u8)>,
    pub scale: f32,
    pub seed: u16,
}

#[derive(Clone, Debug)]
pub struct VsSceneTelemetry {
    pub start_frame: usize,
    pub end_frame: usize,
    pub unwanted_noise_sigma: f32,
    pub authentic_grain_sigma: f32,
    pub blockiness: f32,
    pub mean_luma: f32,
    pub denoise_thsad: u32,
    pub degrain_thsad: u32,
}
