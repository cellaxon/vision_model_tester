use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn};
use ort::execution_providers::{CPUExecutionProviderOptions, CoreMLExecutionProviderOptions};
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::error::{AppError, AppResult};
use crate::utils::{color_utils, fs_utils, math_utils};

// 기존 상수들을 설정에서 가져오기


// 상수 정의 (기존 호환성을 위해 유지)
const MODEL_INPUT_SIZE: u32 = 640;
const CONFIDENCE_THRESHOLD: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.2;
const BBOX_COLOR: Rgb<u8> = Rgb([255, 0, 0]);
// const BASE_PARSE_CONFIDENCE: f32 = 0.05; // 미사용

// 임베디드 리소스: assets/models 폴더의 모든 onnx 파일을 임베딩 (분리된 모듈)
// embedded_models 모듈은 파일 하단에 정의됨

pub fn get_model_info(selected_file_name: &str) -> (String, u32) {
    // 지원 파일: gelan-c.onnx, gelan-e.onnx, yolov9-c.onnx, yolov9-e.onnx
    let lower = selected_file_name.to_ascii_lowercase();
    let model_name = if lower.contains("gelan-e") {
        "YOLOv9-GELAN-E"
    } else if lower.contains("gelan-c") {
        "YOLOv9-GELAN-C"
    } else if lower.contains("yolov9-e") {
        "YOLOv9-E"
    } else if lower.contains("yolov9-c") {
        "YOLOv9-C"
    } else {
        "YOLOv9-Unknown"
    };
    (model_name.to_string(), MODEL_INPUT_SIZE)
}

/// 객체 검출 결과를 나타내는 구조체
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: [f32; 4], // [x1, y1, x2, y2] in normalized coordinates (0-1)
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
}

// 공통 DetectionResult 타입 사용

/// YOLOv9 COCO 클래스 ID를 클래스 이름으로 변환
fn yolov9_id_to_label(class_id: u32) -> Option<&'static str> {
    match class_id {
        0 => Some("person"),
        1 => Some("bicycle"),
        2 => Some("car"),
        3 => Some("motorcycle"),
        4 => Some("airplane"),
        5 => Some("bus"),
        6 => Some("train"),
        7 => Some("truck"),
        8 => Some("boat"),
        9 => Some("traffic light"),
        10 => Some("fire hydrant"),
        11 => Some("stop sign"),
        12 => Some("parking meter"),
        13 => Some("bench"),
        14 => Some("bird"),
        15 => Some("cat"),
        16 => Some("dog"),
        17 => Some("horse"),
        18 => Some("sheep"),
        19 => Some("cow"),
        20 => Some("elephant"),
        21 => Some("bear"),
        22 => Some("zebra"),
        23 => Some("giraffe"),
        24 => Some("backpack"),
        25 => Some("umbrella"),
        26 => Some("handbag"),
        27 => Some("tie"),
        28 => Some("suitcase"),
        29 => Some("frisbee"),
        30 => Some("skis"),
        31 => Some("snowboard"),
        32 => Some("sports ball"),
        33 => Some("kite"),
        34 => Some("baseball bat"),
        35 => Some("baseball glove"),
        36 => Some("skateboard"),
        37 => Some("surfboard"),
        38 => Some("tennis racket"),
        39 => Some("bottle"),
        40 => Some("wine glass"),
        41 => Some("cup"),
        42 => Some("fork"),
        43 => Some("knife"),
        44 => Some("spoon"),
        45 => Some("bowl"),
        46 => Some("banana"),
        47 => Some("apple"),
        48 => Some("sandwich"),
        49 => Some("orange"),
        50 => Some("broccoli"),
        51 => Some("carrot"),
        52 => Some("hot dog"),
        53 => Some("pizza"),
        54 => Some("donut"),
        55 => Some("cake"),
        56 => Some("chair"),
        57 => Some("couch"),
        58 => Some("potted plant"),
        59 => Some("bed"),
        60 => Some("dining table"),
        61 => Some("toilet"),
        62 => Some("tv"),
        63 => Some("laptop"),
        64 => Some("mouse"),
        65 => Some("remote"),
        66 => Some("keyboard"),
        67 => Some("cell phone"),
        68 => Some("microwave"),
        69 => Some("oven"),
        70 => Some("toaster"),
        71 => Some("sink"),
        72 => Some("refrigerator"),
        73 => Some("book"),
        74 => Some("clock"),
        75 => Some("vase"),
        76 => Some("scissors"),
        77 => Some("teddy bear"),
        78 => Some("hair drier"),
        79 => Some("toothbrush"),
        _ => None,
    }
}

/// 이미지 전처리: 리사이징, 레터박싱, 정규화
pub fn preprocess_image(image: &RgbImage) -> anyhow::Result<ArrayD<f32>> {
    let original_width = image.width() as f32;
    let original_height = image.height() as f32;

    // 종횡비 계산
    let aspect_ratio = original_width / original_height;

    let (new_width, new_height, offset_x, offset_y) = if aspect_ratio > 1.0 {
        // 가로가 더 긴 경우
        let new_width = MODEL_INPUT_SIZE as f32;
        let new_height = new_width / aspect_ratio;
        let offset_x = 0.0;
        let offset_y = (MODEL_INPUT_SIZE as f32 - new_height) / 2.0;
        (
            new_width as u32,
            new_height as u32,
            offset_x as u32,
            offset_y as u32,
        )
    } else {
        // 세로가 더 긴 경우
        let new_height = MODEL_INPUT_SIZE as f32;
        let new_width = new_height * aspect_ratio;
        let offset_x = (MODEL_INPUT_SIZE as f32 - new_width) / 2.0;
        let offset_y = 0.0;
        (
            new_width as u32,
            new_height as u32,
            offset_x as u32,
            offset_y as u32,
        )
    };

    // 이미지 리사이즈 (종횡비 유지)
    let resized = image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // 정사각형 캔버스 생성 (회색 배경)
    let mut canvas = RgbImage::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    let padding_color = Rgb([114, 114, 114]); // 회색 패딩

    // 캔버스를 패딩 색상으로 채우기
    for pixel in canvas.pixels_mut() {
        *pixel = padding_color;
    }

    // 리사이즈된 이미지를 캔버스 중앙에 배치
    for y in 0..new_height {
        for x in 0..new_width {
            let canvas_x = x + offset_x;
            let canvas_y = y + offset_y;
            if canvas_x < MODEL_INPUT_SIZE && canvas_y < MODEL_INPUT_SIZE {
                canvas.put_pixel(canvas_x, canvas_y, *resized.get_pixel(x, y));
            }
        }
    }

    // HWC -> CHW 변환 및 정규화 (0~1)
    let mut input_data =
        Vec::with_capacity(1 * 3 * MODEL_INPUT_SIZE as usize * MODEL_INPUT_SIZE as usize);
    for c in 0..3 {
        for y in 0..MODEL_INPUT_SIZE {
            for x in 0..MODEL_INPUT_SIZE {
                let pixel_value = canvas.get_pixel(x, y)[c as usize] as f32 / 255.0;
                input_data.push(pixel_value);
            }
        }
    }

    // 텐서 생성
    Ok(ArrayD::from_shape_vec(
        IxDyn(&[1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize]),
        input_data,
    )?)
}

/// YOLOv9 모델 출력 파싱
pub fn parse_yolov9_outputs(
    output_tensor: &ndarray::ArrayViewD<f32>,
    original_width: u32,
    original_height: u32,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();
    
    // 출력 텐서 형태 확인
    let shape = output_tensor.shape();
    println!("🔍 Output tensor shape: {:?}", shape);
    
    if shape.len() != 3 {
        return Err(anyhow::anyhow!("Invalid output tensor shape: expected 3 dimensions"));
    }
    
    // 다양한 YOLOv9 출력 형태 지원
    let (num_boxes, num_classes) = match shape[1] {
        84 | 85 => (shape[2], 80),
        _ => {
            println!("⚠️ Unexpected shape: {:?}, trying alternative parsing", shape);
            (shape[2], shape[1] - 4)
        }
    };

    println!("📊 Parsing {} boxes with {} classes", num_boxes, num_classes);

    // 출력 데이터를 (84, 8400) 형태로 변환
    let output_data = output_tensor.to_owned();

    // 바운딩 박스와 클래스 점수 분리
    let boxes = output_data.slice(ndarray::s![0, 0..4, ..]); // (4, N): x, y, w, h
    let scores = output_data.slice(ndarray::s![0, 4.., ..]); // (80, N): class scores

    // 디버깅: 첫 번째 박스의 값들 확인
    if num_boxes > 0 {
        let first_box = boxes.slice(ndarray::s![.., 0]);
        let first_scores = scores.slice(ndarray::s![.., 0]);
        println!("🔍 First box: {:?}", first_box.to_vec());
        println!("🔍 First scores (first 10): {:?}", first_scores.slice(ndarray::s![..10]).to_vec());
    }

    for box_idx in 0..num_boxes {
        // 바운딩 박스 좌표 (center_x, center_y, width, height) - 픽셀 좌표로 출력됨
        // 배열 경계 검사 추가
        if box_idx >= boxes.shape()[1] {
            continue;
        }
        
        let cx = boxes[[0, box_idx]];
        let cy = boxes[[1, box_idx]];
        let w = boxes[[2, box_idx]];
        let h = boxes[[3, box_idx]];
        
        // 픽셀 좌표를 정규화된 좌표로 변환 (640x640 기준)
        let cx_norm = cx / MODEL_INPUT_SIZE as f32;
        let cy_norm = cy / MODEL_INPUT_SIZE as f32;
        let w_norm = w / MODEL_INPUT_SIZE as f32;
        let h_norm = h / MODEL_INPUT_SIZE as f32;
        
        // 더 엄격한 바운딩 박스 검증 (정규화된 좌표 기준)
        if w_norm <= 0.0 || h_norm <= 0.0 || w_norm > 1.0 || h_norm > 1.0 {
            continue;
        }
        
        // center 좌표가 이미지 범위 내에 있는지 확인
        if !(0.0..=1.0).contains(&cx_norm) || !(0.0..=1.0).contains(&cy_norm) {
            continue;
        }
        
        // 클래스 확률 계산 (시그모이드 적용)
        let mut max_conf = 0.0;
        let mut best_class = 0;
        
        for class_idx in 0..num_classes {
            // 배열 경계 검사 추가
            if class_idx >= scores.shape()[0] || box_idx >= scores.shape()[1] {
                continue;
            }
            
            let raw_score = scores[[class_idx, box_idx]];
            // 점수 스케일링 (매우 작은 값들을 확대)
            let scaled_score = raw_score * 1000.0; // 스케일링 팩터
            let conf = math_utils::sigmoid(scaled_score); // 시그모이드 적용
            
            if conf > max_conf {
                max_conf = conf;
                best_class = class_idx;
            }
        }
        
        // 신뢰도 임계값 확인 (GUI에서 설정한 값 사용)
        if max_conf > confidence_threshold {
            // 정규화된 좌표로 center_x, center_y, width, height -> x1, y1, x2, y2 변환
            let x1 = (cx_norm - w_norm / 2.0).clamp(0.0, 1.0);
            let y1 = (cy_norm - h_norm / 2.0).clamp(0.0, 1.0);
            let x2 = (cx_norm + w_norm / 2.0).clamp(0.0, 1.0);
            let y2 = (cy_norm + h_norm / 2.0).clamp(0.0, 1.0);
            
            // 더 엄격한 바운딩 박스 유효성 검증  
            if x1 >= x2 || y1 >= y2 || 
               (x2 - x1) < 0.02 || (y2 - y1) < 0.02 ||  // 최소 크기 증가
               (x2 - x1) > 0.95 || (y2 - y1) > 0.95
            {
                // 최대 크기 제한 (전체 이미지 제외)
                continue;
            }
            
            // 레터박싱 좌표를 원본 이미지 좌표로 변환
            let original_bbox = letterbox_to_original_coords([x1, y1, x2, y2], original_width, original_height);
            
            // 변환된 좌표 유효성 검증
            let [ox1, oy1, ox2, oy2] = original_bbox;
            if ox1 >= ox2 || oy1 >= oy2 || 
               (ox2 - ox1) < 0.02 || (oy2 - oy1) < 0.02 ||
               (ox2 - ox1) > 0.95 || (oy2 - oy1) > 0.95 {
                continue;
            }
            
            if let Some(class_name) = yolov9_id_to_label(best_class as u32) {
                // 디버깅: 높은 신뢰도 검출만 출력
                if max_conf > 0.7 {
                    println!("🎯 High confidence detection: {} ({}%) at [{:.3}, {:.3}, {:.3}, {:.3}]", 
                             class_name, (max_conf * 100.0) as i32, ox1, oy1, ox2, oy2);
                }
                
                detections.push(Detection {
                    bbox: original_bbox,
                    confidence: max_conf,
                    class_id: best_class as u32,
                    class_name: class_name.to_string(),
                });
            }
        }
    }
    
    // NMS 적용 (GUI에서 설정한 임계값 사용)
    non_maximum_suppression(&mut detections, nms_threshold);

    Ok(detections)
}

/// 레터박싱 좌표를 원본 이미지 좌표로 변환
fn letterbox_to_original_coords(
    bbox: [f32; 4], // [x1, y1, x2, y2] in letterboxed coordinates (0-1)
    original_width: u32,
    original_height: u32,
) -> [f32; 4] {
    let aspect_ratio = original_width as f32 / original_height as f32;

    let (scale, offset_x, offset_y) = if aspect_ratio > 1.0 {
        // 가로가 더 긴 경우
        let scale = MODEL_INPUT_SIZE as f32 / original_width as f32;
        let offset_x = 0.0;
        let offset_y = (MODEL_INPUT_SIZE as f32 - MODEL_INPUT_SIZE as f32 / aspect_ratio) / 2.0;
        (scale, offset_x, offset_y)
    } else {
        // 세로가 더 긴 경우
        let scale = MODEL_INPUT_SIZE as f32 / original_height as f32;
        let offset_x = (MODEL_INPUT_SIZE as f32 - MODEL_INPUT_SIZE as f32 * aspect_ratio) / 2.0;
        let offset_y = 0.0;
        (scale, offset_x, offset_y)
    };

    // 레터박싱 좌표를 픽셀 좌표로 변환
    let x1_pixel = bbox[0] * MODEL_INPUT_SIZE as f32;
    let y1_pixel = bbox[1] * MODEL_INPUT_SIZE as f32;
    let x2_pixel = bbox[2] * MODEL_INPUT_SIZE as f32;
    let y2_pixel = bbox[3] * MODEL_INPUT_SIZE as f32;

    // 패딩 제거
    let x1_unpadded = (x1_pixel - offset_x) / scale;
    let y1_unpadded = (y1_pixel - offset_y) / scale;
    let x2_unpadded = (x2_pixel - offset_x) / scale;
    let y2_unpadded = (y2_pixel - offset_y) / scale;

    // 원본 이미지 범위로 클리핑
    let x1_final = x1_unpadded.max(0.0).min(original_width as f32);
    let y1_final = y1_unpadded.max(0.0).min(original_height as f32);
    let x2_final = x2_unpadded.max(0.0).min(original_width as f32);
    let y2_final = y2_unpadded.max(0.0).min(original_height as f32);

    // 정규화된 좌표로 변환 (0-1)
    [
        x1_final / original_width as f32,
        y1_final / original_height as f32,
        x2_final / original_width as f32,
        y2_final / original_height as f32,
    ]
}

/// Non-Maximum Suppression (NMS) 구현 (기존 lib.rs와 동일)
fn non_maximum_suppression(detections: &mut Vec<Detection>, nms_threshold: f32) {
    if detections.is_empty() {
        return;
    }

    // 신뢰도 기준으로 정렬 (높은 신뢰도가 먼저) - 패닉 방지
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(i);

        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }

            // 같은 클래스인 경우에만 NMS 적용
            if detections[i].class_id == detections[j].class_id {
                let iou = math_utils::calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > nms_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    // 유지할 검출 결과만 남기기
    let mut new_detections = Vec::new();
    for &idx in &keep {
        if idx < detections.len() {
            new_detections.push(detections[idx].clone());
        }
    }

    *detections = new_detections;

    // 최대 검출 개수 제한 (성능 향상)
    if detections.len() > 50 {
        detections.truncate(50);
    }
}

/// 검출된 객체에 바운딩 박스 그리기
pub fn draw_detections(
    image: &mut RgbImage,
    detections: &[Detection],
    color_mode: &crate::config::ColorMappingMode,
) {
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;

        // 좌표 유효성 검증
        if x1 >= x2 || y1 >= y2 || x1 < 0.0 || y1 < 0.0 || x2 > 1.0 || y2 > 1.0 {
            continue;
        }

        let x1 = (x1 * image.width() as f32) as i32;
        let y1 = (y1 * image.height() as f32) as i32;
        let x2 = (x2 * image.width() as f32) as i32;
        let y2 = (y2 * image.height() as f32) as i32;

        // 픽셀 좌표 유효성 확인
        if x1 >= x2
            || y1 >= y2
            || x1 < 0
            || y1 < 0
            || x2 > image.width() as i32
            || y2 > image.height() as i32
        {
            continue;
        }

        // 신뢰도에 따른 색상 결정 (전달받은 색상 매핑 방식 사용)
        let color = match color_mode {
            crate::config::ColorMappingMode::Fixed => BBOX_COLOR,
            crate::config::ColorMappingMode::RangeBased => {
                color_utils::get_confidence_color(detection.confidence)
            }
            crate::config::ColorMappingMode::Gradient => {
                color_utils::get_confidence_color_gradient(detection.confidence)
            }
            crate::config::ColorMappingMode::HsvBased => {
                color_utils::get_confidence_color_hsv(detection.confidence)
            }
        };

        let rect = Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32);
        draw_hollow_rect_mut(image, rect, color);
    }
}

/// 모델 세션을 캐시하는 구조체
pub struct ModelCache {
    environment: Arc<Environment>,
    session: Option<ort::InMemorySession<'static>>,
    current_model_file: Option<String>,
}

impl ModelCache {
    /// 새로운 모델 캐시 생성
    pub fn new() -> anyhow::Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("yolov9-embedded")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        Ok(Self {
            environment,
            session: None,
            current_model_file: None,
        })
    }

    pub fn get_session(
        &mut self,
        model_file_name: &str,
    ) -> anyhow::Result<&ort::InMemorySession<'static>> {
        let need_reload = match &self.current_model_file {
            Some(cur) => cur != model_file_name,
            None => true,
        };
        if need_reload {
            #[cfg(target_os = "macos")]
            let session = SessionBuilder::new(&self.environment)?
                .with_execution_providers([
                    ExecutionProvider::CoreML(CoreMLExecutionProviderOptions {
                        use_cpu_only: false,
                        enable_on_subgraph: true,
                        only_enable_device_with_ane: true, // M4 ANE 활용
                    }),
                    ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
                ])?
                // 1. 최적화 레벨 조정 (성능 vs 초기화 시간)
                .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
                // 2. 스레드 설정 최적화 (M4 Mac 기준)
                .with_intra_threads(4)? // M4 성능 코어 개수
                .with_inter_threads(2)? // 병렬 실행용
                .with_parallel_execution(false)? // YOLOv9는 순차 실행이 더 빠름
                // 3. 메모리 최적화
                .with_memory_pattern(true)? // 고정 입력 크기라면 활성화
                .with_allocator(ort::AllocatorType::Device)? // GPU 메모리 사용
                .with_model_from_memory(self.load_embedded_model_bytes(model_file_name)?)?;
            #[cfg(not(target_os = "macos"))]
            let session = SessionBuilder::new(&self.environment)?
                .with_execution_providers([ExecutionProvider::CPU(
                    CPUExecutionProviderOptions::default(),
                )])?
                // 1. 최적화 레벨 조정 (성능 vs 초기화 시간)
                .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
                // 2. 스레드 설정 최적화
                .with_intra_threads(16)? // 성능 코어 개수
                .with_inter_threads(8)? // 병렬 실행용
                .with_parallel_execution(false)? // YOLOv9는 순차 실행이 더 빠름
                // 3. 메모리 최적화
                .with_memory_pattern(true)? // 고정 입력 크기라면 활성화
                .with_allocator(ort::AllocatorType::Device)? // GPU 메모리 사용
                .with_model_from_memory(self.load_embedded_model_bytes(model_file_name)?)?;

            self.session = Some(session);
            self.current_model_file = Some(model_file_name.to_string());
            println!(
                "Loading model: {} - Optimized for inference",
                model_file_name
            );
        }

        match self.session.as_ref() {
            Some(session) => Ok(session),
            None => Err(anyhow::anyhow!("Model session is not initialized")),
        }
    }

    /// 임베디드 모델 바이트 로드
    fn load_embedded_model_bytes(&self, file_name: &str) -> anyhow::Result<&'static [u8]> {
        get_embedded_model_bytes(file_name)
    }
}

/// SQLite DB 관리 구조체
pub struct InferenceDb {
    conn: Connection,
}

impl InferenceDb {
    /// DB 초기화 및 테이블 생성
    pub fn new() -> AppResult<Self> {
        // DB 파일 경로 가져오기
        let db_path = fs_utils::get_database_path()?;

        let conn = Connection::open(&db_path).map_err(AppError::DatabaseError)?;

        // 테이블 생성
        conn.execute(
            "CREATE TABLE IF NOT EXISTS inference_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                model_file_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                detections_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_path, model_file_name)
            )",
            [],
        )
        .map_err(AppError::DatabaseError)?;

        // 인덱스 생성 (성능 향상)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_hash ON inference_cache(image_hash)",
            [],
        )
        .map_err(AppError::DatabaseError)?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_file ON inference_cache(model_file_name)",
            [],
        )
        .map_err(AppError::DatabaseError)?;

        Ok(Self { conn })
    }

    /// 이미지 해시 계산
    pub fn calculate_image_hash(image_data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(image_data);
        format!("{:x}", hasher.finalize())
    }

    /// pre-NMS 검출 결과를 DB에 저장
    pub fn save_pre_nms_detections(
        &self,
        image_path: &str,
        image_data: &[u8],
        model_file_name: &str,
        detections: &[Detection],
    ) -> anyhow::Result<()> {
        let image_hash = Self::calculate_image_hash(image_data);
        let (model_name, _) = get_model_info(model_file_name);

        // 이미지 크기 추출 (첫 번째 검출에서 추정)
        let (width, height) = if let Some(_first_det) = detections.first() {
            // bbox는 정규화된 좌표이므로, 실제 픽셀 크기는 추정 불가
            // 대신 기본값 사용 (나중에 실제 이미지 크기로 업데이트)
            (640, 480)
        } else {
            (640, 480)
        };

        let detections_json = serde_json::to_string(detections)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO inference_cache 
             (image_path, image_hash, model_file_name, model_name, image_width, image_height, detections_json)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                image_path,
                image_hash,
                model_file_name,
                model_name,
                width,
                height,
                detections_json
            ],
        )?;

        Ok(())
    }

    /// DB에서 pre-NMS 검출 결과 로드
    pub fn load_pre_nms_detections(
        &self,
        image_path: &str,
        model_file_name: &str,
    ) -> anyhow::Result<Option<Vec<Detection>>> {
        let mut stmt = self.conn.prepare(
            "SELECT detections_json FROM inference_cache 
             WHERE image_path = ? AND model_file_name = ?",
        )?;

        let mut rows = stmt.query(params![image_path, model_file_name])?;

        if let Some(row) = rows.next()? {
            let detections_json: String = row.get(0)?;
            let detections: Vec<Detection> = serde_json::from_str(&detections_json)?;
            Ok(Some(detections))
        } else {
            Ok(None)
        }
    }

    /// 이미지 해시로 검출 결과 검색 (파일 경로가 바뀌었을 때)
    pub fn load_by_image_hash(
        &self,
        image_hash: &str,
        model_file_name: &str,
    ) -> anyhow::Result<Option<(String, Vec<Detection>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT image_path, detections_json FROM inference_cache 
             WHERE image_hash = ? AND model_file_name = ?",
        )?;

        let mut rows = stmt.query(params![image_hash, model_file_name])?;

        if let Some(row) = rows.next()? {
            let image_path: String = row.get(0)?;
            let detections_json: String = row.get(1)?;
            let detections: Vec<Detection> = serde_json::from_str(&detections_json)?;
            Ok(Some((image_path, detections)))
        } else {
            Ok(None)
        }
    }

    /// 캐시 정리 (오래된 항목 삭제)
    pub fn cleanup_old_cache(&self, days_old: i32) -> anyhow::Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM inference_cache 
             WHERE created_at < datetime('now', '-{} days')",
            params![days_old],
        )?;
        Ok(deleted)
    }

    /// 캐시 통계
    pub fn get_cache_stats(&self) -> anyhow::Result<(usize, usize)> {
        let total_count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM inference_cache", [], |row| row.get(0))?;

        let unique_images: usize = self.conn.query_row(
            "SELECT COUNT(DISTINCT image_hash) FROM inference_cache",
            [],
            |row| row.get(0),
        )?;

        Ok((total_count, unique_images))
    }

    /// 특정 이미지와 모델의 캐시 항목 삭제
    pub fn delete_cache_entry(
        &self,
        image_path: &str,
        model_file_name: &str,
    ) -> anyhow::Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM inference_cache WHERE image_path = ? AND model_file_name = ?",
            params![image_path, model_file_name],
        )?;
        Ok(deleted)
    }
}

/// 메인 객체 검출 함수 (캐시 사용)
pub fn detect_objects_with_cache(
    image_data: &[u8], 
    cache: &mut ModelCache,
    model_file: &str,
) -> anyhow::Result<crate::models::DetectionResult> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // 캐시된 세션 가져오기
    let session = cache.get_session(model_file)?;

    // 이미지 전처리
    let input_array = preprocess_image(&img)?;
    let cow_array = CowArray::from(&input_array);
    let input_value = Value::from_array(session.allocator(), &cow_array)?;

    // 추론 시간 측정 시작
    let start_time = std::time::Instant::now();

    // 추론 실행
    let outputs = session.run(vec![input_value])?;

    // 추론 시간 측정 종료
    let inference_time = start_time.elapsed();
    let inference_time_ms = inference_time.as_secs_f64() * 1000.0;

    // 결과 파싱
    let mut detections = Vec::new();
    if let Some(output) = outputs.first() {
        let output_tensor = output.try_extract::<f32>()?;
        let output_view = output_tensor.view();

        // YOLOv9 출력 파싱 (기본 임계값 사용)
        detections = parse_yolov9_outputs(&output_view, img.width(), img.height(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)?;
    }

    // 원본 이미지 그대로 반환 (GUI에서 박스 그리기 처리)
    let result_image = img.clone();

    let result = crate::models::DetectionResult {
        detections,
        result_image,
        inference_time_ms,
    };

    Ok(result)
}

/// 메인 객체 검출 함수 (NMS 적용 전 원시 결과 반환)
pub fn detect_objects_with_cache_pre_nms(
    image_data: &[u8], 
    cache: &mut ModelCache,
    model_file: &str,
) -> anyhow::Result<crate::models::DetectionResult> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // 캐시된 세션 가져오기
    let session = cache.get_session(model_file)?;

    // 이미지 전처리
    let input_array = preprocess_image(&img)?;
    let cow_array = CowArray::from(&input_array);
    let input_value = Value::from_array(session.allocator(), &cow_array)?;

    // 추론 시간 측정 시작
    let start_time = std::time::Instant::now();

    // 추론 실행
    let outputs = session.run(vec![input_value])?;

    // 추론 시간 측정 종료
    let inference_time = start_time.elapsed();
    let inference_time_ms = inference_time.as_secs_f64() * 1000.0;

    // 결과 파싱 (NMS 적용 안 함)
    let mut detections = Vec::new();
    if let Some(output) = outputs.first() {
        let output_tensor = output.try_extract::<f32>()?;
        let output_view = output_tensor.view();

        // YOLOv9 출력 파싱 (매우 낮은 임계값으로 모든 후보 추출)
        detections = parse_yolov9_outputs_no_nms(&output_view, img.width(), img.height(), 0.1)?;
    }

    // 원본 이미지 그대로 반환 (GUI에서 박스 그리기 처리)
    let result_image = img.clone();

    let result = crate::models::DetectionResult {
        detections,
        result_image,
        inference_time_ms,
    };

    Ok(result)
}

/// YOLOv9 모델 출력 파싱 (NMS 적용 안 함)
fn parse_yolov9_outputs_no_nms(
    output_tensor: &ndarray::ArrayViewD<f32>,
    original_width: u32,
    original_height: u32,
    confidence_threshold: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();
    
    // 출력 텐서 형태 확인
    let shape = output_tensor.shape();
    
    if shape.len() != 3 {
        return Err(anyhow::anyhow!("Invalid output tensor shape: expected 3 dimensions"));
    }
    
    // 다양한 YOLOv9 출력 형태 지원
    let (num_boxes, num_classes) = match shape[1] {
        84 | 85 => (shape[2], 80),
        _ => (shape[2], shape[1] - 4)
    };

    // 출력 데이터를 (84, 8400) 형태로 변환
    let output_data = output_tensor.to_owned();

    // 바운딩 박스와 클래스 점수 분리
    let boxes = output_data.slice(ndarray::s![0, 0..4, ..]); // (4, N): x, y, w, h
    let scores = output_data.slice(ndarray::s![0, 4.., ..]); // (80, N): class scores

    for box_idx in 0..num_boxes {
        // 배열 경계 검사 추가
        if box_idx >= boxes.shape()[1] {
            continue;
        }

        let cx = boxes[[0, box_idx]];
        let cy = boxes[[1, box_idx]];
        let w = boxes[[2, box_idx]];
        let h = boxes[[3, box_idx]];

        // 픽셀 좌표를 정규화된 좌표로 변환 (640x640 기준)
        let cx_norm = cx / MODEL_INPUT_SIZE as f32;
        let cy_norm = cy / MODEL_INPUT_SIZE as f32;
        let w_norm = w / MODEL_INPUT_SIZE as f32;
        let h_norm = h / MODEL_INPUT_SIZE as f32;

        // 기본적인 바운딩 박스 검증
        if w_norm <= 0.0 || h_norm <= 0.0 || w_norm > 1.0 || h_norm > 1.0 {
            continue;
        }

        // center 좌표가 이미지 범위 내에 있는지 확인
        if !(0.0..=1.0).contains(&cx_norm) || !(0.0..=1.0).contains(&cy_norm) {
            continue;
        }

        // 클래스 확률 계산 (시그모이드 적용)
        let mut max_conf = 0.0;
        let mut best_class = 0;

        for class_idx in 0..num_classes {
            // 배열 경계 검사 추가
            if class_idx >= scores.shape()[0] || box_idx >= scores.shape()[1] {
                continue;
            }

            let raw_score = scores[[class_idx, box_idx]];
            // 점수 스케일링 (매우 작은 값들을 확대)
            let scaled_score = raw_score * 1000.0; // 스케일링 팩터
            let conf = math_utils::sigmoid(scaled_score); // 시그모이드 적용

            if conf > max_conf {
                max_conf = conf;
                best_class = class_idx;
            }
        }

        // 신뢰도 임계값 확인 (매우 낮은 임계값)
        if max_conf > confidence_threshold {
            // 정규화된 좌표로 center_x, center_y, width, height -> x1, y1, x2, y2 변환
            let x1 = (cx_norm - w_norm / 2.0).clamp(0.0, 1.0);
            let y1 = (cy_norm - h_norm / 2.0).clamp(0.0, 1.0);
            let x2 = (cx_norm + w_norm / 2.0).clamp(0.0, 1.0);
            let y2 = (cy_norm + h_norm / 2.0).clamp(0.0, 1.0);

            // 기본적인 바운딩 박스 유효성 검증
            if x1 >= x2 || y1 >= y2 || (x2 - x1) < 0.01 || (y2 - y1) < 0.01 {
                continue;
            }

            // 레터박싱 좌표를 원본 이미지 좌표로 변환
            let original_bbox = letterbox_to_original_coords([x1, y1, x2, y2], original_width, original_height);

            if let Some(class_name) = yolov9_id_to_label(best_class as u32) {
                detections.push(Detection {
                    bbox: original_bbox,
                    confidence: max_conf,
                    class_id: best_class as u32,
                    class_name: class_name.to_string(),
                });
            }
        }
    }

    Ok(detections)
}

/// 메인 객체 검출 함수 (기본 모델 사용)
pub fn detect_objects(image_data: &[u8]) -> anyhow::Result<crate::models::DetectionResult> {
    // ModelCache를 생성하여 사용
    let mut cache = ModelCache::new()?;
    detect_objects_with_cache(image_data, &mut cache, "gelan-e.onnx")
}

/// YOLOv9 전용 객체 검출기 구현
pub struct YoloV9Detector {
    cache: ModelCache,
    model_file: String,
}

impl YoloV9Detector {
    pub fn new(model_file: String) -> anyhow::Result<Self> {
        Ok(Self {
            cache: ModelCache::new()?,
            model_file,
        })
    }

    pub fn detect(&mut self, image_data: &[u8]) -> anyhow::Result<crate::models::DetectionResult> {
        detect_objects_with_cache(image_data, &mut self.cache, &self.model_file)
    }
}

// 임베디드 모델 모듈을 파일 끝에 정의하고 함수들을 외부로 노출
pub use embedded_models::{get_embedded_model_list, get_embedded_model_bytes};

// 임베디드 모델 모듈
mod embedded_models {
    use include_dir::{Dir, include_dir};

    // 임베디드 리소스: assets/models/yolov9 폴더의 모든 onnx 파일을 임베딩
    static ASSETS_MODELS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/assets/models/yolov9");

    /// 임베디드된 모델 파일(.onnx) 목록 반환
    pub fn get_embedded_model_list() -> Vec<String> {
        ASSETS_MODELS_DIR
            .files()
            .filter_map(|f| {
                let path = f.path();
                if let Some(ext) = path.extension()
                    && ext == "onnx"
                {
                    // 안전한 파일명 추출
                    return path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(|s| s.to_string());
                }
                None
            })
            .collect()
    }

    /// 파일명으로 임베디드된 모델 바이트를 조회
    pub fn get_embedded_model_bytes(file_name: &str) -> anyhow::Result<&'static [u8]> {
        let file = ASSETS_MODELS_DIR
            .files()
            .find(|f| {
                f.path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|s| s == file_name)
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow::anyhow!(format!("Embedded model not found: {}", file_name)))?;
        Ok(file.contents())
    }
} 