use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn};
use ort::execution_providers::{CPUExecutionProviderOptions, CoreMLExecutionProviderOptions};
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use std::sync::Arc;

use crate::utils::{color_utils, math_utils};
use crate::models::{DetectionResult, Detection};

// 상수 정의
const MODEL_INPUT_SIZE: u32 = 560;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.2;

// 임베디드 리소스: RF-DETR 모델 파일
static RF_DETR_ORIGINAL_ONNX: &[u8] = include_bytes!("../../assets/models/rf-detr.onnx");

/// RF-DETR COCO 클래스 ID를 클래스 이름으로 변환
fn rf_detr_id_to_label(class_id: u32) -> Option<&'static str> {
    match class_id {
        1 => Some("person"),
        2 => Some("bicycle"),
        3 => Some("car"),
        4 => Some("motorcycle"),
        5 => Some("airplane"),
        6 => Some("bus"),
        7 => Some("train"),
        8 => Some("truck"),
        9 => Some("boat"),
        10 => Some("traffic light"),
        11 => Some("fire hydrant"),
        13 => Some("stop sign"),
        14 => Some("parking meter"),
        15 => Some("bench"),
        16 => Some("bird"),
        17 => Some("cat"),
        18 => Some("dog"),
        19 => Some("horse"),
        20 => Some("sheep"),
        21 => Some("cow"),
        22 => Some("elephant"),
        23 => Some("bear"),
        24 => Some("zebra"),
        25 => Some("giraffe"),
        27 => Some("backpack"),
        28 => Some("umbrella"),
        31 => Some("handbag"),
        32 => Some("tie"),
        33 => Some("suitcase"),
        34 => Some("frisbee"),
        35 => Some("skis"),
        36 => Some("snowboard"),
        37 => Some("sports ball"),
        38 => Some("kite"),
        39 => Some("baseball bat"),
        40 => Some("baseball glove"),
        41 => Some("skateboard"),
        42 => Some("surfboard"),
        43 => Some("tennis racket"),
        44 => Some("bottle"),
        46 => Some("wine glass"),
        47 => Some("cup"),
        48 => Some("fork"),
        49 => Some("knife"),
        50 => Some("spoon"),
        51 => Some("bowl"),
        52 => Some("banana"),
        53 => Some("apple"),
        54 => Some("sandwich"),
        55 => Some("orange"),
        56 => Some("broccoli"),
        57 => Some("carrot"),
        58 => Some("hot dog"),
        59 => Some("pizza"),
        60 => Some("donut"),
        61 => Some("cake"),
        62 => Some("chair"),
        63 => Some("couch"),
        64 => Some("potted plant"),
        65 => Some("bed"),
        67 => Some("dining table"),
        70 => Some("toilet"),
        72 => Some("tv"),
        73 => Some("laptop"),
        74 => Some("mouse"),
        75 => Some("remote"),
        76 => Some("keyboard"),
        77 => Some("cell phone"),
        78 => Some("microwave"),
        79 => Some("oven"),
        80 => Some("toaster"),
        81 => Some("sink"),
        82 => Some("refrigerator"),
        84 => Some("book"),
        85 => Some("clock"),
        86 => Some("vase"),
        87 => Some("scissors"),
        88 => Some("teddy bear"),
        89 => Some("hair drier"),
        90 => Some("toothbrush"),
        _ => None,
    }
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

/// RF-DETR 모델 출력 파싱
pub fn parse_rf_detr_outputs(
    bbox_tensor: &ndarray::ArrayViewD<f32>,  // 바운딩 박스 좌표
    class_tensor: &ndarray::ArrayViewD<f32>, // 클래스 로짓
    original_width: u32,
    original_height: u32,
) -> anyhow::Result<Vec<Detection>> {
    const MAX_DETECTIONS: usize = 100;

    let mut detections = Vec::new();
    let num_queries = class_tensor.shape()[1];

    for q in 0..num_queries.min(MAX_DETECTIONS) {
        // 클래스 확률 계산
        let mut max_conf = 0.0;
        let mut best_class = 0;

        // background 클래스(0) 제외하고 실제 객체 클래스만 처리 (1-90)
        for c in 1..91 {
            let logit = class_tensor[[0, q, c]];
            let conf = math_utils::sigmoid(logit);
            if conf > max_conf {
                max_conf = conf;
                best_class = c;
            }
        }

        // 신뢰도 임계값 확인
        if max_conf > CONFIDENCE_THRESHOLD {
            // 바운딩 박스 좌표 추출 (레터박싱된 이미지 기준)
            let cx = bbox_tensor[[0, q, 0]];
            let cy = bbox_tensor[[0, q, 1]];
            let w = bbox_tensor[[0, q, 2]];
            let h = bbox_tensor[[0, q, 3]];

            // 유효한 바운딩 박스인지 확인
            if w > 0.0 && h > 0.0 {
                let x1 = (cx - w / 2.0).max(0.0).min(1.0);
                let y1 = (cy - h / 2.0).max(0.0).min(1.0);
                let x2 = (cx + w / 2.0).max(0.0).min(1.0);
                let y2 = (cy + h / 2.0).max(0.0).min(1.0);

                // 레터박싱 좌표를 원본 이미지 좌표로 변환
                let original_bbox =
                    letterbox_to_original_coords([x1, y1, x2, y2], original_width, original_height);

                if let Some(class_name) = rf_detr_id_to_label(best_class as u32) {
                    detections.push(Detection {
                        bbox: original_bbox,
                        confidence: max_conf,
                        class_id: best_class as u32,
                        class_name: class_name.to_string(),
                    });
                }
            }
        }
    }

    Ok(detections)
}

/// 검출된 객체에 바운딩 박스 그리기
pub fn draw_detections(image: &mut RgbImage, detections: &[Detection]) {
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        let x1 = (x1 * image.width() as f32) as i32;
        let y1 = (y1 * image.height() as f32) as i32;
        let x2 = (x2 * image.width() as f32) as i32;
        let y2 = (y2 * image.height() as f32) as i32;

        let rect = Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32);
        // 신뢰도에 따라 색상을 다르게 표시 (시각적 구분 향상)
        let color = color_utils::get_confidence_color_hsv(detection.confidence);
        draw_hollow_rect_mut(image, rect, color);
    }
}

/// 모델 세션을 캐시하는 구조체
pub struct ModelCache {
    environment: Arc<Environment>,
    session: Option<ort::InMemorySession<'static>>,
}

impl ModelCache {
    /// 새로운 모델 캐시 생성
    pub fn new() -> anyhow::Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("rf-detr-embedded")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?,
        );

        Ok(Self {
            environment,
            session: None,
        })
    }

    pub fn get_session(&mut self) -> anyhow::Result<&ort::InMemorySession<'static>> {
        if self.session.is_none() {
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
                .with_optimization_level(ort::GraphOptimizationLevel::Level1)? // Level3 → Level1
                
                // 2. 스레드 설정 최적화 (M4 Mac 기준)
                .with_intra_threads(4)?  // M4 성능 코어 개수
                .with_inter_threads(2)?  // 병렬 실행용
                .with_parallel_execution(false)?  // RF-DETR는 순차 실행이 더 빠름
                
                // 3. 메모리 최적화
                .with_memory_pattern(true)?  // 고정 입력 크기라면 활성화
                .with_allocator(ort::AllocatorType::Device)?  // GPU 메모리 사용
                
                .with_model_from_memory(RF_DETR_ORIGINAL_ONNX)?;
            
            self.session = Some(session);
            println!("Loading model: RF-DETR Original (108 MB) - Optimized for M4");
        }
        
        match self.session.as_ref() {
            Some(session) => Ok(session),
            None => Err(anyhow::anyhow!("Model session is not initialized")),
        }
    }

    /// 모델을 미리 로드
    pub fn preload_model(&mut self) -> anyhow::Result<()> {
        self.get_session()?;
        Ok(())
    }
}

/// 메인 객체 검출 함수 (캐시 사용)
pub fn detect_objects_with_cache(
    image_data: &[u8], 
    cache: &mut ModelCache
) -> anyhow::Result<DetectionResult> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // 캐시된 세션 가져오기
    let session = cache.get_session()?;

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
    if outputs.len() >= 2 {
        let logits_tensor = outputs[0].try_extract::<f32>()?;
        let boxes_tensor = outputs[1].try_extract::<f32>()?;

        let logits_view = logits_tensor.view();
        let boxes_view = boxes_tensor.view();

        // RF-DETR 출력 파싱
        detections = parse_rf_detr_outputs(&logits_view, &boxes_view, img.width(), img.height())?;

        // NMS 적용으로 중복 박스 억제
        apply_nms(&mut detections, NMS_THRESHOLD);
    }

    // 바운딩 박스가 포함된 이미지 생성
    let mut result_image = img.clone();
    draw_detections(&mut result_image, &detections);

    let result = DetectionResult {
        detections,
        result_image,
        inference_time_ms,
    };

    Ok(result)
}

/// 메인 객체 검출 함수 (기본 모델 사용)
pub fn detect_objects(image_data: &[u8]) -> anyhow::Result<DetectionResult> {
    // ModelCache를 생성하여 사용
    let mut cache = ModelCache::new()?;
    detect_objects_with_cache(image_data, &mut cache)
}

/// Non-Maximum Suppression (클래스별 NMS)
fn apply_nms(detections: &mut Vec<Detection>, nms_threshold: f32) {
    if detections.is_empty() {
        return;
    }

    // 신뢰도 기준으로 정렬 (높은 신뢰도 우선)
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());

    for det in detections.iter() {
        // 동일 클래스에 대해서만 NMS 비교
        let mut should_keep = true;
        for kept_det in kept.iter() {
            if det.class_id == kept_det.class_id {
                let iou = math_utils::calculate_iou(&det.bbox, &kept_det.bbox);
                if iou > nms_threshold {
                    should_keep = false;
                    break;
                }
            }
        }
        if should_keep {
            kept.push(det.clone());
        }
    }

    // 너무 많은 박스는 제한 (성능/가독성)
    if kept.len() > 100 {
        kept.truncate(100);
    }

    *detections = kept;
}

/// RF-DETR 전용 객체 검출기 구현
pub struct RfDetrDetector {
    cache: ModelCache,
}

impl RfDetrDetector {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            cache: ModelCache::new()?,
        })
    }

    pub fn detect(&mut self, image_data: &[u8]) -> anyhow::Result<DetectionResult> {
        detect_objects_with_cache(image_data, &mut self.cache)
    }
}

impl crate::models::ObjectDetector for RfDetrDetector {
    fn detect(&mut self, image_data: &[u8]) -> crate::error::AppResult<DetectionResult> {
        self.detect(image_data)
            .map_err(|e| crate::error::AppError::InferenceError(e.to_string()))
    }
    
    fn get_model_info(&self) -> crate::models::ModelInfo {
        crate::models::ModelInfo::new(
            crate::models::ModelType::RfDetr,
            560,
            90,
            "RF-DETR Transformer 기반 객체 검출 모델".to_string(),
        )
    }
    
    fn model_type(&self) -> crate::models::ModelType {
        crate::models::ModelType::RfDetr
    }
} 