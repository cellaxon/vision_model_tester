use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::CowArray;
use ndarray::{ArrayD, IxDyn};
use ort::execution_providers::CPUExecutionProviderOptions;
use ort::{Environment, ExecutionProvider, SessionBuilder, Value};
use std::sync::Arc;

// 상수 정의
const MODEL_INPUT_SIZE: u32 = 640; // YOLOv9-c는 640x640 입력 사용
const CONFIDENCE_THRESHOLD: f32 = 0.6; // 신뢰도 임계값을 더 높임
const NMS_THRESHOLD: f32 = 0.2; // NMS 임계값을 더 낮춤
const BBOX_COLOR: Rgb<u8> = Rgb([255, 0, 0]); // 빨간색

// 임베디드 리소스: assets/models 폴더의 모든 onnx 파일을 임베딩 (분리된 모듈)
mod models;
use models::get_embedded_model_bytes;
pub use models::get_embedded_model_list;

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
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub bbox: [f32; 4], // [x1, y1, x2, y2] in normalized coordinates (0-1)
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
}

/// 검출 결과를 나타내는 구조체 (추론 시간 포함)
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub result_image: RgbImage,
    pub inference_time_ms: f64,
}

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

/// IoU (Intersection over Union) 계산
fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    intersection / union
}

/// Non-Maximum Suppression (NMS) 구현
fn non_maximum_suppression(detections: &mut Vec<Detection>, nms_threshold: f32) {
    if detections.is_empty() {
        return;
    }
    
    // 신뢰도 기준으로 정렬 (높은 신뢰도가 먼저)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
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
                let iou = calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > nms_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    // 유지할 검출 결과만 남기기
    let mut new_detections = Vec::new();
    for &idx in &keep {
        new_detections.push(detections[idx].clone());
    }
    
    *detections = new_detections;
    
    // 최대 검출 개수 제한 (성능 향상)
    if detections.len() > 50 {
        detections.truncate(50);
    }
    
    // 추가 후처리: 작은 바운딩 박스 제거 및 중복 제거
    post_process_detections(detections);
}

/// 추가 후처리: 작은 박스 제거 및 중복 제거
fn post_process_detections(detections: &mut Vec<Detection>) {
    if detections.is_empty() {
        return;
    }
    
    // 1. 너무 작은 바운딩 박스 제거
    detections.retain(|det| {
        let [x1, y1, x2, y2] = det.bbox;
        let width = x2 - x1;
        let height = y2 - y1;
        let area = width * height;
        
        // 최소 면적 기준 (전체 이미지의 0.5% 이상)
        area > 0.005
    });
    
    // 2. 같은 클래스 내에서 매우 유사한 위치의 박스 제거
    let mut to_remove = Vec::new();
    
    for i in 0..detections.len() {
        for j in (i + 1)..detections.len() {
            if detections[i].class_id == detections[j].class_id {
                let iou = calculate_iou(&detections[i].bbox, &detections[j].bbox);
                
                // IoU가 높거나 중심점이 매우 가까운 경우 제거
                if iou > 0.1 || centers_are_close(&detections[i].bbox, &detections[j].bbox) {
                    // 신뢰도가 낮은 것을 제거
                    if detections[i].confidence > detections[j].confidence {
                        to_remove.push(j);
                    } else {
                        to_remove.push(i);
                    }
                }
            }
        }
    }
    
    // 중복 제거 (인덱스 순서대로 제거)
    to_remove.sort();
    to_remove.reverse();
    for &idx in &to_remove {
        if idx < detections.len() {
            detections.remove(idx);
        }
    }
}

/// 두 바운딩 박스의 중심점이 가까운지 확인
fn centers_are_close(box1: &[f32; 4], box2: &[f32; 4]) -> bool {
    let center1_x = (box1[0] + box1[2]) / 2.0;
    let center1_y = (box1[1] + box1[3]) / 2.0;
    let center2_x = (box2[0] + box2[2]) / 2.0;
    let center2_y = (box2[1] + box2[3]) / 2.0;
    
    let distance = ((center1_x - center2_x).powi(2) + (center1_y - center2_y).powi(2)).sqrt();
    
    // 중심점 거리가 0.1 이하이면 가깝다고 판단
    distance < 0.1
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

/// 시그모이드 함수
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    let (num_boxes, num_classes) = if shape[1] == 84 {
        // YOLOv9-C: (1, 84, 8400) 형태
        (shape[2], 80)
    } else if shape[1] == 85 {
        // YOLOv9 with objectness: (1, 85, N) 형태
        (shape[2], 80)
    } else {
        // 기타 형태 시도
        println!("⚠️ Unexpected shape: {:?}, trying alternative parsing", shape);
        (shape[2], shape[1] - 4)
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
        if cx_norm < 0.0 || cx_norm > 1.0 || cy_norm < 0.0 || cy_norm > 1.0 {
            continue;
        }
        
        // 클래스 확률 계산 (시그모이드 적용)
        let mut max_conf = 0.0;
        let mut best_class = 0;
        
        for class_idx in 0..num_classes {
            let raw_score = scores[[class_idx, box_idx]];
            // 점수 스케일링 (매우 작은 값들을 확대)
            let scaled_score = raw_score * 1000.0; // 스케일링 팩터
            let conf = sigmoid(scaled_score); // 시그모이드 적용
            
            if conf > max_conf {
                max_conf = conf;
                best_class = class_idx;
            }
        }
        
        // 신뢰도 임계값 확인 (GUI에서 설정한 값 사용)
        if max_conf > confidence_threshold {
            // 정규화된 좌표로 center_x, center_y, width, height -> x1, y1, x2, y2 변환
            let x1 = (cx_norm - w_norm / 2.0).max(0.0).min(1.0);
            let y1 = (cy_norm - h_norm / 2.0).max(0.0).min(1.0);
            let x2 = (cx_norm + w_norm / 2.0).max(0.0).min(1.0);
            let y2 = (cy_norm + h_norm / 2.0).max(0.0).min(1.0);
            
            // 더 엄격한 바운딩 박스 유효성 검증
            if x1 >= x2 || y1 >= y2 || 
               (x2 - x1) < 0.02 || (y2 - y1) < 0.02 ||  // 최소 크기 증가
               (x2 - x1) > 0.95 || (y2 - y1) > 0.95 {   // 최대 크기 제한 (전체 이미지 제외)
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

/// 검출된 객체에 바운딩 박스 그리기
pub fn draw_detections(image: &mut RgbImage, detections: &[Detection]) {
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
        if x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 || x2 > image.width() as i32 || y2 > image.height() as i32 {
            continue;
        }

        let rect = Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32);
        draw_hollow_rect_mut(image, rect, BBOX_COLOR);
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

    pub fn get_session(&mut self, model_file_name: &str) -> anyhow::Result<&ort::InMemorySession<'static>> {
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
            println!("Loading model: {} - Optimized for inference", model_file_name);
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

/// 메인 객체 검출 함수 (캐시 사용)
pub fn detect_objects_with_cache(
    image_data: &[u8],
    cache: &mut ModelCache,
    model_file_name: &str,
) -> anyhow::Result<DetectionResult> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // 캐시된 세션 가져오기 (선택된 모델 기준)
    let session = cache.get_session(model_file_name)?;

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

    // 원본 이미지를 그대로 반환 (bbox는 GUI에서 오버레이로 렌더링)
    let result_image = img.clone();

    Ok(DetectionResult {
        detections,
        result_image,
        inference_time_ms,
    })
}

/// 메인 객체 검출 함수 (기본 모델 사용)
pub fn detect_objects(image_data: &[u8]) -> anyhow::Result<DetectionResult> {
    // ModelCache를 생성하여 사용
    let mut cache = ModelCache::new()?;
    detect_objects_with_cache(image_data, &mut cache, "gelan-e.onnx")
}

/// 설정 가능한 임계값으로 객체 검출 함수
pub fn detect_objects_with_settings(
    image_data: &[u8],
    cache: &mut ModelCache,
    model_file_name: &str,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<DetectionResult> {
    // 이미지 로드
    let img = ImageReader::new(std::io::Cursor::new(image_data))
        .with_guessed_format()?
        .decode()?
        .to_rgb8();

    // 캐시된 세션 가져오기 (선택된 모델 기준)
    let session = cache.get_session(model_file_name)?;

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

        // YOLOv9 출력 파싱 (설정된 임계값 사용)
        detections = parse_yolov9_outputs(&output_view, img.width(), img.height(), confidence_threshold, nms_threshold)?;
    }

    // 원본 이미지를 그대로 반환 (bbox는 GUI에서 오버레이로 렌더링)
    let result_image = img.clone();

    Ok(DetectionResult {
        detections,
        result_image,
        inference_time_ms,
    })
}


