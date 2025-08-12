// 모듈 선언
pub mod config;
pub mod error;
pub mod utils;
pub mod models;
pub mod database;

// 공통 구조체들을 models 모듈에서 가져오기
use models::{DetectionResult, ModelType, UnifiedInferenceEngine};

// Detection 타입은 별도로 정의 (YOLOv9와 RF-DETR에서 공통 사용)
pub use models::yolov9::Detection;

// 기존 호환성을 위한 재export
pub use models::yolov9::{get_embedded_model_list, get_model_info, ModelCache, InferenceDb};
pub use models::rf_detr::RfDetrDetector;

// 새로운 데이터베이스 시스템
pub use database::EnhancedInferenceDb;

use std::sync::{Mutex, OnceLock};

/// 통합 추론 엔진 인스턴스 (전역, thread-safe)
static INFERENCE_ENGINE: OnceLock<Mutex<UnifiedInferenceEngine>> = OnceLock::new();

/// 전역 추론 엔진 초기화
pub fn init_inference_engine() -> crate::error::AppResult<()> {
    let engine = UnifiedInferenceEngine::new()?;
    INFERENCE_ENGINE.set(Mutex::new(engine))
        .map_err(|_| crate::error::AppError::ModelError("추론 엔진이 이미 초기화되었습니다".to_string()))?;
    Ok(())
}

/// 전역 추론 엔진에서 객체 검출 수행
pub fn detect_with_global_engine(image_data: &[u8], model_type: ModelType) -> crate::error::AppResult<DetectionResult> {
    let engine_mutex = INFERENCE_ENGINE.get()
        .ok_or(crate::error::AppError::ModelError("추론 엔진이 초기화되지 않았습니다".to_string()))?;
    
    let mut engine = engine_mutex.lock()
        .map_err(|_| crate::error::AppError::ModelError("추론 엔진 lock 실패".to_string()))?;
    
    engine.set_model(model_type)?;
    engine.detect(image_data)
}

/// 통합 객체 검출 함수
pub fn detect_objects_unified(
    image_data: &[u8],
    model_type: ModelType,
) -> crate::error::AppResult<DetectionResult> {
    detect_with_global_engine(image_data, model_type)
}

/// 기존 호환성을 위한 함수들
pub fn detect_objects(image_data: &[u8]) -> crate::error::AppResult<DetectionResult> {
    // 기본 모델로 검출 (YOLOv9 GELAN-E)
    detect_objects_unified(image_data, ModelType::YoloV9("gelan-e.onnx".to_string()))
}

/// Pre-NMS 데이터를 캐시하고 NMS만 재적용하는 함수
pub fn apply_nms_only(
    mut detections: Vec<Detection>,
    nms_threshold: f32,
) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
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
                let iou = utils::math_utils::calculate_iou(&det.bbox, &kept_det.bbox);
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

    kept
}

/// 캐시된 Pre-NMS 데이터를 로드하고 NMS를 적용하는 함수
pub fn load_or_infer_pre_nms(
    image_path: &str,
    model_file_name: &str,
    image_data: &[u8],
    force_reinfer: bool,
) -> crate::error::AppResult<Vec<Detection>> {
    if !force_reinfer {
        // 캐시에서 로드 시도
        if let Ok(db) = InferenceDb::new() {
            if let Ok(Some(cached_detections)) = db.load_pre_nms_detections(image_path, model_file_name) {
                return Ok(cached_detections);
            }
        }
    }

    // 캐시가 없거나 강제 재추론인 경우
    let model_type = if model_file_name.contains("rf_detr") {
        ModelType::RfDetr
    } else {
        ModelType::YoloV9(model_file_name.to_string())
    };

    let result = detect_objects_unified(image_data, model_type)?;
    
    // Pre-NMS 데이터를 캐시에 저장
    if let Ok(db) = InferenceDb::new() {
        let _ = db.save_pre_nms_detections(
            image_path,
            image_data,
            model_file_name,
            &result.detections,
        );
    }

    Ok(result.detections)
}

/// 모델 정보 가져오기 (통합)
pub fn get_unified_model_info(model_type: &ModelType) -> (String, u32) {
    match model_type {
        ModelType::YoloV9(file_name) => get_model_info(file_name),
        ModelType::RfDetr => ("RF-DETR Original".to_string(), 560),
    }
}

/// 사용 가능한 모델 목록 가져오기 (통합)
pub fn get_all_available_models() -> Vec<ModelType> {
    UnifiedInferenceEngine::get_available_models()
}

/// 모델별 기본 설정값 가져오기
pub fn get_model_default_settings(model_type: &ModelType) -> (f32, f32) {
    match model_type {
        ModelType::YoloV9(_) => (0.6, 0.2), // confidence, nms
        ModelType::RfDetr => (0.5, 0.2),     // confidence, nms
    }
}

/// 모델별 입력 크기 가져오기
pub fn get_model_input_size(model_type: &ModelType) -> u32 {
    model_type.input_size()
}

/// 모델별 클래스 수 가져오기
pub fn get_model_class_count(model_type: &ModelType) -> u32 {
    match model_type {
        ModelType::YoloV9(_) => 80,  // COCO 80 classes
        ModelType::RfDetr => 90,      // COCO 90 classes (1-90, background 제외)
    }
}

/// 모델별 설명 가져오기
pub fn get_model_description(model_type: &ModelType) -> String {
    match model_type {
        ModelType::YoloV9(file_name) => {
            let (name, _) = get_model_info(file_name);
            format!("YOLOv9 기반 모델: {}", name)
        }
        ModelType::RfDetr => "RF-DETR Transformer 기반 객체 검출 모델".to_string(),
    }
}

/// 모델 성능 비교 정보
pub fn get_model_performance_info(model_type: &ModelType) -> (String, String) {
    match model_type {
        ModelType::YoloV9(file_name) => {
            if file_name.contains("gelan-c") {
                ("매우 빠름".to_string(), "높음".to_string())
            } else if file_name.contains("gelan-e") {
                ("빠름".to_string(), "높음".to_string())
            } else if file_name.contains("yolov9-c") {
                ("보통".to_string(), "매우 높음".to_string())
            } else if file_name.contains("yolov9-e") {
                ("느림".to_string(), "최고".to_string())
            } else {
                ("보통".to_string(), "높음".to_string())
            }
        }
        ModelType::RfDetr => ("보통".to_string(), "매우 높음".to_string()),
    }
}

/// 모델별 권장 사용 사례
pub fn get_model_recommended_use(model_type: &ModelType) -> String {
    match model_type {
        ModelType::YoloV9(file_name) => {
            if file_name.contains("gelan-c") {
                "실시간 처리, 모바일, 임베디드".to_string()
            } else if file_name.contains("gelan-e") {
                "일반적인 실시간 처리, 웹캠".to_string()
            } else if file_name.contains("yolov9-c") {
                "정확도와 속도 균형, 범용".to_string()
            } else if file_name.contains("yolov9-e") {
                "고정밀 검출, 연구, 분석".to_string()
            } else {
                "일반적인 객체 검출".to_string()
            }
        }
        ModelType::RfDetr => "Transformer 기반 정밀 검출, 연구, 고품질 요구".to_string(),
    }
}

/// 모델 초기화 및 검증
pub fn initialize_models() -> crate::error::AppResult<()> {
    // 추론 엔진 초기화
    init_inference_engine()?;
    
    // 사용 가능한 모델 목록 확인
    let available_models = get_all_available_models();
    println!("사용 가능한 모델: {}개", available_models.len());
    
    for model in &available_models {
        let (name, input_size) = get_unified_model_info(model);
        let (speed, accuracy) = get_model_performance_info(model);
        let recommended = get_model_recommended_use(model);
        
        println!("  - {} ({}x{}): 속도={}, 정확도={}", 
                name, input_size, input_size, speed, accuracy);
        println!("    권장: {}", recommended);
    }
    
    Ok(())
}

/// 모델별 메모리 사용량 추정 (MB)
pub fn estimate_model_memory_usage(model_type: &ModelType) -> u32 {
    match model_type {
        ModelType::YoloV9(file_name) => {
            if file_name.contains("gelan-c") {
                102
            } else if file_name.contains("gelan-e") {
                233
            } else if file_name.contains("yolov9-c") {
                205
            } else if file_name.contains("yolov9-e") {
                278
            } else {
                200
            }
        }
        ModelType::RfDetr => 108,
    }
}

/// 모델별 추론 시간 추정 (ms, M4 Mac 기준)
pub fn estimate_inference_time(model_type: &ModelType) -> (u32, u32) {
    match model_type {
        ModelType::YoloV9(file_name) => {
            if file_name.contains("gelan-c") {
                (150, 200)
            } else if file_name.contains("gelan-e") {
                (300, 400)
            } else if file_name.contains("yolov9-c") {
                (200, 300)
            } else if file_name.contains("yolov9-e") {
                (400, 500)
            } else {
                (250, 350)
            }
        }
        ModelType::RfDetr => (350, 450),
    }
}
