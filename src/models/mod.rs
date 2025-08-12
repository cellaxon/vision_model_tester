// 모델 모듈 통합 관리
pub mod yolov9;
pub mod rf_detr;

// 공통 모델 인터페이스
use crate::error::AppResult;
use image::RgbImage;

// Detection 타입을 yolov9에서 가져오기 (공통 사용)
pub use yolov9::Detection;

/// 객체 검출 결과를 나타내는 구조체 (추론 시간 포함)
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub detections: Vec<Detection>,
    pub result_image: RgbImage,
    pub inference_time_ms: f64,
}

/// 객체 검출 모델 타입
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    /// YOLOv9 모델 (gelan-c, gelan-e, yolov9-c, yolov9-e)
    YoloV9(String),
    /// RF-DETR 모델 (원본)
    RfDetr,
}

impl ModelType {
    /// 모델 타입을 문자열로 변환
    pub fn to_string(&self) -> String {
        match self {
            ModelType::YoloV9(name) => format!("YOLOv9-{}", name),
            ModelType::RfDetr => "RF-DETR".to_string(),
        }
    }

    /// 모델 타입을 표시용 이름으로 변환
    pub fn display_name(&self) -> String {
        match self {
            ModelType::YoloV9(name) => {
                let model_name = if name.contains("gelan-e") {
                    "GELAN-E"
                } else if name.contains("gelan-c") {
                    "GELAN-C"
                } else if name.contains("yolov9-e") {
                    "YOLOv9-E"
                } else if name.contains("yolov9-c") {
                    "YOLOv9-C"
                } else {
                    "Unknown"
                };
                format!("{} ({})", model_name, name)
            }
            ModelType::RfDetr => "RF-DETR (Original)".to_string(),
        }
    }

    /// 모델 입력 크기 반환
    pub fn input_size(&self) -> u32 {
        match self {
            ModelType::YoloV9(_) => 640,
            ModelType::RfDetr => 560,
        }
    }
}

/// 객체 검출기 공통 인터페이스
pub trait ObjectDetector {
    /// 이미지에서 객체 검출 수행
    fn detect(&mut self, image_data: &[u8]) -> AppResult<DetectionResult>;
    
    /// 모델 정보 반환
    fn get_model_info(&self) -> ModelInfo;
    
    /// 모델 타입 반환
    fn model_type(&self) -> ModelType;
}

/// 모델 정보 구조체
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub input_size: u32,
    pub class_count: u32,
    pub description: String,
}

impl ModelInfo {
    pub fn new(model_type: ModelType, input_size: u32, class_count: u32, description: String) -> Self {
        Self {
            model_type,
            input_size,
            class_count,
            description,
        }
    }
}

// 모델별 구현을 외부로 노출
pub use yolov9::YoloV9Detector;
pub use rf_detr::RfDetrDetector;

// 통합 추론 엔진
pub struct UnifiedInferenceEngine {
    yolov9_cache: Option<yolov9::ModelCache>,
    rf_detr_cache: Option<rf_detr::ModelCache>,
    current_model: ModelType,
}

impl UnifiedInferenceEngine {
    /// 새로운 통합 추론 엔진 생성
    pub fn new() -> AppResult<Self> {
        Ok(Self {
            yolov9_cache: None,
            rf_detr_cache: None,
            current_model: ModelType::YoloV9("gelan-e.onnx".to_string()), // 기본값
        })
    }

    /// 모델 타입 설정
    pub fn set_model(&mut self, model_type: ModelType) -> AppResult<()> {
        self.current_model = model_type.clone();
        
        // 모델별 캐시 초기화
        match &self.current_model {
            ModelType::YoloV9(_) => {
                if self.yolov9_cache.is_none() {
                    self.yolov9_cache = Some(yolov9::ModelCache::new()
                        .map_err(|e| crate::error::AppError::ModelError(e.to_string()))?);
                }
            }
            ModelType::RfDetr => {
                if self.rf_detr_cache.is_none() {
                    self.rf_detr_cache = Some(rf_detr::ModelCache::new()
                        .map_err(|e| crate::error::AppError::ModelError(e.to_string()))?);
                }
            }
        }
        
        Ok(())
    }

    /// 현재 모델로 객체 검출 수행
    pub fn detect(&mut self, image_data: &[u8]) -> AppResult<DetectionResult> {
        match &self.current_model {
            ModelType::YoloV9(model_file) => {
                if let Some(cache) = &mut self.yolov9_cache {
                    yolov9::detect_objects_with_cache(image_data, cache, model_file)
                        .map_err(|e| crate::error::AppError::InferenceError(e.to_string()))
                } else {
                    Err(crate::error::AppError::ModelError("YOLOv9 캐시가 초기화되지 않았습니다".to_string()))
                }
            }
            ModelType::RfDetr => {
                if let Some(cache) = &mut self.rf_detr_cache {
                    rf_detr::detect_objects_with_cache(image_data, cache)
                        .map_err(|e| crate::error::AppError::InferenceError(e.to_string()))
                } else {
                    Err(crate::error::AppError::ModelError("RF-DETR 캐시가 초기화되지 않았습니다".to_string()))
                }
            }
        }
    }

    /// 현재 모델 정보 반환
    pub fn get_current_model_info(&self) -> ModelInfo {
        match &self.current_model {
            ModelType::YoloV9(model_file) => {
                let (name, input_size) = yolov9::get_model_info(model_file);
                ModelInfo::new(
                    self.current_model.clone(),
                    input_size,
                    80, // COCO classes
                    format!("YOLOv9 기반 모델: {}", name),
                )
            }
            ModelType::RfDetr => {
                ModelInfo::new(
                    self.current_model.clone(),
                    560,
                    90, // COCO classes (1-90, background 제외)
                    "RF-DETR Transformer 기반 객체 검출 모델".to_string(),
                )
            }
        }
    }

    /// 사용 가능한 모델 목록 반환
    pub fn get_available_models() -> Vec<ModelType> {
        let mut models = Vec::new();
        
        // YOLOv9 모델들
        for model_file in yolov9::get_embedded_model_list() {
            models.push(ModelType::YoloV9(model_file));
        }
        
        // RF-DETR 모델
        models.push(ModelType::RfDetr);
        
        models
    }
} 