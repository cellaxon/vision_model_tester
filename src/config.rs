use serde::{Deserialize, Serialize};

/// 애플리케이션 설정 구조체
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// 모델 관련 설정
    pub model: ModelConfig,
    /// 추론 관련 설정
    pub inference: InferenceConfig,
    /// UI 관련 설정
    pub ui: UiConfig,
    /// 데이터베이스 관련 설정
    pub database: DatabaseConfig,
}

/// 모델 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 기본 모델 입력 크기
    pub input_size: u32,
    /// 기본 신뢰도 임계값
    pub default_confidence_threshold: f32,
    /// 기본 NMS 임계값
    pub default_nms_threshold: f32,
    /// 파싱 시 기본 최소 신뢰도
    pub base_parse_confidence: f32,
}

/// 추론 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// 최대 검출 개수
    pub max_detections: usize,
    /// 최소 바운딩 박스 면적 (전체 이미지 대비)
    pub min_bbox_area: f32,
    /// 최대 바운딩 박스 면적 (전체 이미지 대비)
    pub max_bbox_area: f32,
    /// 중심점 거리 임계값
    pub center_distance_threshold: f32,
    /// IoU 임계값 (중복 제거용)
    pub iou_threshold_for_duplicates: f32,
}

/// UI 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// 마우스 휠 줌 변화량 (로그 공간)
    pub mouse_wheel_zoom_delta: f32,
    /// 키보드 줌 변화량 (로그 공간)
    pub keyboard_zoom_delta: f32,
    /// 최소 줌 로그값
    pub min_zoom_log: f32,
    /// 최대 줌 로그값
    pub max_zoom_log: f32,
    /// 기본 줌 레벨
    pub default_zoom: f32,
    /// 기본 신뢰도 임계값
    pub default_confidence_threshold: f32,
    /// 기본 NMS 임계값
    pub default_nms_threshold: f32,
    /// 바운딩 박스 색상 매핑 방식
    pub bounding_box_color_mode: ColorMappingMode,
}

/// 바운딩 박스 색상 매핑 방식
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColorMappingMode {
    /// 고정 색상 (기본 빨간색)
    Fixed,
    /// 범위별 색상 (5단계)
    RangeBased,
    /// 그라데이션 색상 (선형)
    Gradient,
    /// HSV 기반 색상 (자연스러운 전환)
    HsvBased,
}

/// 데이터베이스 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// 데이터베이스 파일명
    pub filename: String,
    /// 캐시 정리 기준 일수
    pub cleanup_days: i32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            inference: InferenceConfig::default(),
            ui: UiConfig::default(),
            database: DatabaseConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            default_confidence_threshold: 0.6,
            default_nms_threshold: 0.2,
            base_parse_confidence: 0.05,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_detections: 50,
            min_bbox_area: 0.005,
            max_bbox_area: 0.95,
            center_distance_threshold: 0.1,
            iou_threshold_for_duplicates: 0.1,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            mouse_wheel_zoom_delta: 0.02,
            keyboard_zoom_delta: 0.05,
            min_zoom_log: -2.3, // ln(0.1)
            max_zoom_log: 3.0,  // ln(20.0)
            default_zoom: 1.0,
            default_confidence_threshold: 0.6,
            default_nms_threshold: 0.2,
            bounding_box_color_mode: ColorMappingMode::Fixed,
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            filename: "inference_cache.db".to_string(),
            cleanup_days: 30,
        }
    }
}

/// 전역 설정 인스턴스
pub static CONFIG: once_cell::sync::Lazy<AppConfig> = once_cell::sync::Lazy::new(|| {
    // TODO: 설정 파일에서 로드하거나 기본값 사용
    AppConfig::default()
});
