use thiserror::Error;

/// 애플리케이션 에러 타입
#[derive(Error, Debug)]
pub enum AppError {
    #[error("이미지 처리 오류: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("ONNX 런타임 오류: {0}")]
    OrtError(String),

    #[error("데이터베이스 오류: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    #[error("JSON 직렬화 오류: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("파일 시스템 오류: {0}")]
    IoError(#[from] std::io::Error),

    #[error("모델 로드 오류: {0}")]
    ModelError(String),

    #[error("추론 오류: {0}")]
    InferenceError(String),

    #[error("설정 오류: {0}")]
    ConfigError(String),

    #[error("유효하지 않은 입력: {0}")]
    ValidationError(String),

    #[error("캐시 오류: {0}")]
    CacheError(String),
}

/// 결과 타입 별칭
pub type AppResult<T> = Result<T, AppError>;

/// 에러 컨텍스트를 위한 확장 트레이트
pub trait ErrorContext<T> {
    fn with_context<C>(self, context: C) -> AppResult<T>
    where
        C: std::fmt::Display;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<AppError>,
{
    fn with_context<C>(self, context: C) -> AppResult<T>
    where
        C: std::fmt::Display,
    {
        self.map_err(|e| {
            let app_error: AppError = e.into();
            match app_error {
                AppError::ModelError(msg) => AppError::ModelError(format!("{}: {}", context, msg)),
                AppError::InferenceError(msg) => {
                    AppError::InferenceError(format!("{}: {}", context, msg))
                }
                AppError::ConfigError(msg) => {
                    AppError::ConfigError(format!("{}: {}", context, msg))
                }
                AppError::ValidationError(msg) => {
                    AppError::ValidationError(format!("{}: {}", context, msg))
                }
                AppError::CacheError(msg) => AppError::CacheError(format!("{}: {}", context, msg)),
                _ => app_error,
            }
        })
    }
}

/// 유효성 검사 헬퍼 함수들
pub mod validation {
    use super::*;

    /// 이미지 데이터 유효성 검사
    pub fn validate_image_data(data: &[u8]) -> AppResult<()> {
        if data.is_empty() {
            return Err(AppError::ValidationError("빈 이미지 데이터".to_string()));
        }
        Ok(())
    }

    /// 신뢰도 임계값 유효성 검사
    pub fn validate_confidence_threshold(threshold: f32) -> AppResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(AppError::ValidationError(format!(
                "신뢰도 임계값은 0.0과 1.0 사이여야 합니다: {}",
                threshold
            )));
        }
        Ok(())
    }

    /// NMS 임계값 유효성 검사
    pub fn validate_nms_threshold(threshold: f32) -> AppResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(AppError::ValidationError(format!(
                "NMS 임계값은 0.0과 1.0 사이여야 합니다: {}",
                threshold
            )));
        }
        Ok(())
    }

    /// 줌 레벨 유효성 검사
    pub fn validate_zoom_level(zoom: f32) -> AppResult<()> {
        if zoom <= 0.0 {
            return Err(AppError::ValidationError(format!(
                "줌 레벨은 0보다 커야 합니다: {}",
                zoom
            )));
        }
        Ok(())
    }
}
