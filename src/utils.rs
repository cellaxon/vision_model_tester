use crate::error::AppResult;
use image::RgbImage;
use std::path::Path;

/// 이미지 관련 유틸리티 함수들
pub mod image_utils {
    use super::*;
    
    /// 이미지 크기 정보
    #[derive(Debug, Clone)]
    pub struct ImageInfo {
        pub width: u32,
        pub height: u32,
        pub aspect_ratio: f32,
    }
    
    /// 이미지 정보 추출
    pub fn get_image_info(image: &RgbImage) -> ImageInfo {
        let width = image.width();
        let height = image.height();
        let aspect_ratio = width as f32 / height as f32;
        
        ImageInfo {
            width,
            height,
            aspect_ratio,
        }
    }
    
    /// 이미지 파일 확장자 확인
    pub fn is_valid_image_extension(path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                let ext_lower = ext_str.to_lowercase();
                return matches!(ext_lower.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "webp");
            }
        }
        false
    }
    
    /// 이미지 파일 크기 확인
    pub fn get_file_size(path: &Path) -> AppResult<u64> {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|e| crate::error::AppError::IoError(e))
    }
}

/// 수학 관련 유틸리티 함수들
pub mod math_utils {
    
    /// 시그모이드 함수
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// IoU (Intersection over Union) 계산
    pub fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
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
    
    /// 두 바운딩 박스의 중심점 거리 계산
    pub fn calculate_center_distance(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let center1_x = (box1[0] + box1[2]) / 2.0;
        let center1_y = (box1[1] + box1[3]) / 2.0;
        let center2_x = (box2[0] + box2[2]) / 2.0;
        let center2_y = (box2[1] + box2[3]) / 2.0;
        
        ((center1_x - center2_x).powi(2) + (center1_y - center2_y).powi(2)).sqrt()
    }
    
    /// 바운딩 박스 면적 계산
    pub fn calculate_bbox_area(bbox: &[f32; 4]) -> f32 {
        let [x1, y1, x2, y2] = bbox;
        (x2 - x1) * (y2 - y1)
    }
}

/// 파일 시스템 관련 유틸리티 함수들
pub mod fs_utils {
    use std::env;
    use crate::error::AppResult;
    use crate::config::CONFIG;
    
    /// 홈 디렉토리 경로 가져오기
    pub fn get_home_directory() -> AppResult<String> {
        env::var("HOME")
            .or_else(|_| env::var("USERPROFILE"))
            .or_else(|_| {
                env::current_dir()
                    .map(|dir| dir.to_string_lossy().to_string())
            })
            .map_err(|_| crate::error::AppError::ConfigError("홈 디렉토리를 찾을 수 없습니다".to_string()))
    }
    
    /// 애플리케이션 데이터 디렉토리 경로 가져오기
    pub fn get_app_data_directory() -> AppResult<std::path::PathBuf> {
        let home_dir = get_home_directory()?;
        let app_dir = std::path::Path::new(&home_dir)
            .join("cellaxon")
            .join("yolov9_onnx_test");
        
        // 디렉토리가 없으면 생성
        if !app_dir.exists() {
            std::fs::create_dir_all(&app_dir)
                .map_err(|e| crate::error::AppError::IoError(e))?;
        }
        
        Ok(app_dir)
    }
    
    /// 데이터베이스 파일 경로 가져오기
    pub fn get_database_path() -> AppResult<std::path::PathBuf> {
        let app_dir = get_app_data_directory()?;
        Ok(app_dir.join(&CONFIG.database.filename))
    }
}

/// 성능 측정 유틸리티 함수들
pub mod perf_utils {
    use std::time::Instant;
    
    /// 성능 측정 구조체
    pub struct PerformanceTimer {
        start_time: Instant,
        name: String,
    }
    
    impl PerformanceTimer {
        /// 새로운 타이머 생성
        pub fn new(name: &str) -> Self {
            Self {
                start_time: Instant::now(),
                name: name.to_string(),
            }
        }
        
        /// 경과 시간 측정 (밀리초)
        pub fn elapsed_ms(&self) -> f64 {
            self.start_time.elapsed().as_secs_f64() * 1000.0
        }
        
        /// 경과 시간 측정 (마이크로초)
        pub fn elapsed_us(&self) -> u64 {
            self.start_time.elapsed().as_micros() as u64
        }
    }
    
    impl Drop for PerformanceTimer {
        fn drop(&mut self) {
            let elapsed = self.elapsed_ms();
            println!("⏱️ {}: {:.2} ms", self.name, elapsed);
        }
    }
    
    /// 함수 실행 시간 측정 매크로
    #[macro_export]
    macro_rules! measure_time {
        ($name:expr, $block:expr) => {{
            let timer = $crate::utils::perf_utils::PerformanceTimer::new($name);
            let result = $block;
            let elapsed = timer.elapsed_ms();
            println!("⏱️ {}: {:.2} ms", $name, elapsed);
            result
        }};
    }
}

/// 색상 관련 유틸리티 함수들
pub mod color_utils {
    use image::Rgb;

    /// 신뢰도에 따른 색상 매핑
    /// 
    /// 신뢰도 범위별 색상:
    /// - 0.9-1.0: 빨간색 (높은 신뢰도)
    /// - 0.7-0.9: 주황색 (중간-높은 신뢰도)
    /// - 0.5-0.7: 노란색 (중간 신뢰도)
    /// - 0.3-0.5: 초록색 (중간-낮은 신뢰도)
    /// - 0.0-0.3: 파란색 (낮은 신뢰도)
    pub fn get_confidence_color(confidence: f32) -> Rgb<u8> {
        match confidence {
            conf if conf >= 0.9 => Rgb([255, 0, 0]),      // 빨간색 (높은 신뢰도)
            conf if conf >= 0.7 => Rgb([255, 165, 0]),     // 주황색 (중간-높은 신뢰도)
            conf if conf >= 0.5 => Rgb([255, 255, 0]),     // 노란색 (중간 신뢰도)
            conf if conf >= 0.3 => Rgb([0, 255, 0]),       // 초록색 (중간-낮은 신뢰도)
            _ => Rgb([0, 0, 255]),                         // 파란색 (낮은 신뢰도)
        }
    }

    /// 신뢰도에 따른 색상 매핑 (그라데이션 방식)
    /// 
    /// 신뢰도를 0-1 범위에서 색상으로 매핑:
    /// - 낮은 신뢰도: 파란색
    /// - 중간 신뢰도: 초록색-노란색
    /// - 높은 신뢰도: 주황색-빨간색
    pub fn get_confidence_color_gradient(confidence: f32) -> Rgb<u8> {
        let clamped_confidence = confidence.max(0.0).min(1.0);
        
        if clamped_confidence < 0.5 {
            // 파란색에서 초록색으로 (0.0-0.5)
            let t = clamped_confidence * 2.0; // 0.0-1.0으로 정규화
            let r = 0;
            let g = (t * 255.0) as u8;
            let b = ((1.0 - t) * 255.0) as u8;
            Rgb([r, g, b])
        } else {
            // 초록색에서 빨간색으로 (0.5-1.0)
            let t = (clamped_confidence - 0.5) * 2.0; // 0.0-1.0으로 정규화
            let r = (t * 255.0) as u8;
            let g = ((1.0 - t) * 255.0) as u8;
            let b = 0;
            Rgb([r, g, b])
        }
    }

    /// 신뢰도에 따른 색상 매핑 (HSV 기반)
    /// 
    /// HSV 색상 공간을 사용하여 더 자연스러운 색상 전환:
    /// - 파란색(240°) → 초록색(120°) → 노란색(60°) → 빨간색(0°)
    pub fn get_confidence_color_hsv(confidence: f32) -> Rgb<u8> {
        let clamped_confidence = confidence.max(0.0).min(1.0);
        
        // HSV에서 H(색조) 계산: 파란색(240°)에서 빨간색(0°)으로
        let hue = 240.0 - (clamped_confidence * 240.0);
        let saturation = 1.0;
        let value = 1.0;
        
        // HSV를 RGB로 변환
        hsv_to_rgb(hue, saturation, value)
    }

    /// HSV를 RGB로 변환하는 함수
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb<u8> {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = match (h / 60.0) as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };
        
        Rgb([
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        ])
    }
}
