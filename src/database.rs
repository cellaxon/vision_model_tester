// 개선된 데이터베이스 구조 및 관리
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AppError, AppResult};
use crate::models::{Detection, DetectionResult, ModelType};
use crate::utils::fs_utils;

/// 이미지 메타데이터
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: i64,
    pub path: String,
    pub hash: String,
    pub width: i32,
    pub height: i32,
    pub file_size: i64,
    pub created_at: String,
    pub last_accessed: String,
}

/// 모델 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: i64,
    pub name: String,
    pub file_name: String,
    pub input_width: i32,
    pub input_height: i32,
    pub class_count: i32,
    pub model_type: String,
    pub version: String,
    pub created_at: String,
}

/// 추론 세션 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSession {
    pub id: i64,
    pub image_id: i64,
    pub model_id: i64,
    pub inference_time_ms: f64,
    pub detection_count: i32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub created_at: String,
}

/// 검출 결과 (모델 공통)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionRecord {
    pub id: i64,
    pub session_id: i64,
    pub class_id: i32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox_x1: f32,
    pub bbox_y1: f32,
    pub bbox_x2: f32,
    pub bbox_y2: f32,
    pub detection_order: i32,
    pub query_index: Option<i32>, // RF-DETR 전용
}

/// 개선된 데이터베이스 관리자
pub struct EnhancedInferenceDb {
    conn: Connection,
}

impl EnhancedInferenceDb {
    /// 새로운 데이터베이스 인스턴스 생성
    pub fn new() -> AppResult<Self> {
        let db_path = fs_utils::get_database_path()?;
        let conn = Connection::open(&db_path).map_err(AppError::DatabaseError)?;
        
        let db = Self { conn };
        db.initialize_schema()?;
        db.initialize_default_models()?;
        
        Ok(db)
    }

    /// 스키마 초기화
    fn initialize_schema(&self) -> AppResult<()> {
        let schema = r#"
-- 개선된 데이터베이스 스키마
-- 모델별 테이블을 생성하여 효율적인 저장 및 쿼리 지원

-- 1. 이미지 메타데이터 테이블 (공통)
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. 모델 정보 테이블
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL UNIQUE,
    input_width INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    class_count INTEGER NOT NULL,
    model_type TEXT NOT NULL, -- 'yolov9' or 'rf_detr'
    version TEXT DEFAULT '1.0',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 3. 추론 세션 테이블 (추론 실행 정보)
CREATE TABLE IF NOT EXISTS inference_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    inference_time_ms REAL NOT NULL,
    detection_count INTEGER NOT NULL,
    confidence_threshold REAL NOT NULL,
    nms_threshold REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    UNIQUE(image_id, model_id)
);

-- 4. YOLOv9 검출 결과 테이블
CREATE TABLE IF NOT EXISTS yolov9_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL, -- 0-79 (COCO 80 classes)
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL, -- normalized coordinates (0-1)
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    detection_order INTEGER NOT NULL, -- 원본 검출 순서
    FOREIGN KEY (session_id) REFERENCES inference_sessions(id) ON DELETE CASCADE
);

-- 5. RF-DETR 검출 결과 테이블
CREATE TABLE IF NOT EXISTS rf_detr_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL, -- 1-90 (COCO 90 classes, background 제외)
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL, -- normalized coordinates (0-1)
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    detection_order INTEGER NOT NULL, -- 원본 검출 순서
    query_index INTEGER, -- RF-DETR specific: which query produced this detection
    FOREIGN KEY (session_id) REFERENCES inference_sessions(id) ON DELETE CASCADE
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);

CREATE INDEX IF NOT EXISTS idx_sessions_image_model ON inference_sessions(image_id, model_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON inference_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_yolov9_session ON yolov9_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_yolov9_class ON yolov9_detections(class_id);
CREATE INDEX IF NOT EXISTS idx_yolov9_confidence ON yolov9_detections(confidence);

CREATE INDEX IF NOT EXISTS idx_rf_detr_session ON rf_detr_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_rf_detr_class ON rf_detr_detections(class_id);
CREATE INDEX IF NOT EXISTS idx_rf_detr_confidence ON rf_detr_detections(confidence);

-- 뷰 생성 (쿼리 편의성)
CREATE VIEW IF NOT EXISTS detection_summary AS
SELECT 
    i.path as image_path,
    m.name as model_name,
    m.model_type,
    s.inference_time_ms,
    s.detection_count,
    s.created_at
FROM inference_sessions s
JOIN images i ON s.image_id = i.id
JOIN models m ON s.model_id = m.id
ORDER BY s.created_at DESC;

-- 통계 뷰
CREATE VIEW IF NOT EXISTS model_performance_stats AS
SELECT 
    m.name as model_name,
    m.model_type,
    COUNT(s.id) as total_inferences,
    AVG(s.inference_time_ms) as avg_inference_time,
    AVG(s.detection_count) as avg_detections,
    MIN(s.created_at) as first_used,
    MAX(s.created_at) as last_used
FROM models m
LEFT JOIN inference_sessions s ON m.id = s.model_id
GROUP BY m.id, m.name, m.model_type;
        "#;
        
        self.conn.execute_batch(schema).map_err(AppError::DatabaseError)?;
        Ok(())
    }

    /// 기본 모델 정보 초기화
    fn initialize_default_models(&self) -> AppResult<()> {
        let models = [
            ("YOLOv9-GELAN-C", "yolov9/gelan-c.onnx", 640, 640, 80, "yolov9"),
            ("YOLOv9-GELAN-E", "yolov9/gelan-e.onnx", 640, 640, 80, "yolov9"),
            ("YOLOv9-C", "yolov9/yolov9-c.onnx", 640, 640, 80, "yolov9"),
            ("YOLOv9-E", "yolov9/yolov9-e.onnx", 640, 640, 80, "yolov9"),
            ("RF-DETR", "rf-detr/rf-detr.onnx", 560, 560, 90, "rf_detr"),
        ];

        for (name, file_name, width, height, classes, model_type) in models {
            self.conn.execute(
                "INSERT OR IGNORE INTO models (name, file_name, input_width, input_height, class_count, model_type)
                 VALUES (?, ?, ?, ?, ?, ?)",
                params![name, file_name, width, height, classes, model_type],
            ).map_err(AppError::DatabaseError)?;
        }

        Ok(())
    }

    /// 이미지 해시 계산
    pub fn calculate_image_hash(image_data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(image_data);
        format!("{:x}", hasher.finalize())
    }

    /// 이미지 정보 저장 또는 업데이트
    pub fn save_or_update_image(&self, path: &str, image_data: &[u8], width: i32, height: i32) -> AppResult<i64> {
        let hash = Self::calculate_image_hash(image_data);
        let file_size = image_data.len() as i64;

        // 기존 이미지 확인
        let existing_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM images WHERE hash = ?",
            params![hash],
            |row| row.get(0),
        ).optional().map_err(AppError::DatabaseError)?;

        if let Some(id) = existing_id {
            // 접근 시간 업데이트
            self.conn.execute(
                "UPDATE images SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                params![id],
            ).map_err(AppError::DatabaseError)?;
            Ok(id)
        } else {
            // 새 이미지 삽입
            self.conn.execute(
                "INSERT INTO images (path, hash, width, height, file_size) VALUES (?, ?, ?, ?, ?)",
                params![path, hash, width, height, file_size],
            ).map_err(AppError::DatabaseError)?;
            
            Ok(self.conn.last_insert_rowid())
        }
    }

    /// 모델 ID 가져오기
    pub fn get_model_id(&self, model_type: &ModelType) -> AppResult<i64> {
        let file_name = match model_type {
            ModelType::YoloV9(name) => format!("yolov9/{}", name),
            ModelType::RfDetr => "rf-detr/rf-detr.onnx".to_string(),
        };

        let id: i64 = self.conn.query_row(
            "SELECT id FROM models WHERE file_name = ?",
            params![file_name],
            |row| row.get(0),
        ).map_err(AppError::DatabaseError)?;

        Ok(id)
    }

    /// 추론 결과 저장
    pub fn save_inference_result(
        &self,
        image_path: &str,
        image_data: &[u8],
        image_width: i32,
        image_height: i32,
        model_type: &ModelType,
        result: &DetectionResult,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> AppResult<()> {
        // 트랜잭션 시작
        let tx = self.conn.unchecked_transaction().map_err(AppError::DatabaseError)?;

        // 이미지 저장
        let image_id = self.save_or_update_image(image_path, image_data, image_width, image_height)?;
        
        // 모델 ID 가져오기
        let model_id = self.get_model_id(model_type)?;

        // 추론 세션 저장
        tx.execute(
            "INSERT OR REPLACE INTO inference_sessions 
             (image_id, model_id, inference_time_ms, detection_count, confidence_threshold, nms_threshold)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                image_id,
                model_id,
                result.inference_time_ms,
                result.detections.len() as i32,
                confidence_threshold,
                nms_threshold
            ],
        ).map_err(AppError::DatabaseError)?;

        let session_id = tx.last_insert_rowid();

        // 모델별 검출 결과 저장
        match model_type {
            ModelType::YoloV9(_) => {
                self.save_yolov9_detections(&tx, session_id, &result.detections)?;
            }
            ModelType::RfDetr => {
                self.save_rf_detr_detections(&tx, session_id, &result.detections)?;
            }
        }

        tx.commit().map_err(AppError::DatabaseError)?;
        Ok(())
    }

    /// YOLOv9 검출 결과 저장
    fn save_yolov9_detections(&self, conn: &Connection, session_id: i64, detections: &[Detection]) -> AppResult<()> {
        let mut stmt = conn.prepare(
            "INSERT INTO yolov9_detections 
             (session_id, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_order)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).map_err(AppError::DatabaseError)?;

        for (i, det) in detections.iter().enumerate() {
            stmt.execute(params![
                session_id,
                det.class_id,
                det.class_name,
                det.confidence,
                det.bbox[0],
                det.bbox[1],
                det.bbox[2],
                det.bbox[3],
                i as i32
            ]).map_err(AppError::DatabaseError)?;
        }

        Ok(())
    }

    /// RF-DETR 검출 결과 저장
    fn save_rf_detr_detections(&self, conn: &Connection, session_id: i64, detections: &[Detection]) -> AppResult<()> {
        let mut stmt = conn.prepare(
            "INSERT INTO rf_detr_detections 
             (session_id, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_order, query_index)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).map_err(AppError::DatabaseError)?;

        for (i, det) in detections.iter().enumerate() {
            stmt.execute(params![
                session_id,
                det.class_id,
                det.class_name,
                det.confidence,
                det.bbox[0],
                det.bbox[1],
                det.bbox[2],
                det.bbox[3],
                i as i32,
                None::<i32> // RF-DETR의 query_index는 추후 구현
            ]).map_err(AppError::DatabaseError)?;
        }

        Ok(())
    }

    /// 캐시된 검출 결과 로드
    pub fn load_cached_detections(&self, image_path: &str, model_type: &ModelType) -> AppResult<Option<Vec<Detection>>> {
        let model_id = self.get_model_id(model_type)?;
        
        // 이미지 ID 가져오기
        let image_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM images WHERE path = ?",
            params![image_path],
            |row| row.get(0),
        ).optional().map_err(AppError::DatabaseError)?;

        let image_id = match image_id {
            Some(id) => id,
            None => return Ok(None),
        };

        // 세션 확인
        let session_exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM inference_sessions WHERE image_id = ? AND model_id = ?)",
            params![image_id, model_id],
            |row| row.get(0),
        ).map_err(AppError::DatabaseError)?;

        if !session_exists {
            return Ok(None);
        }

        // 모델별 검출 결과 로드
        let detections = match model_type {
            ModelType::YoloV9(_) => self.load_yolov9_detections(image_id, model_id)?,
            ModelType::RfDetr => self.load_rf_detr_detections(image_id, model_id)?,
        };

        Ok(Some(detections))
    }

    /// YOLOv9 검출 결과 로드
    fn load_yolov9_detections(&self, image_id: i64, model_id: i64) -> AppResult<Vec<Detection>> {
        let mut stmt = self.conn.prepare(
            "SELECT d.class_id, d.class_name, d.confidence, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2
             FROM yolov9_detections d
             JOIN inference_sessions s ON d.session_id = s.id
             WHERE s.image_id = ? AND s.model_id = ?
             ORDER BY d.detection_order"
        ).map_err(AppError::DatabaseError)?;

        let rows = stmt.query_map(params![image_id, model_id], |row| {
            Ok(Detection {
                class_id: row.get(0)?,
                class_name: row.get(1)?,
                confidence: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
            })
        }).map_err(AppError::DatabaseError)?;

        let mut detections = Vec::new();
        for row in rows {
            detections.push(row.map_err(AppError::DatabaseError)?);
        }

        Ok(detections)
    }

    /// RF-DETR 검출 결과 로드
    fn load_rf_detr_detections(&self, image_id: i64, model_id: i64) -> AppResult<Vec<Detection>> {
        let mut stmt = self.conn.prepare(
            "SELECT d.class_id, d.class_name, d.confidence, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2
             FROM rf_detr_detections d
             JOIN inference_sessions s ON d.session_id = s.id
             WHERE s.image_id = ? AND s.model_id = ?
             ORDER BY d.detection_order"
        ).map_err(AppError::DatabaseError)?;

        let rows = stmt.query_map(params![image_id, model_id], |row| {
            Ok(Detection {
                class_id: row.get(0)?,
                class_name: row.get(1)?,
                confidence: row.get(2)?,
                bbox: [row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?],
            })
        }).map_err(AppError::DatabaseError)?;

        let mut detections = Vec::new();
        for row in rows {
            detections.push(row.map_err(AppError::DatabaseError)?);
        }

        Ok(detections)
    }

    /// 캐시 엔트리 삭제
    pub fn delete_cache_entry(&self, image_path: &str, model_type: &ModelType) -> AppResult<()> {
        let model_id = self.get_model_id(model_type)?;
        
        // 이미지 ID 가져오기
        let image_id: Option<i64> = self.conn.query_row(
            "SELECT id FROM images WHERE path = ?",
            params![image_path],
            |row| row.get(0),
        ).optional().map_err(AppError::DatabaseError)?;

        if let Some(image_id) = image_id {
            self.conn.execute(
                "DELETE FROM inference_sessions WHERE image_id = ? AND model_id = ?",
                params![image_id, model_id],
            ).map_err(AppError::DatabaseError)?;
        }

        Ok(())
    }

    /// 성능 통계 가져오기
    pub fn get_performance_stats(&self) -> AppResult<Vec<(String, String, i32, f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT model_name, model_type, total_inferences, avg_inference_time, avg_detections
             FROM model_performance_stats
             WHERE total_inferences > 0"
        ).map_err(AppError::DatabaseError)?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i32>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
            ))
        }).map_err(AppError::DatabaseError)?;

        let mut stats = Vec::new();
        for row in rows {
            stats.push(row.map_err(AppError::DatabaseError)?);
        }

        Ok(stats)
    }

    /// 오래된 캐시 정리
    pub fn cleanup_old_cache(&self, days: i32) -> AppResult<i32> {
        let deleted = self.conn.execute(
            "DELETE FROM images WHERE created_at < datetime('now', '-' || ? || ' days')",
            params![days],
        ).map_err(AppError::DatabaseError)?;

        Ok(deleted as i32)
    }
}