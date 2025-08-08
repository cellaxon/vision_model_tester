# YOLOv9 ONNX Object Detection Application

YOLOv9 모델을 사용한 객체 검출 애플리케이션입니다. ONNX Runtime을 통해 추론을 수행하며, GUI 인터페이스를 제공합니다.

## 🚀 주요 기능

### 🔍 객체 검출
- YOLOv9 모델을 사용한 실시간 객체 검출
- COCO 데이터셋의 80개 클래스 지원
- 신뢰도 임계값 및 NMS 임계값 조정 가능
- Pre-NMS 데이터 캐싱으로 빠른 재처리
- **신뢰도 기반 바운딩 박스 색상 매핑**

### 🖥️ GUI 인터페이스
- egui 기반의 현대적인 사용자 인터페이스
- **자연로그 기반 세밀한 줌 제어**
- 실시간 이미지 줌 및 패닝
- 검출 결과 테이블 형태 표시
- 정렬 및 필터링 기능
- 키보드 단축키 지원
- **바운딩 박스 색상 매핑 모드 선택**

### 💾 캐싱 시스템
- SQLite 데이터베이스를 사용한 추론 결과 캐싱
- 이미지 해시 기반 중복 검출 방지
- 설정 가능한 캐시 정리 정책
- **Pre-NMS 데이터 저장으로 NMS 재실행 최적화**

### ⚙️ 설정 관리
- 중앙화된 설정 시스템
- 런타임 설정 변경 지원
- 모듈화된 설정 구조
- **컴파일 타임 상수 관리**

### 🎨 바운딩 박스 색상 시스템
- **고정 색상 모드**: 모든 박스를 빨간색으로 표시
- **범위별 색상 모드**: 5단계 색상 (파랑→초록→노랑→주황→빨강)
- **그라데이션 모드**: 선형 색상 전환
- **HSV 기반 모드**: 자연스러운 색상 전환

## 📁 프로젝트 구조

```
yolov9_onnx_test/
├── src/
│   ├── lib.rs          # 메인 라이브러리
│   ├── main.rs         # 애플리케이션 진입점
│   ├── gui.rs          # GUI 구현
│   ├── models.rs       # 임베디드 모델 관리
│   ├── config.rs       # 설정 관리
│   ├── error.rs        # 에러 처리
│   └── utils.rs        # 유틸리티 함수들
│       ├── image_utils     # 이미지 처리 유틸리티
│       ├── math_utils      # 수학 계산 함수들
│       ├── fs_utils        # 파일 시스템 유틸리티
│       ├── perf_utils      # 성능 측정 도구
│       └── color_utils     # 색상 처리 유틸리티
├── assets/
│   └── models/         # ONNX 모델 파일들
├── Cargo.toml          # 프로젝트 의존성
└── README.md           # 프로젝트 문서
```

## 🔧 모듈 설명

### `lib.rs` - 메인 라이브러리
- ONNX 모델 추론 로직
- 이미지 전처리 및 후처리
- NMS (Non-Maximum Suppression) 구현
- 모델 캐싱 시스템
- 데이터베이스 관리
- **신뢰도 기반 바운딩 박스 색상 처리**

### `gui.rs` - GUI 구현
- egui 기반 사용자 인터페이스
- **자연로그 기반 줌 컨트롤**
- 이미지 표시 및 줌 컨트롤
- 검출 결과 테이블
- 설정 패널
- **바운딩 박스 색상 매핑 모드 선택**

### `config.rs` - 설정 관리
- 애플리케이션 설정 구조체
- 모델, 추론, UI, 데이터베이스 설정
- 기본값 관리
- **바운딩 박스 색상 매핑 모드 정의**

### `error.rs` - 에러 처리
- **커스텀 에러 타입 정의** (`thiserror` 사용)
- 에러 컨텍스트 제공
- 유효성 검사 함수들
- **패닉 방지 에러 처리**

### `utils.rs` - 유틸리티 함수들
- **이미지 처리 유틸리티** (`image_utils`)
- **수학 계산 함수들** (`math_utils`)
- **파일 시스템 유틸리티** (`fs_utils`)
- **성능 측정 도구** (`perf_utils`)
- **색상 처리 유틸리티** (`color_utils`)

## 🛠️ 설치 및 실행

### 요구사항
- Rust 1.70+
- Windows/macOS/Linux

### 빌드
```bash
cargo build --release
```

### 실행
```bash
cargo run
```

## 🎮 사용법

### 기본 사용법
1. 애플리케이션 실행
2. "Select Image" 버튼 클릭하여 이미지 선택
3. 모델 선택 (기본: gelan-e.onnx)
4. 신뢰도 및 NMS 임계값 조정
5. **바운딩 박스 색상 매핑 모드 선택**
6. 검출 결과 확인

### 줌 컨트롤
- **마우스 휠**: **자연로그 기반 세밀한 줌 조정**
- **Ctrl + Plus/Minus**: 키보드 줌
- **0**: 100% 줌으로 리셋
- **1**: 50% 줌
- **2**: 200% 줌

### 설정 조정
- **Confidence Threshold**: 검출 신뢰도 임계값 (0.1-1.0)
- **NMS Threshold**: 중복 제거 임계값 (0.05-0.8)
- **Image Zoom**: 이미지 확대/축소 (0.1x-20.0x)
- **Color Mapping Mode**: 바운딩 박스 색상 매핑 방식

## ⚙️ 설정 시스템

### 설정 구조
```rust
pub struct AppConfig {
    pub model: ModelConfig,        // 모델 관련 설정
    pub inference: InferenceConfig, // 추론 관련 설정
    pub ui: UiConfig,              // UI 관련 설정
    pub database: DatabaseConfig,   // 데이터베이스 설정
}
```

### 주요 설정값들
- **모델 입력 크기**: 640x640
- **기본 신뢰도 임계값**: 0.6
- **기본 NMS 임계값**: 0.2
- **최대 검출 개수**: 50
- **줌 범위**: 0.1x - 20.0x
- **자연로그 줌 변화량**: 0.05 (매우 세밀한 제어)

## 🔄 캐싱 시스템

### 데이터베이스 구조
```sql
CREATE TABLE inference_cache (
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
);
```

### 캐시 기능
- 이미지 해시 기반 중복 검출
- 모델별 캐시 분리
- 자동 캐시 정리 (30일)
- 강제 재추론 옵션
- **Pre-NMS 데이터 저장으로 NMS 재실행 최적화**

## 🎨 바운딩 박스 색상 시스템

### 색상 매핑 모드

#### 1. 고정 색상 모드 (Fixed)
- 모든 바운딩 박스를 빨간색으로 표시
- 가장 단순한 모드

#### 2. 범위별 색상 모드 (Range-Based)
- 5단계 색상 구분:
  - 0.0-0.2: 파랑 (낮은 신뢰도)
  - 0.2-0.4: 초록
  - 0.4-0.6: 노랑
  - 0.6-0.8: 주황
  - 0.8-1.0: 빨강 (높은 신뢰도)

#### 3. 그라데이션 모드 (Gradient)
- 선형 색상 전환
- 신뢰도에 따른 부드러운 색상 변화

#### 4. HSV 기반 모드 (HSV-Based)
- HSV 색상 공간을 활용한 자연스러운 전환
- 가장 시각적으로 매력적인 모드

### 색상 처리 유틸리티
```rust
// 범위별 색상
pub fn get_confidence_color(confidence: f32) -> Rgb<u8>

// 그라데이션 색상
pub fn get_confidence_color_gradient(confidence: f32) -> Rgb<u8>

// HSV 기반 색상
pub fn get_confidence_color_hsv(confidence: f32) -> Rgb<u8>
```

## 🚀 성능 최적화

### 추론 최적화
- ONNX Runtime 최적화 레벨 설정
- 스레드 풀 최적화
- 메모리 패턴 최적화
- GPU 가속 지원 (macOS CoreML)
- **Pre-NMS 데이터 캐싱으로 NMS 재실행 최적화**

### GUI 최적화
- **자연로그 기반 줌 제어** (매우 세밀한 조정)
- 효율적인 이미지 렌더링
- 가상화된 테이블 표시
- 비동기 이미지 처리
- **색상 매핑 최적화**

## 🐛 에러 처리

### 에러 타입 (`thiserror` 사용)
- `ImageError`: 이미지 처리 오류
- `OrtError`: ONNX 런타임 오류
- `DatabaseError`: 데이터베이스 오류
- `ValidationError`: 유효성 검사 오류
- `ConfigError`: 설정 오류

### 에러 처리 전략
- **패닉 방지**: 모든 `unwrap()` 제거
- Graceful degradation
- 사용자 친화적 에러 메시지
- 자동 복구 시도
- 상세한 로깅
- **커스텀 에러 타입으로 타입 안전성 확보**

## 📊 성능 측정

### 측정 항목
- 추론 시간 (밀리초)
- 이미지 로딩 시간
- GUI 렌더링 시간
- 캐시 히트율
- **색상 처리 성능**

### 성능 최적화 팁
- SSD 사용 권장
- 충분한 RAM 확보 (8GB+)
- GPU 가속 활용 (지원 시)
- 정기적인 캐시 정리
- **자연로그 줌으로 부드러운 사용자 경험**

## 🔧 개발 정보

### 의존성
- `ort`: ONNX Runtime
- `egui`: GUI 프레임워크
- `image`: 이미지 처리
- `rusqlite`: SQLite 데이터베이스
- `serde`: 직렬화/역직렬화
- **`thiserror`: 커스텀 에러 처리**
- **`once_cell`: 지연 정적 초기화**

### 빌드 최적화
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

### 코드 구조 개선
- **모듈화**: `config`, `error`, `utils` 모듈 분리
- **중앙화된 설정**: `CONFIG` 정적 인스턴스
- **유틸리티 함수 분리**: 공통 기능 모듈화
- **타입 안전성**: 커스텀 에러 타입 사용

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📞 지원

문제가 발생하면 이슈를 생성해 주세요.
