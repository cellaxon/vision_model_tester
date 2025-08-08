use include_dir::{Dir, include_dir};

// 임베디드 리소스: assets/models 폴더의 모든 onnx 파일을 임베딩
static ASSETS_MODELS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/assets/models");

/// 임베디드된 모델 파일(.onnx) 목록 반환
pub fn get_embedded_model_list() -> Vec<String> {
    ASSETS_MODELS_DIR
        .files()
        .filter_map(|f| {
            let path = f.path();
            if let Some(ext) = path.extension() {
                if ext == "onnx" {
                    return Some(path.file_name().unwrap().to_string_lossy().to_string());
                }
            }
            None
        })
        .collect()
}

/// 파일명으로 임베디드된 모델 바이트를 조회
pub fn get_embedded_model_bytes(file_name: &str) -> anyhow::Result<&'static [u8]> {
    let file = ASSETS_MODELS_DIR
        .files()
        .find(|f| f.path().file_name().unwrap().to_string_lossy() == file_name)
        .ok_or_else(|| anyhow::anyhow!(format!("Embedded model not found: {}", file_name)))?;
    Ok(file.contents())
}
