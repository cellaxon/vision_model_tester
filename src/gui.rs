use eframe::egui;
use std::fs;
use std::path::PathBuf;
use yolov9_onnx_test_lib::{Detection, ModelCache, detect_objects_with_settings, get_model_info};

/// GUI 애플리케이션 실행
pub fn run_gui() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "YOLOv9 Object Detection",
        options,
        Box::new(|_cc| Ok(Box::new(YoloV9App::default()))),
    ) {
        eprintln!("GUI 실행 오류: {e}");
    }
}

/// YOLOv9 GUI 애플리케이션 구조체
struct YoloV9App {
    detections: Vec<Detection>,
    is_processing: bool,
    error_message: Option<String>,
    selected_image_path: Option<PathBuf>,
    processed_image: Option<egui::TextureHandle>,
    image_size: egui::Vec2,
    inference_time_ms: Option<f64>,
    model_cache: Option<ModelCache>,
    // 설정값들
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl Default for YoloV9App {
    fn default() -> Self {
        Self {
            detections: Vec::new(),
            is_processing: false,
            error_message: None,
            selected_image_path: None,
            processed_image: None,
            image_size: egui::Vec2::ZERO,
            inference_time_ms: None,
            model_cache: None,
            // 기본 설정값들
            confidence_threshold: 0.6,
            nms_threshold: 0.2,
        }
    }
}

impl eframe::App for YoloV9App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 좌측 사이드 패널 (검출 결과 및 설정)
        egui::SidePanel::left("detections_panel")
            .resizable(false)
            .default_width(400.0)
            .width_range(400.0..=400.0)
            .show(ctx, |ui| {
                self.render_header(ui);
                self.render_settings_panel(ui);
                self.render_error_message(ui);
                self.render_detections_panel(ui);
            });

        // 중앙 패널 (이미지)
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_image_panel(ui);
        });
    }
}

impl YoloV9App {
    /// 헤더 영역 렌더링
    fn render_header(&mut self, ui: &mut egui::Ui) {
        ui.heading("YOLOv9 Object Detection");
        ui.add_space(10.0);

        // 모델 정보 표시
        ui.horizontal(|ui| {
            ui.label("Model:");
            let (model_name, input_size) = get_model_info();
            ui.colored_label(
                egui::Color32::from_rgb(0, 150, 255),
                format!("{} ({}x{})", model_name, input_size, input_size),
            );
        });

        ui.vertical(|ui| {
            if ui
                .add_sized(
                    egui::vec2(380.0, 40.0),
                    egui::Button::new("📁 Select Image"),
                )
                .clicked()
                && !self.is_processing
            {
                self.select_image(ui.ctx());
            }

            if self.is_processing {
                ui.label("Processing...");
            }

            if let Some(path) = &self.selected_image_path {
                let file_name = path
                    .file_name()
                    .map(|f| f.to_string_lossy())
                    .unwrap_or_else(|| "<unknown>".into());
                ui.label(format!("Selected: {}", file_name));
            }
        });

        // 추론 시간 표시
        if let Some(inference_time) = self.inference_time_ms {
            ui.horizontal(|ui| {
                ui.label("⏱️ Inference Time:");
                ui.colored_label(
                    egui::Color32::from_rgb(0, 150, 255),
                    format!("{:.2} ms", inference_time),
                );
            });
        }
    }

    /// 설정 패널 렌더링
    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.collapsing("⚙️ Settings", |ui| {
            ui.add_space(5.0);

            // 신뢰도 임계값 설정
            ui.label("Confidence Threshold:");
            ui.horizontal(|ui| {
                // 슬라이더
                let mut confidence = self.confidence_threshold;
                if ui
                    .add(
                        egui::Slider::new(&mut confidence, 0.1..=1.0)
                            .text("Confidence")
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    self.confidence_threshold = confidence;
                }

                // 숫자 입력 박스
                let mut confidence_text = format!("{:.2}", self.confidence_threshold);
                if ui
                    .add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut confidence_text),
                    )
                    .changed()
                {
                    if let Ok(value) = confidence_text.parse::<f32>() {
                        if value >= 0.1 && value <= 1.0 {
                            self.confidence_threshold = value;
                        }
                    }
                }
            });

            ui.add_space(5.0);

            // NMS 임계값 설정
            ui.label("NMS Threshold:");
            ui.horizontal(|ui| {
                // 슬라이더
                let mut nms = self.nms_threshold;
                if ui
                    .add(
                        egui::Slider::new(&mut nms, 0.05..=0.8)
                            .text("NMS")
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    self.nms_threshold = nms;
                }

                // 숫자 입력 박스
                let mut nms_text = format!("{:.2}", self.nms_threshold);
                if ui
                    .add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut nms_text),
                    )
                    .changed()
                {
                    if let Ok(value) = nms_text.parse::<f32>() {
                        if value >= 0.05 && value <= 0.8 {
                            self.nms_threshold = value;
                        }
                    }
                }
            });

            ui.add_space(5.0);

            // 현재 설정값 표시
            ui.horizontal(|ui| {
                ui.label("Current Settings:");
                ui.colored_label(
                    egui::Color32::from_rgb(100, 200, 100),
                    format!(
                        "Conf: {:.2}, NMS: {:.2}",
                        self.confidence_threshold, self.nms_threshold
                    ),
                );
            });

            ui.add_space(5.0);

            // 재처리 버튼
            if ui
                .add_sized(
                    egui::vec2(380.0, 30.0),
                    egui::Button::new("🔄 Reprocess with New Settings"),
                )
                .clicked()
                && !self.is_processing
            {
                if let Some(path) = &self.selected_image_path {
                    self.process_image(ui.ctx(), path.clone());
                }
            }
        });
    }

    /// 에러 메시지 렌더링
    fn render_error_message(&self, ui: &mut egui::Ui) {
        if let Some(error) = &self.error_message {
            ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
        }
        ui.add_space(10.0);
    }

    /// 검출 결과 패널 렌더링
    fn render_detections_panel(&self, ui: &mut egui::Ui) {
        ui.heading(format!("Detections ({})", self.detections.len()));
        let available_height = ui.available_height();

        // 스크롤 가능한 영역 생성
        egui::ScrollArea::vertical()
            .id_salt("scroll_area_detections")
            .max_height(available_height)
            .show(ui, |ui| {
                if self.detections.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(50.0);
                        ui.label("No detections yet.");
                        ui.label("Select an image to get started.");
                    });
                } else {
                    for (i, detection) in self.detections.iter().enumerate() {
                        self.render_detection_item(ui, i, detection);
                        ui.add_space(5.0);
                    }
                }
            });
    }

    /// 개별 검출 결과 아이템 렌더링
    fn render_detection_item(&self, ui: &mut egui::Ui, index: usize, detection: &Detection) {
        ui.group(|ui| {
            ui.heading(format!("Detection #{}", index + 1));
            ui.label(format!(
                "Class: {} (ID: {})",
                detection.class_name, detection.class_id
            ));
            ui.label(format!("Confidence: {:.1}%", detection.confidence * 100.0));
            ui.label(format!(
                "BBox: [{:.3}, {:.3}, {:.3}, {:.3}]",
                detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
            ));
        });
    }

    /// 이미지 패널 렌더링
    fn render_image_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Processed Image");
        let available_height = ui.available_height();

        // 스크롤 가능한 영역 생성
        egui::ScrollArea::both()
            .id_salt("scroll_area_image")
            .max_height(available_height)
            .show(ui, |ui| {
                if let Some(texture) = &self.processed_image {
                    // 이미지 표시 (egui가 자동으로 스케일링)
                    ui.image(texture);
                } else {
                    self.render_empty_image_placeholder(ui);
                }
            });
    }

    /// 빈 이미지 플레이스홀더 렌더링
    fn render_empty_image_placeholder(&self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(100.0);
            ui.label(egui::RichText::new("📷").size(64.0));
            ui.label("Drag and drop an image here");
            ui.label("or click 'Select Image' to choose a file");
        });
    }

    /// 이미지 파일 선택
    fn select_image(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Image files", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.selected_image_path = Some(path.clone());
            self.process_image(ctx, path);
        }
    }

    /// 이미지 처리
    fn process_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.is_processing = true;
        self.error_message = None;
        self.processed_image = None;
        self.detections.clear();
        self.inference_time_ms = None;

        // 이미지 파일 읽기
        match fs::read(&path) {
            Ok(image_data) => {
                // 모델 캐시 초기화 (필요한 경우)
                if self.model_cache.is_none() {
                    match ModelCache::new() {
                        Ok(cache) => {
                            self.model_cache = Some(cache);
                            println!("Model cache initialized");
                        }
                        Err(e) => {
                            self.error_message =
                                Some(format!("Failed to initialize model cache: {}", e));
                            return;
                        }
                    }
                }

                // 객체 검출 실행 (설정된 임계값 사용)
                if let Some(cache) = &mut self.model_cache {
                    match detect_objects_with_settings(
                        &image_data,
                        cache,
                        self.confidence_threshold,
                        self.nms_threshold,
                    ) {
                        Ok(result) => {
                            self.detections = result.detections;
                            self.inference_time_ms = Some(result.inference_time_ms);
                            self.load_texture(ctx, result.result_image);
                        }
                        Err(e) => {
                            self.error_message = Some(format!("Detection error: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to read file: {}", e));
            }
        }

        self.is_processing = false;
    }

    /// 텍스처 로딩
    fn load_texture(&mut self, ctx: &egui::Context, result_image: image::RgbImage) {
        let mut buffer = Vec::new();
        if let Ok(()) = result_image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Png,
        ) {
            if let Ok(image) = image::load_from_memory(&buffer) {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as _, rgba.height() as _];

                // ColorImage 생성
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());

                let texture = ctx.load_texture("processed_image", color_image, Default::default());
                self.processed_image = Some(texture);
                self.image_size = egui::vec2(size[0] as f32, size[1] as f32);
            }
        }
    }
}
