use eframe::egui;
use std::fs;
use std::path::PathBuf;
use yolov9_onnx_test_lib::{
    Detection, ModelCache, detect_objects_with_settings, get_embedded_model_list, get_model_info,
};

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
    // 이미지 화면 배율 (줌)
    image_zoom: f32,
    // 테이블 선택 상태 (기본 전체 선택)
    selection: Vec<bool>,
    // 정렬 상태
    sort_by: DetectionSortBy,
    sort_asc: bool,
    // 모델 선택
    available_models: Vec<String>,
    selected_model: String,
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
            image_zoom: 1.0,
            selection: Vec::new(),
            sort_by: DetectionSortBy::Index,
            sort_asc: true,
            available_models: get_embedded_model_list(),
            selected_model: "".to_string(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum DetectionSortBy {
    Index,
    Class,
    Id,
    Confidence,
}

impl eframe::App for YoloV9App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 좌측 사이드 패널 (검출 결과 및 설정)
        egui::SidePanel::left("detections_panel")
            .resizable(false)
            .default_width(450.0)
            .width_range(450.0..=450.0)
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
            if self.available_models.is_empty() {
                self.available_models = get_embedded_model_list();
            }
            if self.selected_model.is_empty() {
                if let Some(first) = self.available_models.first() {
                    self.selected_model = first.clone();
                }
            }

            let mut selected = self.selected_model.clone();
            egui::ComboBox::from_id_salt("model_combo")
                .selected_text(&selected)
                .show_ui(ui, |ui| {
                    for m in &self.available_models {
                        ui.selectable_value(&mut selected, m.clone(), m);
                    }
                });
            if selected != self.selected_model {
                self.selected_model = selected;
                // 모델이 바뀌면 다음 추론에서 로딩되도록 세션 유지 (get_session에서 교체)
            }

            let (model_name, input_size) = get_model_info(&self.selected_model);
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

            // 화면 배율 설정
            ui.separator();
            ui.label("Image Zoom:");
            ui.horizontal(|ui| {
                let mut zoom = self.image_zoom;
                if ui
                    .add(
                        egui::Slider::new(&mut zoom, 0.25..=6.0)
                            .text("Zoom")
                            .logarithmic(true),
                    )
                    .changed()
                {
                    self.image_zoom = (zoom * 100.0).round() / 100.0;
                }

                let mut zoom_text = format!("{:.2}x", self.image_zoom);
                if ui
                    .add_sized(
                        egui::vec2(70.0, 20.0),
                        egui::TextEdit::singleline(&mut zoom_text),
                    )
                    .changed()
                {
                    let cleaned = zoom_text.trim_end_matches('x');
                    if let Ok(v) = cleaned.parse::<f32>() {
                        if (0.05..=20.0).contains(&v) {
                            self.image_zoom = v;
                        }
                    }
                }

                if ui.button("100% ").clicked() {
                    self.image_zoom = 1.0;
                }
                if ui.button("Fit").clicked() {
                    // 가능한 경우 중앙패널 높이에 맞춰 대략 맞춤
                    // 정확한 fit은 이미지 표시 위치에서 계산됨
                    self.image_zoom = 1.0; // 일단 1.0으로 리셋
                }
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

    /// 검출 결과 패널 렌더링 (테이블 형태)
    fn render_detections_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading(format!("Detections ({})", self.detections.len()));
        let available_height = ui.available_height();

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
                    // 정렬 컨트롤
                    ui.horizontal(|ui| {
                        ui.label("Sort by:");
                        let mut sort_by = self.sort_by;
                        egui::ComboBox::from_id_salt("sort_by")
                            .selected_text(match sort_by {
                                DetectionSortBy::Index => "Index",
                                DetectionSortBy::Class => "Class",
                                DetectionSortBy::Id => "ID",
                                DetectionSortBy::Confidence => "Conf",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut sort_by, DetectionSortBy::Index, "Index");
                                ui.selectable_value(&mut sort_by, DetectionSortBy::Class, "Class");
                                ui.selectable_value(&mut sort_by, DetectionSortBy::Id, "ID");
                                ui.selectable_value(
                                    &mut sort_by,
                                    DetectionSortBy::Confidence,
                                    "Conf",
                                );
                            });
                        if sort_by != self.sort_by {
                            self.sort_by = sort_by;
                        }
                        if ui
                            .button(if self.sort_asc { "Asc" } else { "Desc" })
                            .clicked()
                        {
                            self.sort_asc = !self.sort_asc;
                        }
                        ui.separator();
                        // 전체 선택/해제
                        if ui.button("Select All").clicked() {
                            self.selection.fill(true);
                        }
                        if ui.button("Select None").clicked() {
                            self.selection.fill(false);
                        }
                    });
                    ui.add_space(6.0);

                    // 정렬된 인덱스 목록 생성
                    let mut indices: Vec<usize> = (0..self.detections.len()).collect();
                    indices.sort_by(|&a, &b| {
                        let ord = match self.sort_by {
                            DetectionSortBy::Index => a.cmp(&b),
                            DetectionSortBy::Class => self.detections[a]
                                .class_name
                                .cmp(&self.detections[b].class_name),
                            DetectionSortBy::Id => self.detections[a]
                                .class_id
                                .cmp(&self.detections[b].class_id),
                            DetectionSortBy::Confidence => self.detections[a]
                                .confidence
                                .partial_cmp(&self.detections[b].confidence)
                                .unwrap_or(std::cmp::Ordering::Equal),
                        };
                        if self.sort_asc { ord } else { ord.reverse() }
                    });

                    egui::Grid::new("detections_table")
                        .striped(true)
                        .num_columns(6)
                        .spacing([6.0, 4.0])
                        .show(ui, |ui| {
                            // 헤더 (클릭해서 정렬)
                            ui.label(egui::RichText::new("Sel").strong());

                            let mut hdr = String::from("#");
                            if self.sort_by == DetectionSortBy::Index {
                                hdr.push_str(if self.sort_asc { " ↑" } else { " ↓" });
                            }
                            if ui.button(hdr).clicked() {
                                if self.sort_by == DetectionSortBy::Index {
                                    self.sort_asc = !self.sort_asc;
                                } else {
                                    self.sort_by = DetectionSortBy::Index;
                                    self.sort_asc = true;
                                }
                            }

                            let mut hdr = String::from("Class");
                            if self.sort_by == DetectionSortBy::Class {
                                hdr.push_str(if self.sort_asc { " ↑" } else { " ↓" });
                            }
                            if ui.button(hdr).clicked() {
                                if self.sort_by == DetectionSortBy::Class {
                                    self.sort_asc = !self.sort_asc;
                                } else {
                                    self.sort_by = DetectionSortBy::Class;
                                    self.sort_asc = true;
                                }
                            }

                            let mut hdr = String::from("ID");
                            if self.sort_by == DetectionSortBy::Id {
                                hdr.push_str(if self.sort_asc { " ↑" } else { " ↓" });
                            }
                            if ui.button(hdr).clicked() {
                                if self.sort_by == DetectionSortBy::Id {
                                    self.sort_asc = !self.sort_asc;
                                } else {
                                    self.sort_by = DetectionSortBy::Id;
                                    self.sort_asc = true;
                                }
                            }

                            let mut hdr = String::from("Conf");
                            if self.sort_by == DetectionSortBy::Confidence {
                                hdr.push_str(if self.sort_asc { " ↑" } else { " ↓" });
                            }
                            if ui.button(hdr).clicked() {
                                if self.sort_by == DetectionSortBy::Confidence {
                                    self.sort_asc = !self.sort_asc;
                                } else {
                                    self.sort_by = DetectionSortBy::Confidence;
                                    self.sort_asc = true;
                                }
                            }

                            ui.label(egui::RichText::new("BBox [x1,y1,x2,y2]").strong());
                            ui.end_row();

                            // 데이터 행
                            for &i in &indices {
                                let det = &self.detections[i];
                                if i >= self.selection.len() {
                                    continue;
                                }
                                let mut checked = self.selection[i];
                                ui.add(egui::Checkbox::without_text(&mut checked));
                                if checked != self.selection[i] {
                                    self.selection[i] = checked;
                                }

                                // Row 클릭 토글: 체크박스 외의 어떤 셀을 클릭해도 토글
                                let mut row_clicked = false;

                                if ui
                                    .add(
                                        egui::Label::new(format!("{}", i + 1))
                                            .sense(egui::Sense::click()),
                                    )
                                    .clicked()
                                {
                                    row_clicked = true;
                                }

                                if ui
                                    .add(
                                        egui::Label::new(det.class_name.clone())
                                            .sense(egui::Sense::click()),
                                    )
                                    .clicked()
                                {
                                    row_clicked = true;
                                }

                                if ui
                                    .add(
                                        egui::Label::new(format!("{}", det.class_id))
                                            .sense(egui::Sense::click()),
                                    )
                                    .clicked()
                                {
                                    row_clicked = true;
                                }

                                if ui
                                    .add(
                                        egui::Label::new(format!("{:.1}%", det.confidence * 100.0))
                                            .sense(egui::Sense::click()),
                                    )
                                    .clicked()
                                {
                                    row_clicked = true;
                                }

                                let bbox_text = format!(
                                    "[{:.3}, {:.3}, {:.3}, {:.3}]",
                                    det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
                                );
                                if ui
                                    .add(egui::Label::new(bbox_text).sense(egui::Sense::click()))
                                    .clicked()
                                {
                                    row_clicked = true;
                                }

                                if row_clicked {
                                    self.selection[i] = !self.selection[i];
                                }

                                ui.end_row();
                            }
                        });
                }
            });
    }

    // (제거됨) 리스트 렌더링에서 테이블 렌더링으로 대체

    /// 이미지 패널 렌더링 (원본 이미지 + egui 오버레이로 bbox/텍스트)
    fn render_image_panel(&mut self, ui: &mut egui::Ui) {
        let available_height = ui.available_height();

        egui::ScrollArea::both()
            .id_salt("scroll_area_image")
            .max_height(available_height)
            .show(ui, |ui| {
                // 우측(이미지 영역) 전체에서 휠 입력을 줌으로 처리
                let pointer_in_area = ui
                    .input(|i| i.pointer.hover_pos())
                    .map_or(false, |pos| ui.clip_rect().contains(pos));
                if pointer_in_area {
                    let scroll_delta = ui.input(|i| i.smooth_scroll_delta).y;
                    if scroll_delta != 0.0 {
                        let zoom_factor = if scroll_delta > 0.0 { 1.1 } else { 1.0 / 1.1 };
                        self.image_zoom = (self.image_zoom * zoom_factor).clamp(0.1, 20.0);
                    }
                }

                if let Some(texture) = &self.processed_image {
                    // 줌 적용된 크기 계산
                    let desired_size = egui::vec2(
                        texture.size()[0] as f32 * self.image_zoom,
                        texture.size()[1] as f32 * self.image_zoom,
                    );

                    // 가용 영역의 중앙에 배치 (이미지가 작을 때만 오프셋 적용)
                    let avail_rect = ui.available_rect_before_wrap();
                    let offset_x = ((avail_rect.width() - desired_size.x) * 0.5).max(0.0);
                    let offset_y = ((avail_rect.height() - desired_size.y) * 0.5).max(0.0);
                    let top_left =
                        egui::pos2(avail_rect.left() + offset_x, avail_rect.top() + offset_y);
                    let img_rect = egui::Rect::from_min_size(top_left, desired_size);

                    // 이미지 표시 (배율 적용, 지정된 위치에 배치)
                    let response = ui.put(
                        img_rect,
                        egui::Image::from_texture(texture).fit_to_exact_size(desired_size),
                    );

                    // 드래그로 스크롤 이동 (스크롤 영역에 드래그 전달)
                    if response.dragged() {
                        let delta = response.drag_delta();
                        ui.scroll_with_delta(egui::vec2(-delta.x, -delta.y));
                    }

                    let rect = response.rect;
                    let painter = ui.painter_at(rect);

                    // bbox와 라벨을 오버레이로 렌더링 (선택된 항목만)
                    for (i, det) in self.detections.iter().enumerate() {
                        if i >= self.selection.len() || !self.selection[i] {
                            continue;
                        }
                        let [x1, y1, x2, y2] = det.bbox;
                        // 정규화 좌표(0-1)를 화면 좌표로 변환
                        let p_min = egui::pos2(
                            rect.left() + x1 * rect.width(),
                            rect.top() + y1 * rect.height(),
                        );
                        let p_max = egui::pos2(
                            rect.left() + x2 * rect.width(),
                            rect.top() + y2 * rect.height(),
                        );

                        // 박스 (라인 세그먼트로 사각형 그리기: 버전 차이 회피)
                        let box_color = egui::Color32::from_rgb(255, 0, 0);
                        let stroke = egui::Stroke::new(2.0, box_color);
                        let p1 = p_min; // 좌상
                        let p2 = egui::pos2(p_max.x, p_min.y); // 우상
                        let p3 = p_max; // 우하
                        let p4 = egui::pos2(p_min.x, p_max.y); // 좌하
                        painter.line_segment([p1, p2], stroke);
                        painter.line_segment([p2, p3], stroke);
                        painter.line_segment([p3, p4], stroke);
                        painter.line_segment([p4, p1], stroke);

                        // 텍스트 (좌상단 박스 위쪽)
                        let label = format!("{} {:.0}%", det.class_name, det.confidence * 100.0);
                        let text_pos = egui::pos2(p_min.x, (p_min.y - 4.0).max(rect.top() + 12.0));
                        painter.text(
                            text_pos,
                            egui::Align2::LEFT_BOTTOM,
                            label,
                            egui::FontId::proportional(14.0),
                            egui::Color32::WHITE,
                        );
                    }
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
                        &self.selected_model,
                        self.confidence_threshold,
                        self.nms_threshold,
                    ) {
                        Ok(result) => {
                            self.detections = result.detections;
                            // 선택 상태 초기화 (기본 전체 선택)
                            self.selection = vec![true; self.detections.len()];
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
