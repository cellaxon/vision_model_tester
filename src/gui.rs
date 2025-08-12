use eframe::egui;
use std::fs;
use std::path::PathBuf;
use vision_model_tester_lib::{
    Detection, InferenceDb, apply_nms_only,
    models::{ModelType, UnifiedInferenceEngine},
};

// 설정에서 줌 관련 상수 가져오기
use vision_model_tester_lib::config::CONFIG;

/// GUI 애플리케이션 실행
pub fn run_gui() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "YOLOv9 & RF-DETR Object Detection",
        options,
        Box::new(|_cc| Ok(Box::new(UnifiedDetectionApp::default()))),
    ) {
        eprintln!("GUI 실행 오류: {e}");
    }
}

/// 통합 객체 검출 GUI 애플리케이션 구조체
struct UnifiedDetectionApp {
    detections: Vec<Detection>,
    pre_nms_detections: Vec<Detection>,
    is_processing: bool,
    error_message: Option<String>,
    selected_image_path: Option<PathBuf>,
    processed_image: Option<egui::TextureHandle>,
    image_size: egui::Vec2,
    inference_time_ms: Option<f64>,
    inference_db: Option<InferenceDb>,
    // 통합 추론 엔진
    inference_engine: Option<UnifiedInferenceEngine>,
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
    available_models: Vec<ModelType>,
    selected_model: ModelType,
    // 색상 매핑 방식
    color_mapping_mode: vision_model_tester_lib::config::ColorMappingMode,
}

impl Default for UnifiedDetectionApp {
    fn default() -> Self {
        let available_models = UnifiedInferenceEngine::get_available_models();
        let selected_model = available_models.first().cloned().unwrap_or(ModelType::YoloV9("gelan-e.onnx".to_string()));
        
        Self {
            detections: Vec::new(),
            pre_nms_detections: Vec::new(),
            is_processing: false,
            error_message: None,
            selected_image_path: None,
            processed_image: None,
            image_size: egui::Vec2::ZERO,
            inference_time_ms: None,
            inference_db: None,
            inference_engine: None,
            // 기본 설정값들
            confidence_threshold: CONFIG.ui.default_confidence_threshold,
            nms_threshold: CONFIG.ui.default_nms_threshold,
            image_zoom: CONFIG.ui.default_zoom,
            selection: Vec::new(),
            sort_by: DetectionSortBy::Index,
            sort_asc: true,
            available_models,
            selected_model,
            color_mapping_mode: CONFIG.ui.bounding_box_color_mode.clone(),
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

impl eframe::App for UnifiedDetectionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 키보드 단축키 처리 (줌 컨트롤)
        self.handle_keyboard_shortcuts(ctx);

        // 좌측 사이드 패널 (검출 결과 및 설정)
        egui::SidePanel::left("detections_panel")
            .resizable(false)
            .default_width(450.0)
            .width_range(450.0..=450.0)
            .show(ctx, |ui| {
                self.render_header(ui, ctx);
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

impl UnifiedDetectionApp {
    /// 키보드 단축키 처리
    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        ctx.input(|input| {
            // Ctrl + Plus/Minus: 줌 인/아웃 (더 세밀한 제어)
            if input.key_pressed(egui::Key::Plus) && input.modifiers.ctrl {
                let current_log_zoom = self.image_zoom.ln();
                let new_log_zoom = (current_log_zoom + CONFIG.ui.keyboard_zoom_delta)
                    .clamp(CONFIG.ui.min_zoom_log, CONFIG.ui.max_zoom_log);
                self.image_zoom = new_log_zoom.exp();
            }

            if input.key_pressed(egui::Key::Minus) && input.modifiers.ctrl {
                let current_log_zoom = self.image_zoom.ln();
                let new_log_zoom = (current_log_zoom - CONFIG.ui.keyboard_zoom_delta)
                    .clamp(CONFIG.ui.min_zoom_log, CONFIG.ui.max_zoom_log);
                self.image_zoom = new_log_zoom.exp();
            }

            // 숫자 키 0: 줌 리셋 (100%)
            if input.key_pressed(egui::Key::Num0) {
                self.image_zoom = 1.0;
            }

            // 숫자 키 1: 50% 줌
            if input.key_pressed(egui::Key::Num1) {
                self.image_zoom = 0.5;
            }

            // 숫자 키 2: 200% 줌
            if input.key_pressed(egui::Key::Num2) {
                self.image_zoom = 2.0;
            }
        });
    }

    /// 헤더 영역 렌더링
    fn render_header(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("YOLOv9 & RF-DETR Object Detection");
        ui.add_space(10.0);

        // 모델 정보 표시
        if self.available_models.is_empty() {
            self.available_models = UnifiedInferenceEngine::get_available_models();
        }

        let mut selected = self.selected_model.clone();
        let available_models = &self.available_models;
        
        ui.horizontal(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_id_salt("model_combo")
                .selected_text(&selected.display_name())
                .show_ui(ui, |ui| {
                    for m in available_models {
                        ui.selectable_value(&mut selected, m.clone(), m.display_name());
                    }
                });
        });
        
        if selected != self.selected_model {
            self.selected_model = selected;
            // 모델이 바뀌면 추론 엔진 재초기화
            if let Some(engine) = &mut self.inference_engine {
                if let Err(e) = engine.set_model(self.selected_model.clone()) {
                    self.error_message = Some(format!("모델 변경 실패: {}", e));
                }
            }
            // 현재 이미지가 있다면 즉시 재추론 실행
            if let Some(image_path) = &self.selected_image_path {
                self.process_image(ctx, image_path.clone());
            }
        }

        ui.horizontal(|ui| {
            let model_info = &self.selected_model;
            ui.colored_label(
                egui::Color32::from_rgb(0, 150, 255),
                format!("{} ({}x{})", model_info.display_name(), model_info.input_size(), model_info.input_size()),
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
                    self.update_selection_by_confidence();
                }

                // 숫자 입력 박스
                let mut confidence_text = format!("{:.2}", self.confidence_threshold);
                if ui
                    .add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut confidence_text),
                    )
                    .changed()
                    && let Ok(value) = confidence_text.parse::<f32>()
                    && (0.1..=1.0).contains(&value)
                {
                    self.confidence_threshold = value;
                    self.update_selection_by_confidence();
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
                    self.reapply_nms_only();
                }

                // 숫자 입력 박스
                let mut nms_text = format!("{:.2}", self.nms_threshold);
                if ui
                    .add_sized(
                        egui::vec2(60.0, 20.0),
                        egui::TextEdit::singleline(&mut nms_text),
                    )
                    .changed()
                    && let Ok(value) = nms_text.parse::<f32>()
                    && (0.05..=0.8).contains(&value)
                {
                    self.nms_threshold = value;
                    self.reapply_nms_only();
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

            // 화면 배율 설정 (자연로그 기반)
            ui.separator();
            ui.label("Image Zoom (Natural Log):");
            ui.horizontal(|ui| {
                // 자연로그 공간에서 슬라이더 작동
                let log_zoom = self.image_zoom.ln();
                let mut log_zoom_value = log_zoom;
                if ui
                    .add(
                        egui::Slider::new(
                            &mut log_zoom_value,
                            CONFIG.ui.min_zoom_log..=CONFIG.ui.max_zoom_log,
                        )
                        .text("Log Zoom")
                        .fixed_decimals(2),
                    )
                    .changed()
                {
                    // 로그 공간에서 선형 공간으로 변환
                    self.image_zoom = log_zoom_value.exp();
                }

                // 현재 줌 레벨 표시 (더 정확한 소수점)
                let mut zoom_text = format!("{:.3}x", self.image_zoom);
                if ui
                    .add_sized(
                        egui::vec2(80.0, 20.0),
                        egui::TextEdit::singleline(&mut zoom_text),
                    )
                    .changed()
                {
                    let cleaned = zoom_text.trim_end_matches('x');
                    if let Ok(v) = cleaned.parse::<f32>()
                        && (0.1..=20.0).contains(&v)
                    {
                        self.image_zoom = v;
                    }
                }
            });

            ui.horizontal(|ui| {
                // 줌 컨트롤 버튼들
                if ui.button("100%").clicked() {
                    self.image_zoom = 1.0;
                }
                if ui.button("50%").clicked() {
                    self.image_zoom = 0.5;
                }
                if ui.button("200%").clicked() {
                    self.image_zoom = 2.0;
                }
                if ui.button("Fit").clicked() {
                    // 이미지가 화면에 맞도록 자동 조정
                    self.image_zoom = 1.0;
                }
            });

            // 현재 줌 정보 표시
            ui.horizontal(|ui| {
                ui.label("Zoom Info:");
                ui.colored_label(
                    egui::Color32::from_rgb(150, 150, 255),
                    format!(
                        "Current: {:.3}x (log: {:.3})",
                        self.image_zoom,
                        self.image_zoom.ln()
                    ),
                );
            });

            // 색상 매핑 방식 설정
            ui.add_space(5.0);
            ui.separator();
            ui.label("Bounding Box Color Mapping:");
            ui.horizontal(|ui| {
                ui.label("Color Mode:");
                egui::ComboBox::from_id_salt("color_mapping_mode")
                    .selected_text(match self.color_mapping_mode {
                        vision_model_tester_lib::config::ColorMappingMode::Fixed => "Fixed (Red)",
                        vision_model_tester_lib::config::ColorMappingMode::RangeBased => {
                            "Range-Based (5 levels)"
                        }
                        vision_model_tester_lib::config::ColorMappingMode::Gradient => {
                            "Gradient (Linear)"
                        }
                        vision_model_tester_lib::config::ColorMappingMode::HsvBased => {
                            "HSV-Based (Smooth)"
                        }
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.color_mapping_mode,
                            vision_model_tester_lib::config::ColorMappingMode::Fixed,
                            "Fixed (Red)",
                        );
                        ui.selectable_value(
                            &mut self.color_mapping_mode,
                            vision_model_tester_lib::config::ColorMappingMode::RangeBased,
                            "Range-Based (5 levels)",
                        );
                        ui.selectable_value(
                            &mut self.color_mapping_mode,
                            vision_model_tester_lib::config::ColorMappingMode::Gradient,
                            "Gradient (Linear)",
                        );
                        ui.selectable_value(
                            &mut self.color_mapping_mode,
                            vision_model_tester_lib::config::ColorMappingMode::HsvBased,
                            "HSV-Based (Smooth)",
                        );
                    });
            });

            // 색상 매핑 설명
            ui.collapsing("🎨 Color Mapping Info", |ui| {
                ui.label("Color mapping based on confidence level:");
                ui.label("• Fixed: All boxes are red");
                ui.label("• Range-Based: 5 distinct colors (Blue→Green→Yellow→Orange→Red)");
                ui.label("• Gradient: Smooth linear transition");
                ui.label("• HSV-Based: Natural color transition using HSV space");
            });

            // 키보드 단축키 도움말
            ui.add_space(5.0);
            ui.collapsing("⌨️ Keyboard Shortcuts", |ui| {
                ui.label("Zoom Controls:");
                ui.label("• Ctrl + Plus: Zoom In");
                ui.label("• Ctrl + Minus: Zoom Out");
                ui.label("• 0: Reset to 100%");
                ui.label("• 1: 50% Zoom");
                ui.label("• 2: 200% Zoom");
                ui.label("• Mouse Wheel: Fine zoom control");
            });

            ui.add_space(5.0);

            // 재처리 버튼 (캐시 무시 후 강제 재추론)
            if ui
                .add_sized(
                    egui::vec2(380.0, 30.0),
                    egui::Button::new("🔄 Force Re-infer (ignore DB cache)"),
                )
                .clicked()
                && !self.is_processing
                && let Some(path) = &self.selected_image_path
            {
                self.process_image_with_force(ui.ctx(), path.clone());
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
                            // Sel 헤더: 전체 선택/해제 토글
                            let all_selected =
                                !self.selection.is_empty() && self.selection.iter().all(|&s| s);
                            let sel_hdr = if all_selected { "Sel (All)" } else { "Sel" };
                            if ui.button(sel_hdr).clicked() {
                                let to = !all_selected;
                                self.selection.fill(to);
                            }

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
                // 우측(이미지 영역) 전체에서 휠 입력을 줌으로 처리 (자연로그 기반)
                let pointer_in_area = ui
                    .input(|i| i.pointer.hover_pos())
                    .is_some_and(|pos| ui.clip_rect().contains(pos));
                if pointer_in_area {
                    let scroll_delta = ui.input(|i| i.smooth_scroll_delta).y;
                    if scroll_delta != 0.0 {
                        // 자연로그 기반 줌: 더 세밀한 제어
                        // 현재 줌 값을 자연로그 공간으로 변환
                        let current_log_zoom = self.image_zoom.ln();
                        
                        // 스크롤 델타에 따른 로그 공간에서의 변화량 (매우 세밀한 제어)
                        let log_delta = if scroll_delta > 0.0 {
                            // 확대: 매우 작은 증가량
                            CONFIG.ui.mouse_wheel_zoom_delta
                        } else {
                            // 축소: 매우 작은 감소량
                            -CONFIG.ui.mouse_wheel_zoom_delta
                        };
                        
                        // 새로운 로그 줌 값 계산
                        let new_log_zoom = (current_log_zoom + log_delta).clamp(CONFIG.ui.min_zoom_log, CONFIG.ui.max_zoom_log);
                        
                        // 로그 공간에서 다시 선형 공간으로 변환
                        self.image_zoom = new_log_zoom.exp();
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

                        // 신뢰도에 따른 색상 결정
                        let box_color = match self.color_mapping_mode {
                            vision_model_tester_lib::config::ColorMappingMode::Fixed => {
                                egui::Color32::from_rgb(255, 0, 0)
                            }
                            vision_model_tester_lib::config::ColorMappingMode::RangeBased => {
                                let color = vision_model_tester_lib::utils::color_utils::get_confidence_color(det.confidence);
                                egui::Color32::from_rgb(color[0], color[1], color[2])
                            }
                            vision_model_tester_lib::config::ColorMappingMode::Gradient => {
                                let color = vision_model_tester_lib::utils::color_utils::get_confidence_color_gradient(det.confidence);
                                egui::Color32::from_rgb(color[0], color[1], color[2])
                            }
                            vision_model_tester_lib::config::ColorMappingMode::HsvBased => {
                                let color = vision_model_tester_lib::utils::color_utils::get_confidence_color_hsv(det.confidence);
                                egui::Color32::from_rgb(color[0], color[1], color[2])
                            }
                        };
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
        self.pre_nms_detections.clear();
        self.inference_time_ms = None;

        // 이미지 파일 읽기
        let image_data = match fs::read(&path) {
            Ok(data) => {
                if data.is_empty() {
                    self.error_message = Some("Empty image file".to_string());
                    self.is_processing = false;
                    return;
                }
                data
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to read file: {}", e));
                self.is_processing = false;
                return;
            }
        };

        // 모델 캐시 초기화 (필요한 경우)
        if self.inference_engine.is_none() {
            match UnifiedInferenceEngine::new() {
                Ok(mut engine) => {
                    // 선택된 모델로 설정
                    if let Err(e) = engine.set_model(self.selected_model.clone()) {
                        self.error_message = Some(format!("Failed to set model: {}", e));
                        self.is_processing = false;
                        return;
                    }
                    self.inference_engine = Some(engine);
                    println!("Inference engine initialized for model: {}", self.selected_model.display_name());
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to initialize inference engine: {}", e));
                    self.is_processing = false;
                    return;
                }
            }
        }

        // SQLite DB 초기화 (필요한 경우)
        if self.inference_db.is_none() {
            match InferenceDb::new() {
                Ok(db) => {
                    self.inference_db = Some(db);
                    println!("Inference DB initialized");
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to initialize inference DB: {}", e));
                    self.is_processing = false;
                    return;
                }
            }
        }

        // 이미지 경로를 문자열로 변환
        let _image_path_str = path.to_string_lossy().to_string();

        if let (Some(engine), Some(_db)) = (&mut self.inference_engine, &self.inference_db) {
            // 통합 추론 엔진을 사용하여 객체 검출 수행
            match engine.detect(&image_data) {
                Ok(result) => {
                    self.pre_nms_detections = result.detections.clone();
                    self.detections = result.detections;
                    self.selection = vec![true; self.detections.len()];
                    self.update_selection_by_confidence();

                    if result.inference_time_ms > 0.0 {
                        self.inference_time_ms = Some(result.inference_time_ms);
                    } else {
                        self.inference_time_ms = None;
                    }

                    // 결과 이미지를 텍스처로 로딩
                    self.load_texture(ctx, result.result_image);
                }
                Err(e) => {
                    self.error_message = Some(format!("Detection error: {}", e));
                }
            }
        }

        self.is_processing = false;
    }

    /// 강제 재추론(DB 캐시 무시)
    fn process_image_with_force(&mut self, ctx: &egui::Context, path: PathBuf) {
        // DB에서 해당 항목 삭제 후 일반 처리
        if let Some(db) = &self.inference_db {
            let image_path_str = path.to_string_lossy().to_string();
            let model_name = match &self.selected_model {
                ModelType::YoloV9(file_name) => file_name.clone(),
                ModelType::RfDetr => "rf_detr".to_string(),
            };
            if let Err(e) = db.delete_cache_entry(&image_path_str, &model_name) {
                eprintln!("Failed to delete from DB: {}", e);
                // DB 삭제 실패해도 계속 진행 (강제 재추론이므로)
            }
        }
        self.process_image(ctx, path);
    }

    fn update_selection_by_confidence(&mut self) {
        if self.detections.is_empty() {
            self.selection.clear();
            return;
        }

        // selection 벡터 크기 조정
        while self.selection.len() < self.detections.len() {
            self.selection.push(true);
        }
        if self.selection.len() > self.detections.len() {
            self.selection.truncate(self.detections.len());
        }

        // 신뢰도 임계값에 따라 선택 상태 업데이트
        for (i, detection) in self.detections.iter().enumerate() {
            if i < self.selection.len() {
                self.selection[i] = detection.confidence >= self.confidence_threshold;
            }
        }
    }

    fn reapply_nms_only(&mut self) {
        if self.pre_nms_detections.is_empty() {
            self.detections.clear();
            self.selection.clear();
            return;
        }

        self.detections = apply_nms_only(self.pre_nms_detections.clone(), self.nms_threshold);

        // selection 벡터 크기 조정
        self.selection = vec![true; self.detections.len()];
        self.update_selection_by_confidence();
    }

    /// 텍스처 로딩
    fn load_texture(&mut self, _ctx: &egui::Context, result_image: image::RgbImage) {
        let mut buffer = Vec::new();
        if let Ok(()) = result_image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Png,
        ) && let Ok(image) = image::load_from_memory(&buffer)
        {
            let rgba = image.to_rgba8();
            let size = rgba.dimensions();
            self.image_size = egui::vec2(size.0 as f32, size.1 as f32);

            // 텍스처 생성 및 저장
            let texture = _ctx.load_texture(
                "processed_image",
                egui::ColorImage::from_rgba_unmultiplied([
                    size.0 as usize,
                    size.1 as usize,
                ],
                &rgba),
                egui::TextureOptions::default(),
            );
            self.processed_image = Some(texture);
        }
    }
}
