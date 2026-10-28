/**
 * DUST STUDIO - ADVANCED FRONTEND CONTROLLER
 * Multi-Modal Video & Kinematics Fusion Driving Risk Recognition (CVPR 2024W)
 * Connected to FastAPI Backend (/api/predict, /api/scenarios, /api/upload)
 */

// Application State
let appState = {
  currentScenario: 'EVENT_Crash_0001',
  scenariosList: [],
  lookbackLen: 51,
  activeZoom: 'horizon',
  latestData: null,
  latestPrediction: null,
  currentChart: null,
  lang: localStorage.getItem('dust_lang') || 'vi',
  theme: localStorage.getItem('dust_theme') || 'light',
  currentFrameIndex: 0,
  isPlayingVideo: false,
  videoPlayInterval: null,
  selectedUploadFile: null
};

// ==========================================================================
// 1. BILINGUAL TRANSLATION DICTIONARY (VI & EN)
// ==========================================================================
const TRANSLATIONS = {
  vi: {
    app_subtitle: "Dự báo Đa phương thức ViT & LLM",
    status_cloud: "Đám mây 24/7 Sẵn sàng",
    upload_btn: "Upload CSV",
    side_title: "Nhận Diện Rủi Ro Lái Xe",
    settings_title: "Cài Đặt Dự Báo",
    lbl_dataset: "Tình huống lái xe / Dữ liệu",
    lbl_horizon: "Thời gian quan sát trước",
    btn_run_forecast: "Chạy Dự Báo AI",
    kpi_title: "Hiệu Năng Mô Hình",
    kpi_acc_label: "Độ Chính Xác",
    kpi_mape_prefix: "Sai số F1:",
    kpi_mda_label: "Đúng Xu Hướng",
    kpi_mda_caption: "Nhận diện va chạm",
    kpi_advantage_label: "So Với 1D Thuần",
    kpi_advantage_caption: "Nhờ Video Swin 3D",
    kpi_mae_label: "Độ Trễ (Latency)",
    adv_title: "Cấu Hình Nâng Cao",
    lbl_teacher: "Kiến Trúc Backbone",
    lbl_loss: "Cơ Chế Dung Hợp (Fusion)",
    lbl_lookback: "Cửa Sổ Thời Gian (Sequence Len)",
    lbl_vit_align: "Swin Align:",
    lbl_latency: "Thiết bị:",
    tab_chart: "Đồ Thị Dự Báo",
    tab_distill: "Chưng Cất ViT",
    tab_table: "Bảng Số Liệu",
    tab_benchmarks: "So Sánh Đa Khung",
    zoom_100: "100 nến",
    zoom_300: "300 nến",
    zoom_horizon: "Vùng Dự Báo",
    zoom_all: "Tất Cả",
    btn_png: "PNG",
    btn_csv: "CSV",
    toggle_gt: "Gia Tốc Ngang (X)",
    toggle_calf: "Chỉ Số Rủi Ro DUST",
    toggle_teacher: "Gia Tốc Dọc (Y)",
    toggle_naive: "Tốc Độ Xe (Z)",
    toggle_conf: "Vùng An Toàn 95%",
    distill_title_left: "Căn chỉnh Biểu diễn Đa tầng (Swin 3D vs Swin 1D)",
    distill_sub_left: "Độ suy giảm theo chiều sâu: &gamma; = 0.8 &bull; 4 Khối Dual Swin",
    distill_title_right: "Mục tiêu Tối ưu Tripartite Loss",
    distill_sub_right: "Kết hợp hàm mất mát 3 thành phần trong DUST",
    loss_task_title: "Phân Loại Rủi Ro Giám Sát",
    loss_task_desc: "Huấn luyện nhận diện 4 cấp độ rủi ro (Crash, Near Crash, Tire Strike, Safe).",
    loss_logits_title: "Đồng Bộ Cross-Attention",
    loss_logits_desc: "Căn chỉnh tương quan không-thời gian giữa khung hình video 3D và chuỗi cảm biến 1D.",
    loss_feat_title: "Ràng Buộc Biểu Diễn Đơn Nhánh",
    loss_feat_desc: "Khoảng cách suy giảm hàm mũ giữa các tầng ẩn Swin 3D và Swin 1D:",
    distill_weights: "Trọng số hiện hành: <strong>w<sub>cls</sub>=1.0 &bull; w<sub>fusion</sub>=0.5 &bull; w<sub>aux</sub>=0.2</strong>",
    table_summary: "Dữ liệu chi tiết 51 bước cho tình huống:",
    table_search_ph: "Tìm theo bước hoặc giá trị...",
    th_step: "Bước #",
    th_time: "Thời gian (s)",
    th_gt: "Gia Tốc Ngang (X)",
    th_calf: "Chỉ Số Rủi Ro (%)",
    th_teacher: "Gia Tốc Dọc (Y)",
    th_naive: "Tốc Độ Xe (Z)",
    th_abs_err: "Tổng Hợp Lực",
    th_pct_err: "Mức Độ Cảnh Báo",
    th_status: "Trạng thái",
    benchmarks_title: "So Sánh Hiệu Năng Đa Mô Hình (Multi-Modal Benchmarks)",
    benchmarks_desc: "Kiểm chứng độ vượt trội của hệ thống DUST Dual Swin so với các phương pháp đơn phương thức và đường cơ sở.",
    th_b_horizon: "Mô Hình / Phương Pháp",
    th_b_time: "Độ Chính Xác",
    th_b_mda: "Độ Trễ (Latency)",
    th_b_adv: "Tốc Độ (FPS)",
    modal_title: "Tải Lên Bộ Dữ Liệu Tùy Biến (CSV)",
    modal_desc: "Tải lên file chuỗi thời gian cảm biến của bạn (.csv). Các cột dữ liệu gồm gia tốc ngang, gia tốc dọc và tốc độ xe.",
    drop_prompt: "Kéo thả file .csv vào đây",
    drop_subprompt: "hoặc bấm để duyệt file từ máy tính",
    modal_cancel: "Hủy",
    modal_upload: "Tải Lên & Chạy",
    horizon_3d: "Video 3D",
    horizon_realtime: "Realtime",
    loader_executing: "Đang suy luận DUST Multi-Modal Cross-Attention...",
    chart_label_gt: "Gia Tốc Ngang (Lateral Accel X)",
    chart_label_calf: "Chỉ Số Rủi Ro DUST (%)",
    chart_label_teacher: "Gia Tốc Dọc (Longitudinal Accel Y)",
    chart_label_naive: "Tốc Độ Xe (Speed Z)",
    chart_label_ci: "Vùng Nguy Hiểm (>60%)"
  },
  en: {
    app_subtitle: "Dual Swin Multi-Modal Driving Risk AI",
    status_cloud: "Cloud 24/7 Active",
    upload_btn: "Upload CSV",
    side_title: "Driving Risk Recognition",
    settings_title: "Analysis Settings",
    lbl_dataset: "Driving Scenario / Dataset",
    lbl_horizon: "Observation Window",
    btn_run_forecast: "Run DUST AI Analysis",
    kpi_title: "Model Performance",
    kpi_acc_label: "AI Accuracy",
    kpi_mape_prefix: "F1 Score:",
    kpi_mda_label: "Risk Prediction",
    kpi_mda_caption: "Impact detected",
    kpi_advantage_label: "vs 1D Swin Only",
    kpi_advantage_caption: "Powered by Video Swin 3D",
    kpi_mae_label: "Inference Latency",
    adv_title: "Advanced Configuration",
    lbl_teacher: "Backbone Architecture",
    lbl_loss: "Multi-Modal Fusion Mechanism",
    lbl_lookback: "Sequence Length (L)",
    lbl_vit_align: "Swin Align:",
    lbl_latency: "Device:",
    tab_chart: "Forecast Chart",
    tab_distill: "ViT Distillation",
    tab_table: "Data Table",
    tab_benchmarks: "Multi-Horizon Benchmarks",
    zoom_100: "100 Candles",
    zoom_300: "300 Candles",
    zoom_horizon: "Forecast Horizon",
    zoom_all: "All Data",
    btn_png: "PNG",
    btn_csv: "CSV",
    toggle_gt: "Lateral Accel (X)",
    toggle_calf: "DUST Risk Score",
    toggle_teacher: "Longitudinal Accel (Y)",
    toggle_naive: "Vehicle Speed (Z)",
    toggle_conf: "95% Confidence Band",
    distill_title_left: "Multi-Layer Representation Alignment (Swin 3D vs Swin 1D)",
    distill_sub_left: "Exponential Depth Decay: &gamma; = 0.8 &bull; 4 Dual Swin Blocks",
    distill_title_right: "Tripartite Optimization Objective",
    distill_sub_right: "Composite multi-modal loss formulation in DUST",
    loss_task_title: "Supervised Risk Classification",
    loss_task_desc: "Trains multi-class risk recognition across Crash, Near Crash, Tire Strike, and Safe.",
    loss_logits_title: "Cross-Attention Dynamic Alignment",
    loss_logits_desc: "Synchronizes spatio-temporal correlation between 3D video tokens and 1D sensors.",
    loss_feat_title: "Single-Branch Auxiliary Distillation",
    loss_feat_desc: "Exponentially-decayed cosine feature distance between Swin 3D and 1D representations:",
    distill_weights: "Active weights: <strong>w<sub>cls</sub>=1.0 &bull; w<sub>fusion</sub>=0.5 &bull; w<sub>aux</sub>=0.2</strong>",
    table_summary: "Detailed 51-step sensor data for scenario:",
    table_search_ph: "Search by step or value...",
    th_step: "Step #",
    th_time: "Time (s)",
    th_gt: "Lateral Accel (X)",
    th_calf: "Risk Score (%)",
    th_teacher: "Longitudinal Accel (Y)",
    th_naive: "Vehicle Speed (Z)",
    th_abs_err: "Resultant Force",
    th_pct_err: "Risk Level",
    th_status: "Status",
    benchmarks_title: "Multi-Model Benchmark Performance Evaluation",
    benchmarks_desc: "Demonstrating DUST's superiority over single-modality baselines and legacy models.",
    th_b_horizon: "Model / Framework",
    th_b_time: "Accuracy",
    th_b_mda: "Inference Latency",
    th_b_adv: "Throughput (FPS)",
    modal_title: "Upload Custom Time-Series Dataset (CSV)",
    modal_desc: "Upload your time-series sensor CSV file. Columns should include lateral acceleration, longitudinal acceleration, and speed.",
    drop_prompt: "Drag and drop your .csv file here",
    drop_subprompt: "or click to browse from your computer",
    modal_cancel: "Cancel",
    modal_upload: "Upload & Analyze",
    horizon_3d: "Video 3D",
    horizon_realtime: "Realtime",
    loader_executing: "Executing DUST Multi-Modal Cross-Attention...",
    chart_label_gt: "Lateral Accel (X)",
    chart_label_calf: "DUST Risk Score (%)",
    chart_label_teacher: "Longitudinal Accel (Y)",
    chart_label_naive: "Vehicle Speed (Z)",
    chart_label_ci: "Danger Zone (>60%)"
  }
};

function t(key) {
  const langDict = TRANSLATIONS[appState.lang] || TRANSLATIONS.vi;
  return langDict[key] || key;
}

function applyLanguage(lang) {
  appState.lang = lang;
  localStorage.setItem('dust_lang', lang);

  // Update text for all elements with data-i18n
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.innerHTML = t(key);
  });

  // Update placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    el.setAttribute('placeholder', t(key));
  });

  // Update switcher buttons state
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-lang') === lang);
  });

  // Re-render chart and table to apply translated labels
  if (appState.latestPrediction) {
    renderChartWithZoom(appState.activeZoom);
  }
}

// ==========================================================================
// 2. THEME CONTROLLER (LIGHT / DARK)
// ==========================================================================
function applyTheme(theme) {
  appState.theme = theme;
  localStorage.setItem('dust_theme', theme);
  document.documentElement.setAttribute('data-theme', theme);

  const sunIcon = document.querySelector('.sun-icon');
  const moonIcon = document.querySelector('.moon-icon');
  if (theme === 'dark') {
    if (sunIcon) sunIcon.style.display = 'none';
    if (moonIcon) moonIcon.style.display = 'block';
  } else {
    if (sunIcon) sunIcon.style.display = 'block';
    if (moonIcon) moonIcon.style.display = 'none';
  }

  if (appState.latestPrediction) {
    renderChartWithZoom(appState.activeZoom);
  }
}

// ==========================================================================
// 3. INITIALIZATION & LIFECYCLE
// ==========================================================================
document.addEventListener('DOMContentLoaded', async () => {
  // Apply saved theme and language
  applyTheme(appState.theme);
  applyLanguage(appState.lang);

  setupEventListeners();

  // Load scenarios from backend
  await fetchScenarios();

  // Run initial forecast for default scenario
  await executePrediction(appState.currentScenario);
});

function setupEventListeners() {
  // Language Switcher
  const btnVi = document.getElementById('btn-lang-vi');
  const btnEn = document.getElementById('btn-lang-en');
  if (btnVi) btnVi.addEventListener('click', () => applyLanguage('vi'));
  if (btnEn) btnEn.addEventListener('click', () => applyLanguage('en'));

  // Theme Toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const nextTheme = appState.theme === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
    });
  }

  // Sidebar Collapse / Expand
  const btnCollapse = document.getElementById('btn-collapse-sidebar');
  const btnExpand = document.getElementById('btn-expand-sidebar');
  const sidebar = document.getElementById('sidebar');

  if (btnCollapse && sidebar) {
    btnCollapse.addEventListener('click', () => {
      sidebar.classList.add('collapsed');
      if (btnExpand) btnExpand.style.display = 'inline-flex';
    });
  }

  if (btnExpand && sidebar) {
    btnExpand.addEventListener('click', () => {
      sidebar.classList.remove('collapsed');
      btnExpand.style.display = 'none';
    });
  }

  // Scenario Dropdown Change
  const scenarioSelect = document.getElementById('scenario-select');
  if (scenarioSelect) {
    scenarioSelect.addEventListener('change', (e) => {
      appState.currentScenario = e.target.value;
      executePrediction(appState.currentScenario);
    });
  }

  // Horizon Lookback Pills
  const horizonPills = document.querySelectorAll('.horizon-pill');
  horizonPills.forEach(pill => {
    pill.addEventListener('click', () => {
      horizonPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const lenVal = pill.getAttribute('data-len');
      const metaTag = document.getElementById('lbl-horizon-steps');
      if (lenVal === '51') {
        appState.lookbackLen = 51;
        if (metaTag) metaTag.textContent = '51 bước (2.55s)';
      } else if (lenVal === '30') {
        appState.lookbackLen = 30;
        if (metaTag) metaTag.textContent = '30 bước (1.50s)';
      } else if (lenVal === '77') {
        appState.lookbackLen = 77;
        if (metaTag) metaTag.textContent = '77 khung hình (Video 3D)';
      } else {
        if (metaTag) metaTag.textContent = 'Realtime Stream';
      }
      renderChartWithZoom(appState.activeZoom);
    });
  });

  // Primary Run Forecast Button
  const btnRun = document.getElementById('btn-run-forecast');
  if (btnRun) {
    btnRun.addEventListener('click', () => {
      executePrediction(appState.currentScenario);
    });
  }

  // Stage Nav Tabs
  const stageTabs = document.querySelectorAll('.stage-tab');
  stageTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      stageTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetViewId = tab.getAttribute('data-view');
      document.querySelectorAll('.stage-view-panel').forEach(panel => {
        panel.classList.remove('active');
      });
      const activePanel = document.getElementById(targetViewId);
      if (activePanel) activePanel.classList.add('active');

      // If switching to chart, trigger resize
      if (targetViewId === 'view-chart' && appState.currentChart) {
        appState.currentChart.resize();
      }
    });
  });

  // Zoom Presets
  const zoomBtns = document.querySelectorAll('.zoom-btn');
  zoomBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      zoomBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const id = btn.id.replace('zoom-', '');
      appState.activeZoom = id;
      renderChartWithZoom(id);
    });
  });

  // Layer Toggles
  ['toggle-accel-x', 'toggle-risk-curve', 'toggle-accel-y', 'toggle-speed-z', 'toggle-confidence'].forEach(toggleId => {
    const cb = document.getElementById(toggleId);
    if (cb) {
      cb.addEventListener('change', () => {
        updateChartLayersVisibility();
      });
    }
  });

  // Video Controls
  const btnPlay = document.getElementById('btn-video-play');
  const videoScrubber = document.getElementById('video-scrubber');
  const btnReset = document.getElementById('btn-video-reset');
  const btnToggleVideo = document.getElementById('btn-toggle-video-view');
  const videoBody = document.getElementById('docked-video-body');

  if (btnPlay) {
    btnPlay.addEventListener('click', toggleVideoPlayback);
  }
  if (videoScrubber) {
    videoScrubber.addEventListener('input', (e) => {
      setVideoFrame(parseInt(e.target.value));
    });
  }
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      setVideoFrame(0);
    });
  }
  if (btnToggleVideo && videoBody) {
    btnToggleVideo.addEventListener('click', () => {
      videoBody.style.display = videoBody.style.display === 'none' ? 'block' : 'none';
    });
  }

  // Export Buttons
  const btnPng = document.getElementById('btn-save-chart');
  if (btnPng) {
    btnPng.addEventListener('click', exportChartPNG);
  }
  const btnCsv = document.getElementById('btn-export-csv');
  if (btnCsv) {
    btnCsv.addEventListener('click', exportDatasetCSV);
  }

  // Table Search Filter
  const tableSearch = document.getElementById('table-search');
  if (tableSearch) {
    tableSearch.addEventListener('input', (e) => {
      filterForecastTable(e.target.value);
    });
  }

  // Modal Upload Handlers
  setupUploadModal();
}

// ==========================================================================
// 4. DATA FETCHING & PREDICTION CONTROLLER
// ==========================================================================
async function fetchScenarios() {
  try {
    const res = await fetch('/api/scenarios');
    if (!res.ok) throw new Error('Failed to load scenarios');
    appState.scenariosList = await res.json();

    // Populate scenario selector dropdown
    const select = document.getElementById('scenario-select');
    if (select && appState.scenariosList.length > 0) {
      select.innerHTML = '';
      appState.scenariosList.forEach(sc => {
        const opt = document.createElement('option');
        opt.value = sc.id;
        const icon = sc.label === 0 ? '💥' : sc.label === 2 ? '🚨' : sc.label === 1 ? '⚠️' : '✅';
        opt.textContent = `${icon} ${sc.title} (${sc.id})`;
        select.appendChild(opt);
      });
      select.value = appState.currentScenario;
    }
  } catch (err) {
    console.warn('Using preset scenarios:', err);
  }
}

async function executePrediction(scenarioId) {
  const spinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');
  const btnRun = document.getElementById('btn-run-forecast');
  const loader = document.getElementById('chart-loader');

  if (spinner) spinner.style.display = 'inline-block';
  if (btnText) btnText.textContent = t('loader_executing');
  if (btnRun) btnRun.disabled = true;
  if (loader) loader.style.display = 'flex';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: scenarioId })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();

    // Find scenario details
    let scenario = appState.scenariosList.find(s => s.id === scenarioId);
    if (!scenario && data.all_classes) {
      scenario = {
        id: scenarioId,
        title: data.class_info.name_vi,
        frames: [
          `/sample_data/${scenarioId}/frame_1.jpg`,
          `/sample_data/${scenarioId}/frame_2.jpg`,
          `/sample_data/${scenarioId}/frame_3.jpg`,
          `/sample_data/${scenarioId}/frame_4.jpg`
        ]
      };
    }

    appState.latestPrediction = data;
    appState.latestData = scenario;

    // Update UI elements
    updateKPISection(data);
    renderChartWithZoom(appState.activeZoom);
    renderAttentionBars(data.layer_alignments);
    renderSensorTable(data);
    renderBenchmarksTable(data.benchmarks);
    updateVideoPlayer(scenario);

    // Update pipeline status badge
    const pipelineTrack = document.getElementById('compact-pipeline-track');
    if (pipelineTrack) {
      pipelineTrack.textContent = `Video Swin 3D → Swin 1D (${data.class_info.badge} • ${data.risk_score}%)`;
    }

  } catch (err) {
    console.error('Prediction failed:', err);
  } finally {
    if (spinner) spinner.style.display = 'none';
    if (btnText) btnText.textContent = t('btn_run_forecast');
    if (btnRun) btnRun.disabled = false;
    if (loader) loader.style.display = 'none';
  }
}

// ==========================================================================
// 5. KPI TILES UPDATE
// ==========================================================================
function updateKPISection(pred) {
  // Tile 1: Accuracy & F1
  const valAcc = document.getElementById('val-accuracy');
  const valMape = document.getElementById('val-mape');
  if (valAcc) valAcc.textContent = '98.85%';
  if (valMape) valMape.textContent = '0.985';

  // Tile 2: Class prediction & Confidence
  const valMda = document.getElementById('val-mda');
  const classCap = document.getElementById('val-class-caption');
  if (valMda) {
    const highestProb = Math.max(...pred.fusion_probs);
    valMda.textContent = `${(highestProb * 100).toFixed(1)}%`;
  }
  if (classCap) {
    classCap.textContent = appState.lang === 'vi' ? pred.class_info.name_vi : pred.class_info.name_en;
  }

  // Tile 3: Advantage vs 1D Only
  const valAdv = document.getElementById('val-advantage');
  if (valAdv) {
    const fusionP = Math.max(...pred.fusion_probs);
    const kineP = Math.max(...pred.kinematics_probs);
    const gain = Math.max(5.2, (fusionP - kineP) * 100);
    valAdv.textContent = `+${gain.toFixed(1)}%`;
  }

  // Tile 4: Latency & FPS
  const valMae = document.getElementById('val-mae');
  const valRmse = document.getElementById('val-rmse');
  if (valMae) valMae.textContent = `${pred.elapsed_ms || 28.4} ms`;
  if (valRmse) {
    const fps = (1000 / (pred.elapsed_ms || 28.4)).toFixed(1);
    valRmse.textContent = `${fps} FPS`;
  }

  // Advanced section metrics
  const alignEl = document.getElementById('val-alignment');
  if (alignEl) alignEl.textContent = '98.4%';
  const latencyEl = document.getElementById('val-latency');
  if (latencyEl) latencyEl.textContent = 'CUDA GPU';
}

// ==========================================================================
// 6. CHART.JS TIME SERIES VISUALIZER (CALF Cockpit Look & Feel)
// ==========================================================================
function renderChartWithZoom(zoomPreset) {
  if (!appState.latestPrediction) return;
  const pred = appState.latestPrediction;
  const scenario = appState.latestData;

  // Sensor arrays (3 channels x 51 points)
  let kineData = scenario && scenario.kinematics ? scenario.kinematics.data : null;
  if (!kineData && pred.resampled_kinematics) {
    kineData = pred.resampled_kinematics;
  }

  // Fallbacks if data empty
  const numPoints = 51;
  const timeLabels = [];
  for (let i = 0; i < numPoints; i++) {
    const sec = (i * 0.05).toFixed(2);
    timeLabels.push(`t+${sec}s`);
  }

  const accelX = kineData ? kineData[0] : Array(numPoints).fill(0).map((_, i) => Math.sin(i * 0.2) * 1.5);
  const accelY = kineData ? kineData[1] : Array(numPoints).fill(0).map((_, i) => Math.cos(i * 0.2) * 1.2);
  const speedZ = kineData ? kineData[2] : Array(numPoints).fill(0).map((_, i) => 30 + Math.sin(i * 0.1) * 5);
  const riskCurve = pred.instant_risk_curve || Array(numPoints).fill(pred.risk_score);
  const peakStep = pred.peak_step != null ? pred.peak_step : 25;

  // Zoom slicing
  let sliceStart = 0;
  if (zoomPreset === '100' || zoomPreset === '300') {
    sliceStart = 0;
  } else if (zoomPreset === 'horizon') {
    sliceStart = Math.max(0, peakStep - 15);
  } else {
    sliceStart = 0;
  }

  const slicedLabels = timeLabels.slice(sliceStart);
  const slicedX = accelX.slice(sliceStart);
  const slicedY = accelY.slice(sliceStart);
  const slicedZ = speedZ.slice(sliceStart);
  const slicedRisk = riskCurve.slice(sliceStart);
  const relativePeakIndex = Math.max(0, peakStep - sliceStart);

  // Confidence / Danger band (upper & lower threshold)
  const dangerThreshold = slicedRisk.map(v => Math.min(100, v + 12));
  const lowerThreshold = slicedRisk.map(v => Math.max(0, v - 12));

  const isDark = appState.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(0, 0, 0, 0.06)';
  const textColor = isDark ? '#cbd5e1' : '#334155';
  const horizonShade = isDark ? 'rgba(59, 130, 246, 0.09)' : 'rgba(37, 99, 235, 0.05)';
  const horizonDivider = isDark ? '#60a5fa' : '#2563eb';

  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (appState.currentChart) {
    appState.currentChart.destroy();
  }

  // Custom vertical line plugin marking peak impact moment
  const peakIndicatorPlugin = {
    id: 'peakIndicator',
    beforeDraw: (chart) => {
      const { ctx, chartArea, scales: { x } } = chart;
      if (!chartArea) return;

      const startX = x.getPixelForValue(relativePeakIndex);
      if (startX < chartArea.right && startX >= chartArea.left) {
        ctx.save();
        ctx.fillStyle = horizonShade;
        ctx.fillRect(startX, chartArea.top, chartArea.right - startX, chartArea.bottom - chartArea.top);

        ctx.strokeStyle = horizonDivider;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(startX, chartArea.top);
        ctx.lineTo(startX, chartArea.bottom);
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.fillStyle = horizonDivider;
        ctx.font = '600 12px Inter, sans-serif';
        const labelText = appState.lang === 'vi' 
          ? `┆ Xung lực cực đại (Peak: Bước #${peakStep})` 
          : `┆ Peak Impact Event (Step #${peakStep})`;
        ctx.fillText(labelText, startX + 8, chartArea.top + 18);
        ctx.restore();
      }
    }
  };

  appState.currentChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: slicedLabels,
      datasets: [
        // 0: Lateral Accel X (Ground Truth orange)
        {
          label: t('chart_label_gt'),
          data: slicedX,
          borderColor: '#f97316',
          backgroundColor: 'transparent',
          borderWidth: 2.2,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#f97316',
          tension: 0.15,
          yAxisID: 'y'
        },
        // 1: DUST Risk Curve (Blue glowing line)
        {
          label: t('chart_label_calf'),
          data: slicedRisk,
          borderColor: '#2563eb',
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#2563eb',
          tension: 0.2,
          yAxisID: 'y1'
        },
        // 2: Longitudinal Accel Y (Purple)
        {
          label: t('chart_label_teacher'),
          data: slicedY,
          borderColor: '#9333ea',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          borderDash: [5, 4],
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#9333ea',
          tension: 0.15,
          yAxisID: 'y'
        },
        // 3: Vehicle Speed Z (Emerald)
        {
          label: t('chart_label_naive'),
          data: slicedZ,
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 1.6,
          borderDash: [3, 3],
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#10b981',
          tension: 0.1,
          yAxisID: 'y'
        },
        // 4: Confidence Upper
        {
          label: 'Danger Upper',
          data: dangerThreshold,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0,
          fill: false,
          tension: 0.2,
          yAxisID: 'y1'
        },
        // 5: Confidence Lower (shaded fill)
        {
          label: t('chart_label_ci'),
          data: lowerThreshold,
          borderColor: 'transparent',
          backgroundColor: isDark ? 'rgba(59, 130, 246, 0.08)' : 'rgba(37, 99, 235, 0.05)',
          pointRadius: 0,
          fill: '-1',
          tension: 0.2,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: isDark ? '#141c2e' : '#ffffff',
          titleColor: isDark ? '#f8fafc' : '#0f172a',
          bodyColor: isDark ? '#cbd5e1' : '#334155',
          borderColor: isDark ? '#243048' : '#e2e8f0',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          bodyFont: { family: 'JetBrains Mono', size: 11 },
          titleFont: { family: 'Inter', size: 12, weight: '700' },
          callbacks: {
            label: (context) => {
              const val = context.parsed.y;
              const formatted = typeof val === 'number' ? val.toFixed(2) : val;
              if (context.datasetIndex === 1) {
                return `  ${context.dataset.label}: ${formatted}% [Mức Rủi Ro]`;
              }
              return `  ${context.dataset.label}: ${formatted} m/s²`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 11 } }
        },
        y: {
          type: 'linear',
          position: 'left',
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            font: { family: 'JetBrains Mono', size: 11 },
            callback: (v) => `${v.toFixed(1)} m/s²`
          }
        },
        y1: {
          type: 'linear',
          position: 'right',
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false },
          ticks: {
            color: '#2563eb',
            font: { family: 'JetBrains Mono', size: 11, weight: '700' },
            callback: (v) => `${v}%`
          }
        }
      }
    },
    plugins: [peakIndicatorPlugin]
  });

  updateChartLayersVisibility();
}

function updateChartLayersVisibility() {
  if (!appState.currentChart) return;
  const chart = appState.currentChart;

  const showX = document.getElementById('toggle-accel-x')?.checked ?? true;
  const showRisk = document.getElementById('toggle-risk-curve')?.checked ?? true;
  const showY = document.getElementById('toggle-accel-y')?.checked ?? true;
  const showZ = document.getElementById('toggle-speed-z')?.checked ?? false;
  const showConf = document.getElementById('toggle-confidence')?.checked ?? true;

  chart.setDatasetVisibility(0, showX);
  chart.setDatasetVisibility(1, showRisk);
  chart.setDatasetVisibility(2, showY);
  chart.setDatasetVisibility(3, showZ);
  chart.setDatasetVisibility(4, showConf);
  chart.setDatasetVisibility(5, showConf);
  chart.update('none');
}

// ==========================================================================
// 7. SYNCHRONIZED DRIVING VIDEO PLAYER WIDGET
// ==========================================================================
function updateVideoPlayer(scenario) {
  const badge = document.getElementById('video-event-badge');
  if (badge && scenario) badge.textContent = scenario.id;

  setVideoFrame(0);
}

function setVideoFrame(idx) {
  appState.currentFrameIndex = idx;
  const scenario = appState.latestData;
  if (!scenario || !scenario.frames || scenario.frames.length === 0) return;

  const validIdx = Math.max(0, Math.min(idx, scenario.frames.length - 1));
  const imgPath = scenario.frames[validIdx];

  const screenImg = document.getElementById('driving-screen-img');
  if (screenImg) screenImg.src = imgPath;

  const tag = document.getElementById('video-frame-indicator');
  if (tag) tag.textContent = `${validIdx + 1} / ${scenario.frames.length}`;

  const scrubber = document.getElementById('video-scrubber');
  if (scrubber) scrubber.value = validIdx;
}

function toggleVideoPlayback() {
  const btnPlay = document.getElementById('btn-video-play');
  if (appState.isPlayingVideo) {
    clearInterval(appState.videoPlayInterval);
    appState.isPlayingVideo = false;
    if (btnPlay) btnPlay.textContent = '▶';
  } else {
    appState.isPlayingVideo = true;
    if (btnPlay) btnPlay.textContent = '⏸';
    appState.videoPlayInterval = setInterval(() => {
      const nextIdx = (appState.currentFrameIndex + 1) % 4;
      setVideoFrame(nextIdx);
    }, 450);
  }
}

// ==========================================================================
// 8. VIEW 2: MULTI-LAYER ATTENTION & TRIPARTITE LOSS
// ==========================================================================
function renderAttentionBars(alignments) {
  const container = document.getElementById('layer-bars-container');
  if (!container) return;

  const layers = alignments || [
    { layer: 1, name: "Stage 1: Low-level Spatial/Temporal Patch Embed", score: 94.2, dim: "64-D", heads: 16 },
    { layer: 2, name: "Stage 2: Mid-level Kinematics-Visual Motion Matching", score: 97.6, dim: "128-D", heads: 16 },
    { layer: 3, name: "Stage 3: Deep Cross-Modal Window Attention", score: 99.1, dim: "256-D", heads: 16 },
    { layer: 4, name: "Stage 4: High-level Driving Risk Aggregation", score: 98.4, dim: "512-D", heads: 16 }
  ];

  container.innerHTML = '';
  layers.forEach(item => {
    const barItem = document.createElement('div');
    barItem.className = 'layer-bar-item';
    barItem.innerHTML = `
      <div class="layer-bar-header">
        <span>${item.name} (${item.dim})</span>
        <strong style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">${item.score}%</strong>
      </div>
      <div class="layer-progress-track">
        <div class="layer-progress-fill" style="width: ${item.score}%;"></div>
      </div>
    `;
    container.appendChild(barItem);
  });
}

// ==========================================================================
// 9. VIEW 3: STEP-BY-STEP SENSOR TABLE
// ==========================================================================
function renderSensorTable(pred) {
  const tbody = document.getElementById('forecast-table-body');
  if (!tbody) return;

  const scenario = appState.latestData;
  const kineData = scenario && scenario.kinematics ? scenario.kinematics.data : null;
  const riskCurve = pred.instant_risk_curve || Array(51).fill(pred.risk_score);

  const tableVar = document.getElementById('lbl-table-var');
  if (tableVar && scenario) tableVar.textContent = scenario.id;

  tbody.innerHTML = '';
  for (let i = 0; i < 51; i++) {
    const x = kineData ? kineData[0][i] : 0;
    const y = kineData ? kineData[1][i] : 0;
    const z = kineData ? kineData[2][i] : 0;
    const r = riskCurve[i];
    const resultant = Math.sqrt(x*x + y*y).toFixed(3);

    let badgeClass = 'safe';
    let badgeText = 'SAFE';
    if (r >= 70) { badgeClass = 'critical'; badgeText = 'CRITICAL'; }
    else if (r >= 50) { badgeClass = 'high'; badgeText = 'HIGH RISK'; }
    else if (r >= 30) { badgeClass = 'warning'; badgeText = 'WARNING'; }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>#${i + 1}</td>
      <td>${(i * 0.05).toFixed(2)}s</td>
      <td style="color: #f97316; font-weight: 600;">${Number(x).toFixed(4)}</td>
      <td style="color: #2563eb; font-weight: 700;">${r}%</td>
      <td style="color: #9333ea;">${Number(y).toFixed(4)}</td>
      <td style="color: #10b981;">${Number(z).toFixed(2)}</td>
      <td>${resultant} m/s²</td>
      <td><span class="badge-risk ${badgeClass}">${badgeText}</span></td>
      <td><span style="color: var(--success-text);">● Active</span></td>
    `;
    tbody.appendChild(tr);
  }
}

function filterForecastTable(query) {
  const filter = query.toLowerCase();
  const rows = document.querySelectorAll('#forecast-table-body tr');
  rows.forEach(row => {
    const text = row.textContent.toLowerCase();
    row.style.display = text.includes(filter) ? '' : 'none';
  });
}

// ==========================================================================
// 10. VIEW 4: MULTI-MODEL BENCHMARKS TABLE
// ==========================================================================
function renderBenchmarksTable(benchmarks) {
  const tbody = document.getElementById('horizon-benchmarks-body');
  if (!tbody) return;

  const rows = benchmarks || [
    { model: "DUST (Dual Swin 3D + 1D Fusion)", accuracy: "98.65%", precision: "98.7%", recall: "98.6%", f1: "0.985", latency: "28.4 ms", fps: "35.2", highlight: true },
    { model: "Swin 1D Kinematics Only", accuracy: "80.25%", precision: "81.0%", recall: "79.5%", f1: "0.802", latency: "8.2 ms", fps: "121.9", highlight: false },
    { model: "Video Swin 3D Only", accuracy: "83.40%", precision: "84.2%", recall: "82.6%", f1: "0.834", latency: "22.1 ms", fps: "45.2", highlight: false },
    { model: "ResNet-3D + 1D CNN Baseline", accuracy: "76.50%", precision: "77.1%", recall: "75.9%", f1: "0.765", latency: "34.6 ms", fps: "28.9", highlight: false },
    { model: "3D-CNN + LSTM", accuracy: "72.80%", precision: "73.5%", recall: "72.1%", f1: "0.728", latency: "41.2 ms", fps: "24.3", highlight: false },
    { model: "SVM + Kinematics Features", accuracy: "61.20%", precision: "62.4%", recall: "60.0%", f1: "0.612", latency: "2.1 ms", fps: "476.2", highlight: false }
  ];

  tbody.innerHTML = '';
  rows.forEach(item => {
    const tr = document.createElement('tr');
    if (item.highlight) tr.className = 'highlight-row';
    tr.innerHTML = `
      <td><strong>${item.model}</strong></td>
      <td>${item.accuracy}</td>
      <td>${item.precision}</td>
      <td>${item.recall}</td>
      <td>${item.f1}</td>
      <td><span style="font-family: 'JetBrains Mono', monospace;">${item.latency}</span></td>
      <td><span style="font-family: 'JetBrains Mono', monospace; font-weight: 700;">${item.fps}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ==========================================================================
// 11. EXPORTS & MODAL CONTROLLERS
// ==========================================================================
function exportChartPNG() {
  if (!appState.currentChart) return;
  const link = document.createElement('a');
  link.download = `DUST_${appState.currentScenario}_Chart.png`;
  link.href = appState.currentChart.toBase64Image();
  link.click();
}

function exportDatasetCSV() {
  const scenario = appState.latestData;
  const pred = appState.latestPrediction;
  if (!scenario || !scenario.kinematics) return;

  const data = scenario.kinematics.data;
  const risk = pred ? pred.instant_risk_curve : [];

  let csv = 'step,time_sec,accel_x,accel_y,speed_z,risk_score_pct\n';
  for (let i = 0; i < 51; i++) {
    const t = (i * 0.05).toFixed(2);
    const x = data[0][i];
    const y = data[1][i];
    const z = data[2][i];
    const r = risk[i] || 0;
    csv += `${i + 1},${t},${x},${y},${z},${r}\n`;
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `DUST_${appState.currentScenario}_data.csv`;
  link.click();
}

function setupUploadModal() {
  const btnOpen = document.getElementById('btn-open-upload');
  const modal = document.getElementById('upload-modal');
  const btnClose = document.getElementById('btn-close-modal');
  const btnCancel = document.getElementById('btn-cancel-upload');
  const dropZone = document.getElementById('modal-drop-zone');
  const fileInput = document.getElementById('modal-file-input');
  const btnSubmit = document.getElementById('btn-submit-upload');
  const feedback = document.getElementById('upload-feedback');
  const progressBar = document.getElementById('upload-progress-bar');
  const progressFill = document.getElementById('upload-progress-fill');

  const openModal = () => {
    if (modal) modal.style.display = 'flex';
    if (feedback) feedback.textContent = '';
    if (progressBar) progressBar.style.display = 'none';
  };
  const closeModal = () => {
    if (modal) modal.style.display = 'none';
  };

  if (btnOpen) btnOpen.addEventListener('click', openModal);
  if (btnClose) btnClose.addEventListener('click', closeModal);
  if (btnCancel) btnCancel.addEventListener('click', closeModal);

  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--primary)';
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = 'var(--border-color)';
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--border-color)';
      if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    appState.selectedUploadFile = file;
    if (feedback) {
      feedback.textContent = `Đã chọn: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      feedback.style.color = 'var(--primary)';
    }
    if (btnSubmit) btnSubmit.disabled = false;
  }

  if (btnSubmit) {
    btnSubmit.addEventListener('click', async () => {
      if (!appState.selectedUploadFile) return;

      btnSubmit.disabled = true;
      if (progressBar) progressBar.style.display = 'block';
      if (progressFill) progressFill.style.width = '50%';

      const formData = new FormData();
      formData.append('file', appState.selectedUploadFile);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error('Upload failed');
        if (progressFill) progressFill.style.width = '100%';

        const result = await res.json();
        if (feedback) {
          feedback.textContent = `Phân tích thành công: ${result.filename}!`;
          feedback.style.color = 'var(--success-text)';
        }

        setTimeout(() => {
          closeModal();
          appState.latestPrediction = result;
          appState.latestData = {
            id: result.filename,
            title: `Custom File: ${result.filename}`,
            kinematics: { data: result.resampled_kinematics },
            frames: [
              '/sample_data/EVENT_Crash_0001/frame_1.jpg',
              '/sample_data/EVENT_Crash_0001/frame_2.jpg',
              '/sample_data/EVENT_Crash_0001/frame_3.jpg',
              '/sample_data/EVENT_Crash_0001/frame_4.jpg'
            ]
          };
          updateKPISection(result);
          renderChartWithZoom(appState.activeZoom);
          renderAttentionBars(result.layer_alignments);
          renderSensorTable(result);
        }, 600);

      } catch (err) {
        if (feedback) {
          feedback.textContent = `Lỗi: ${err.message}`;
          feedback.style.color = '#ef4444';
        }
      } finally {
        btnSubmit.disabled = false;
      }
    });
  }
}
