/**
 * Cofe_log 專業級咖啡豆 AI 分類器 - 前端邏輯
 */

const state = {
  isStreaming: false,
  isDetecting: false,
  currentModel: null,
  stream: null,
  detectionInterval: null,
  fps: 2, // 每秒偵測次數
};

// DOM 元素
const webcamView = document.getElementById("webcam-view");
const visualPlaceholder = document.getElementById("visual-placeholder");
const cameraBtn = document.getElementById("camera-toggle-btn");
const modeBtn = document.getElementById("mode-toggle-btn");
const modelSelect = document.getElementById("model-select");
const resultLabel = document.getElementById("result-label");
const probContainer = document.getElementById("probability-container");
const lastUpdate = document.getElementById("last-update");
const infoArch = document.getElementById("info-arch");
const infoSize = document.getElementById("info-size");
const imageInput = document.getElementById("image-input");
const staticPreview = document.getElementById("static-preview");
const overlay = document.getElementById("detection-overlay");
const canvas = document.getElementById("frame-capture");
const cameraStatusDot = document.querySelector("#camera-status .dot");
const fullscreenBtn = document.getElementById("fullscreen-btn");
const uploadArea = document.getElementById("upload-area");
const visualViewport = document.querySelector(".visual-viewport");

// 初始化
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
});

function setupEventListeners() {
  cameraBtn.addEventListener("click", toggleCamera);
  modeBtn.addEventListener("click", toggleDetection);
  modelSelect.addEventListener("change", updateModelInfo);
  imageInput.addEventListener("change", handleFileUpload);

  // 頁籤切換
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn));
  });

  // 範例縮圖點擊
  document.querySelectorAll(".example-thumb").forEach((thumb) => {
    thumb.addEventListener("click", () => selectExample(thumb));
  });

  // 全螢幕切換
  fullscreenBtn.addEventListener("click", toggleFullscreen);

  // 拖放上傳
  setupDragAndDrop();
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    visualViewport.requestFullscreen().catch(err => {
      console.error(`無法切換全螢幕: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
}

function setupDragAndDrop() {
  ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  ["dragenter", "dragover"].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
      uploadArea.classList.add("drag-over");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
      uploadArea.classList.remove("drag-over");
    }, false);
  });

  uploadArea.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      const mockEvent = { target: { files: files } };
      handleFileUpload(mockEvent);
    }
  }, false);
}

function switchTab(clickedBtn) {
  const targetId = clickedBtn.dataset.target;
  
  // 更新按鈕狀態
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  clickedBtn.classList.add("active");

  // 更新內容顯示
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));
  document.getElementById(targetId).classList.add("active");
}

async function selectExample(thumb) {
  if (!modelSelect.value) {
    alert("請先選擇一個推理模型。");
    return;
  }

  // 更新選中樣式
  document.querySelectorAll(".example-thumb").forEach(t => t.classList.remove("active"));
  thumb.classList.add("active");

  // 停止相機（如果正在串流）
  if (state.isStreaming) stopCamera();

  // 更新預覽
  staticPreview.src = thumb.src;
  staticPreview.hidden = false;
  visualPlaceholder.hidden = true;
  webcamView.hidden = true;

  // 執行預測
  // 我們將圖片繪製到隱藏畫布中，以便複用 captureAndPredict 的邏輯
  const img = new Image();
  img.crossOrigin = "Anonymous";
  img.onload = () => {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    captureAndPredict(); // 此函數會從畫布讀取並發送 API
  };
  img.src = thumb.src;
}

// 相機控制
async function toggleCamera() {
  if (state.isStreaming) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: 640, height: 640 },
    });
    webcamView.srcObject = state.stream;
    webcamView.hidden = false;
    visualPlaceholder.hidden = true;
    staticPreview.hidden = true;
    state.isStreaming = true;
    cameraBtn.textContent = "關閉相機";
    cameraBtn.classList.replace("btn-secondary", "btn-danger");
    cameraStatusDot.classList.add("active");
  } catch (err) {
    console.error("相機啟動失敗:", err);
    alert("無法存取相機，請檢查權限設定。");
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    webcamView.srcObject = null;
    webcamView.hidden = true;
    visualPlaceholder.hidden = false;
    state.isStreaming = false;
    state.isDetecting = false;
    stopDetectionUI();
    cameraBtn.textContent = "開啟相機";
    cameraBtn.classList.replace("btn-danger", "btn-secondary");
    cameraStatusDot.classList.remove("active");
  }
}

// 偵測開關
function toggleDetection() {
  if (!state.isStreaming) {
    alert("請先開啟相機後再啟動物件偵測。");
    return;
  }
  if (!modelSelect.value) {
    alert("請選擇一個推理模型。");
    return;
  }

  state.isDetecting = !state.isDetecting;
  if (state.isDetecting) {
    startDetectionUI();
    runDetectionLoop();
  } else {
    stopDetectionUI();
  }
}

function startDetectionUI() {
  modeBtn.textContent = "即時偵測：開";
  modeBtn.classList.replace("btn-outline", "btn-primary");
  overlay.hidden = false;
}

function stopDetectionUI() {
  modeBtn.textContent = "即時偵測：關";
  modeBtn.classList.replace("btn-primary", "btn-outline");
  overlay.hidden = true;
  if (state.detectionInterval) {
    clearTimeout(state.detectionInterval);
  }
}

// 核心偵測迴圈
async function runDetectionLoop() {
  if (!state.isDetecting || !state.isStreaming) return;

  await captureAndPredict();
  
  // 排定下一次偵測
  state.detectionInterval = setTimeout(runDetectionLoop, 1000 / state.fps);
}

async function captureAndPredict() {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(webcamView, 0, 0, canvas.width, canvas.height);
  const base64Image = canvas.toDataURL("image/jpeg", 0.7);

  try {
    const response = await fetch("/api/predict_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_key: modelSelect.value,
        image: base64Image,
      }),
    });

    if (response.ok) {
      const result = await response.json();
      updateUI(result);
    }
  } catch (err) {
    console.error("預測請求失敗:", err);
  }
}

// UI 更新
function updateUI(data) {
  resultLabel.textContent = data.prediction.label;
  resultLabel.style.color = data.prediction.label === "good" ? "var(--success)" : "var(--danger)";
  
  lastUpdate.textContent = new Date().toLocaleTimeString();

  // 更新機率條
  probContainer.innerHTML = "";
  data.probabilities.forEach((item) => {
    const pct = (item.probability * 100).toFixed(1);
    const bar = document.createElement("div");
    bar.className = "prob-item";
    bar.innerHTML = `
      <div class="prob-info">
        <span>${item.label}</span>
        <span>${pct}%</span>
      </div>
      <div class="progress-bg">
        <div class="progress-fill" style="width: ${pct}%; background: ${item.label === 'good' ? 'var(--success)' : 'var(--danger)'}"></div>
      </div>
    `;
    probContainer.appendChild(bar);
  });

  // 智能視覺警示：瑕疵豆且置信度 > 90%
  const badProb = data.probabilities.find(p => p.label === 'bad')?.probability || 0;
  if (data.prediction.label === 'bad' && badProb > 0.9) {
    visualViewport.classList.add("alert-bad");
    // 3秒後自動移除警示效果
    setTimeout(() => {
      visualViewport.classList.remove("alert-bad");
    }, 3000);
  } else {
    visualViewport.classList.remove("alert-bad");
  }
}

function updateModelInfo() {
  const selectedOption = modelSelect.options[modelSelect.selectedIndex];
  // 這裡可以從後端獲取模型詳情，或是簡化處理
  infoArch.textContent = "Custom CNN";
  infoSize.textContent = "128x128";
}

// 處理手動上傳
async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  // 顯示預覽
  const reader = new FileReader();
  reader.onload = (event) => {
    staticPreview.src = event.target.result;
    staticPreview.hidden = false;
    visualPlaceholder.hidden = true;
    webcamView.hidden = true;
    if (state.isStreaming) stopCamera();
  };
  reader.readAsDataURL(file);

  if (!modelSelect.value) {
    alert("請選擇模型後再上傳。");
    return;
  }

  // 發送預測
  const formData = new FormData();
  formData.append("image", file);
  formData.append("model_key", modelSelect.value);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    if (response.ok) {
      const result = await response.json();
      updateUI(result);
    }
  } catch (err) {
    console.error("上傳預測失敗:", err);
  }
}
