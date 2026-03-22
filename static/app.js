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

// 初始化
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
});

function setupEventListeners() {
  cameraBtn.addEventListener("click", toggleCamera);
  modeBtn.addEventListener("click", toggleDetection);
  modelSelect.addEventListener("change", updateModelInfo);
  imageInput.addEventListener("change", handleFileUpload);
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
