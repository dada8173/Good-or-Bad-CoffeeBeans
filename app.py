import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny, resnet18

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB 上傳上限

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent / "models"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

BEAN_LABELS = {
    "ethiopia_washed": "衣索比亞水洗豆",
    "kenya_natural": "肯亞日曬豆",
    "honduras_natural": "宏都拉斯日曬豆",
}

ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "convnext_tiny": "ConvNeXt-Tiny",
    "custom": "Custom CNN",
    "ultrafast": "UltraFast CNN",
}

MODEL_NAME_PATTERN = re.compile(
    r"^(?P<bean>.+)_(?P<arch>resnet18|convnext_tiny|custom|ultrafast)(?P<suffix>_Noback)?_best_model$"
)


@dataclass
class ModelInfo:
    key: str
    display_name: str
    bean_type: str
    architecture: str
    img_size: int
    classes: List[str]
    path: Path
    description: Optional[str] = None
    config_path: Optional[Path] = None

    @property
    def bean_display_name(self) -> str:
        return BEAN_LABELS.get(self.bean_type, self.bean_type.replace("_", " ").title())

    @property
    def architecture_label(self) -> str:
        return ARCHITECTURE_LABELS.get(self.architecture, self.architecture)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "bean_type": self.bean_type,
            "bean_display_name": self.bean_display_name,
            "architecture": self.architecture,
            "architecture_label": self.architecture_label,
            "img_size": self.img_size,
            "classes": self.classes,
            "model_file": self.path.name,
            "has_weights": self.path.exists(),
            "description": self.description,
            "config_file": self.config_path.name if self.config_path else None,
        }


MODEL_CACHE: Dict[str, nn.Module] = {}
TRANSFORM_CACHE: Dict[int, transforms.Compose] = {}


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    if width > height:
        diff = width - height
        padding = (0, diff // 2, 0, diff - diff // 2)
    else:
        diff = height - width
        padding = (diff // 2, 0, diff - diff // 2, 0)
    return ImageOps.expand(image, border=padding, fill=fill)


def build_transform(img_size: int) -> transforms.Compose:
    if img_size not in TRANSFORM_CACHE:
        TRANSFORM_CACHE[img_size] = transforms.Compose(
            [
                transforms.Lambda(lambda img: pad_to_square(img, fill=0)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    return TRANSFORM_CACHE[img_size]


def build_model(architecture: str, num_classes: int, img_size: int) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "convnext_tiny":
        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "custom":
        class CustomCNN(nn.Module):
            def __init__(self, num_classes: int, img_size: int) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                feature_dim = 128 * (img_size // 8) * (img_size // 8)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.features(x))

        return CustomCNN(num_classes, img_size)

    if architecture == "ultrafast":
        class UltraFastCNN(nn.Module):
            def __init__(self, num_classes: int) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 8, 5, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, 5, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.features(x))

        return UltraFastCNN(num_classes)

    raise ValueError(f"未知的模型架構：{architecture}")


def load_config_from_json(config_path: Path) -> Optional[ModelInfo]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("無法讀取模型設定 %s：%s", config_path.name, exc)
        return None

    required_fields = {"bean_type", "architecture", "img_size", "classes"}
    missing = required_fields - data.keys()
    if missing:
        logging.warning("設定檔 %s 缺少欄位：%s", config_path.name, ", ".join(sorted(missing)))
        return None

    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        logging.warning("設定檔 %s 的 classes 格式不正確", config_path.name)
        return None

    model_file = data.get("model_file")
    model_path = (
        (config_path.parent / model_file).resolve() if model_file else config_path.with_suffix(".pth").resolve()
    )

    display_name = data.get("display_name")
    if not display_name:
        bean_display = BEAN_LABELS.get(data["bean_type"], data["bean_type"].replace("_", " ").title())
        arch_label = ARCHITECTURE_LABELS.get(data["architecture"], data["architecture"])
        display_name = f"{bean_display} - {arch_label}"

    key = data.get("key") or config_path.stem

    return ModelInfo(
        key=key,
        display_name=display_name,
        bean_type=data["bean_type"],
        architecture=data["architecture"],
        img_size=int(data["img_size"]),
        classes=[str(cls) for cls in classes],
        path=model_path,
        description=data.get("description"),
        config_path=config_path,
    )


def infer_info_from_filename(model_path: Path) -> Optional[ModelInfo]:
    match = MODEL_NAME_PATTERN.match(model_path.stem)
    if not match:
        logging.info("跳過無法推斷資訊的模型檔案：%s", model_path.name)
        return None

    bean_type = match.group("bean")
    architecture = match.group("arch")
    suffix = match.group("suffix") or ""

    bean_display = BEAN_LABELS.get(bean_type, bean_type.replace("_", " ").title())
    arch_label = ARCHITECTURE_LABELS.get(architecture, architecture)
    display_name = f"{bean_display} - {arch_label}"
    if suffix:
        display_name += " (Noback)"

    classes = ["bad", "good"]
    description = "自動從檔名推斷設定，分類標籤預設為 bad / good。"

    return ModelInfo(
        key=model_path.stem,
        display_name=display_name,
        bean_type=bean_type,
        architecture=architecture,
        img_size=128,
        classes=classes,
        path=model_path.resolve(),
        description=description,
        config_path=None,
    )


def load_model_infos() -> List[ModelInfo]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    infos: List[ModelInfo] = []

    for config_path in sorted(MODEL_DIR.glob("*.json")):
        info = load_config_from_json(config_path)
        if info:
            infos.append(info)

    known_paths = {info.path for info in infos}

    for model_path in sorted(MODEL_DIR.glob("*.pth")):
        resolved = model_path.resolve()
        if resolved in known_paths:
            continue
        info = infer_info_from_filename(resolved)
        if info:
            infos.append(info)

    unique_infos: Dict[str, ModelInfo] = {}
    for info in infos:
        if info.key in unique_infos:
            logging.warning("模型 key 重複：%s，僅保留第一個設定。", info.key)
            continue
        unique_infos[info.key] = info

    return sorted(unique_infos.values(), key=lambda item: item.display_name.lower())


MODEL_INFOS = load_model_infos()
MODEL_LOOKUP = {info.key: info for info in MODEL_INFOS}


def get_or_load_model(model_key: str) -> nn.Module:
    if model_key not in MODEL_LOOKUP:
        raise KeyError(f"未知的模型 key：{model_key}")

    if model_key not in MODEL_CACHE:
        info = MODEL_LOOKUP[model_key]
        logging.info("載入模型：%s", info.display_name)
        model = build_model(info.architecture, len(info.classes), info.img_size)
        try:
            state_dict = torch.load(info.path, map_location=DEVICE)
        except FileNotFoundError as exc:
            logging.error("找不到模型檔案：%s", info.path)
            raise exc

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        MODEL_CACHE[model_key] = model

    return MODEL_CACHE[model_key]


def predict_image(model_key: str, image: Image.Image) -> Dict[str, object]:
    info = MODEL_LOOKUP[model_key]
    model = get_or_load_model(model_key)
    transform = build_transform(info.img_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

    results = [
        {"label": label, "probability": float(prob)}
        for label, prob in zip(info.classes, probabilities)
    ]
    prediction = max(results, key=lambda item: item["probability"])

    return {
        "prediction": prediction,
        "probabilities": results,
        "model": info.to_metadata(),
    }


@app.route("/")
def index():
    return render_template("index.html", models=MODEL_INFOS)


@app.post("/api/predict")
def api_predict():
    if not MODEL_LOOKUP:
        return jsonify({"error": "尚未在 models 資料夾中找到可用的模型。"}), 400

    model_key = request.form.get("model_key")
    if not model_key:
        return jsonify({"error": "請選擇要使用的模型。"}), 400

    if model_key not in MODEL_LOOKUP:
        return jsonify({"error": f"未知的模型：{model_key}"}), 400

    file_storage = request.files.get("image")
    if file_storage is None or file_storage.filename == "":
        return jsonify({"error": "請選擇要上傳的咖啡豆照片。"}), 400

    if not is_allowed_file(file_storage.filename):
        return jsonify({"error": "檔案格式僅支援 JPG、JPEG、PNG、BMP、WEBP。"}), 400

    try:
        image = Image.open(file_storage.stream)
    except Exception:
        return jsonify({"error": "無法讀取圖片，請確認檔案是否為有效的影像格式。"}), 400

    try:
        result = predict_image(model_key, image)
    except FileNotFoundError:
        info = MODEL_LOOKUP[model_key]
        return jsonify({"error": f"找不到模型權重檔案：{info.path.name}"}), 500
    except RuntimeError as exc:
        logging.exception("模型推論失敗：%s", exc)
        return jsonify({"error": f"模型推論失敗：{exc}"}), 500

    return jsonify(result)


@app.post("/api/predict_frame")
def api_predict_frame():
    """處理即時影像幀 (Base64)"""
    if not MODEL_LOOKUP:
        return jsonify({"error": "尚未載入模型。"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少資料體。"}), 400

    model_key = data.get("model_key")
    image_b64 = data.get("image")

    if not model_key or model_key not in MODEL_LOOKUP:
        return jsonify({"error": "無效的模型 key。"}), 400
    if not image_b64:
        return jsonify({"error": "缺少影像資料。"}), 400

    try:
        # 去除 header (data:image/jpeg;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        result = predict_image(model_key, image)
        return jsonify(result)
    except Exception as exc:
        logging.exception("幀預測失敗：%s", exc)
        return jsonify({"error": str(exc)}), 500


@app.get("/api/models")
def api_models():
    return jsonify({"models": [info.to_metadata() for info in MODEL_INFOS]})


if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    logging.info("啟動網頁服務：http://%s:%s", host, port)
    app.run(host=host, port=port, debug=False)
