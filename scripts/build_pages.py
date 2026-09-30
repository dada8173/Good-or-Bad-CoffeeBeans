"""Export verified browser models and build a self-contained static demo."""

import json
import hashlib
import gzip
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site-output"
os.environ["USE_LOCAL_MODELS"] = "1"
sys.path.insert(0, str(ROOT))

from app import (MODEL_INFOS, MODEL_REPO_ID, MODEL_REPO_REVISION, app,
                 build_transform, get_or_load_model)
from huggingface_hub import hf_hub_download


def main():
    OUTPUT.mkdir(exist_ok=True)
    models_dir = OUTPUT / "models"
    models_dir.mkdir(exist_ok=True)
    manifest = []
    for info in MODEL_INFOS:
        if not info.path.exists():
            cached = hf_hub_download(
                repo_id=MODEL_REPO_ID, filename=info.model_file,
                revision=MODEL_REPO_REVISION,
            )
            shutil.copyfile(cached, info.path)
        model = get_or_load_model(info.key).cpu().eval()
        destination = models_dir / (info.key + ".onnx")
        torch.onnx.export(
            model, torch.zeros(1, 3, info.img_size, info.img_size), destination,
            input_names=["image"], output_names=["logits"],
            opset_version=17, dynamo=False,
        )
        onnx.checker.check_model(str(destination))
        session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        errors = []
        for image_path in sorted((ROOT / "static/assets/examples").glob("*/*.jpg")):
            with Image.open(image_path) as image:
                tensor = build_transform(info.img_size)(image.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                reference = model(tensor).numpy()
            actual = session.run(None, {"image": tensor.numpy()})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
            assert np.argmax(actual) == np.argmax(reference)
            errors.append(float(np.max(np.abs(actual - reference))))
        print(f"Verified {info.key}: max logit difference {max(errors):.8f}")
        # Only quantize the dense layers; retain the convolution weights and
        # preprocessing, and refuse a build if any reference label changes.
        quantized = destination.with_suffix(".quant.onnx")
        quantize_dynamic(str(destination), str(quantized),
                         weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
        quantized_session = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
        probability_errors = []
        for image_path in sorted((ROOT / "static/assets/examples").glob("*/*.jpg")):
            with Image.open(image_path) as image:
                tensor = build_transform(info.img_size)(image.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                reference = torch.softmax(model(tensor), dim=1).numpy()
            logits = quantized_session.run(None, {"image": tensor.numpy()})[0]
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            actual = exponentials / exponentials.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(actual, reference, rtol=0, atol=.01)
            assert np.argmax(actual) == np.argmax(reference)
            probability_errors.append(float(np.max(np.abs(actual - reference))))
        del quantized_session, session
        os.replace(quantized, destination)
        print(f"Verified compact {info.key}: max probability difference {max(probability_errors):.6f}")
        manifest.append({
            "key": info.key, "file": "models/" + destination.name,
            "img_size": info.img_size, "classes": info.classes,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    shutil.copytree(ROOT / "static", OUTPUT / "static", dirs_exist_ok=True)
    runtime_dir = OUTPUT / "static/vendor/ort"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    distribution = ROOT / "node_modules/onnxruntime-web/dist"
    for name in ["ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]:
        shutil.copyfile(distribution / name, runtime_dir / name)
    compressed_runtime = runtime_dir / "ort-wasm-simd-threaded.wasm.gz"
    compressed_runtime.write_bytes(gzip.compress(
        (runtime_dir / "ort-wasm-simd-threaded.wasm").read_bytes(), mtime=0))
    for info in manifest:
        info["runtime"] = {
            "file": "static/vendor/ort/" + compressed_runtime.name,
            "bytes": compressed_runtime.stat().st_size,
            "sha256": hashlib.sha256(compressed_runtime.read_bytes()).hexdigest(),
        }
    with app.test_request_context("/"):
        html = app.jinja_env.get_template("index.html").render(models=MODEL_INFOS, browser_mode=True)
    # Relative assets work on project Pages URLs and on a local preview.
    html = html.replace('"/static/', '"./static/')
    html = html.replace('<script src="./static/app.js">',
                        '<script src="./static/vendor/ort/ort.wasm.min.js"></script>'
                        '<script src="./static/browser-inference.js"></script>'
                        '<script src="./static/app.js">')
    for name in ["app.js", "browser-inference.js", "styles.css"]:
        version = hashlib.sha256((OUTPUT / "static" / name).read_bytes()).hexdigest()[:12]
        html = html.replace(f'"./static/{name}"', f'"./static/{name}?v={version}"')
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    (OUTPUT / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / ".nojekyll").touch()
    print("Built static demo in site-output/")


if __name__ == "__main__":
    main()
