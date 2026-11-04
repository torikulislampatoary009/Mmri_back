import io, os, time, uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db
from segmenter import Segmenter

BASE = Path(__file__).parent
STORE = BASE / "storage"
for sub in ("images", "masks", "overlays"):
    (STORE / sub).mkdir(parents=True, exist_ok=True)

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
segmenter = Segmenter(BASE / os.getenv("MODEL_PATH", "models/unet_best.pt"))


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return resp


def err(msg, code=400):
    return jsonify({"error": msg}), code


def file_urls(row):
    return {
        "image_url": f"/files/images/{row['image_file']}",
        "mask_url": f"/files/masks/{row['mask_file']}",
        "overlay_url": f"/files/overlays/{row['overlay_file']}",
    }


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "demo_mode": segmenter.demo, "model": segmenter.info})


@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if f is None or not f.filename:
        return err("Attach an image in the 'image' field.")
    if Path(f.filename).suffix.lower() not in ALLOWED:
        return err("Unsupported file type. Use PNG, JPG, TIF or BMP.")
    try:
        thr = min(max(float(request.form.get("threshold", 0.5)), 0.05), 0.95)
    except ValueError:
        return err("threshold must be a number between 0.05 and 0.95.")
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        return err("That file could not be read as an image.")

    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    t0 = time.time()
    prob = segmenter.predict(rgb)
    ms = int((time.time() - t0) * 1000)
    mask = (prob >= thr).astype(np.uint8)

    # binary mask, red RGBA layer (for the viewer), and a flattened overlay for download
    layer = np.zeros((h, w, 4), np.uint8)
    layer[mask == 1] = (255, 77, 94, 255)
    over = rgb.copy()
    over[mask == 1] = (0.55 * over[mask == 1] + 0.45 * np.array([255, 77, 94])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 255), 1)

    uid = uuid.uuid4().hex[:12]
    names = {"image_file": f"{uid}.png", "mask_file": f"{uid}.png", "overlay_file": f"{uid}.png"}
    img.save(STORE / "images" / names["image_file"])
    Image.fromarray(layer, "RGBA").save(STORE / "masks" / names["mask_file"])
    Image.fromarray(over).save(STORE / "overlays" / names["overlay_file"])

    px = int(mask.sum())
    row = {
        "original_name": Path(f.filename).name[:255], **names, "width": w, "height": h,
        "tumor_pixels": px, "tumor_percent": round(100.0 * px / (w * h), 3),
        "threshold_val": thr, "demo_mode": int(segmenter.demo), "inference_ms": ms,
    }
    result = {**row, **file_urls(row), "saved": False}
    try:
        result["id"] = db.insert_prediction(row)
        result["saved"] = True
    except Exception as e:  # prediction still works without the database
        result["db_error"] = str(e)
    return jsonify(result)


@app.get("/api/history")
def history():
    try:
        rows = db.list_predictions(int(request.args.get("limit", 50)))
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    return jsonify([{**r, **file_urls(r)} for r in rows])


@app.get("/api/history/<int:pid>")
def history_one(pid):
    try:
        r = db.get_prediction(pid)
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    return jsonify({**r, **file_urls(r)}) if r else err("Not found.", 404)


@app.delete("/api/history/<int:pid>")
def history_delete(pid):
    try:
        r = db.get_prediction(pid)
        if not r:
            return err("Not found.", 404)
        db.delete_prediction(pid)
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    for sub, key in (("images", "image_file"), ("masks", "mask_file"), ("overlays", "overlay_file")):
        (STORE / sub / r[key]).unlink(missing_ok=True)
    return jsonify({"deleted": pid})


@app.get("/files/<path:p>")
def files(p):
    return send_from_directory(STORE, p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
