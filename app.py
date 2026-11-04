import base64, io, os, time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db
from segmenter import Segmenter

BASE = Path(__file__).parent
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_SIDE = 1024            # larger scans are shrunk: keeps memory low on a 512 MB free instance
Image.MAX_IMAGE_PIXELS = 25_000_000
MIME = {"image": "image/png", "mask": "image/png", "overlay": "image/png", "thumb": "image/jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
segmenter = Segmenter(BASE / os.getenv("MODEL_PATH", "models/unet_best.pt"))


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.getenv("ALLOWED_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return resp


@app.route("/api/<path:_>", methods=["OPTIONS"])
def preflight(_):
    return ("", 204)


def err(msg, code=400):
    return jsonify({"error": msg}), code


def png(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG", optimize=False)
    return buf.getvalue()


def data_uri(b, mime):
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def urls_for(pid):
    return {f"{k}_url": f"/files/{k}/{pid}" for k in ("image", "mask", "overlay", "thumb")}


@app.get("/api/health")
def health():
    out = {"ok": True, "demo_mode": segmenter.demo, "model": segmenter.info}
    if request.args.get("db"):  # optional: also test the database
        try:
            db.ping(); out["db_ok"] = True
        except Exception as e:
            out["db_ok"] = False; out["db_error"] = str(e)
    return jsonify(out)


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
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    t0 = time.time()
    prob = segmenter.predict(rgb)
    ms = int((time.time() - t0) * 1000)
    mask = (prob >= thr).astype(np.uint8)

    layer = np.zeros((h, w, 4), np.uint8)
    layer[mask == 1] = (255, 77, 94, 255)
    over = rgb.copy()
    over[mask == 1] = (0.55 * over[mask == 1] + 0.45 * np.array([255, 77, 94])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 255), 1)

    thumb = img.copy(); thumb.thumbnail((120, 120))
    tb = io.BytesIO(); thumb.save(tb, "JPEG", quality=80)
    files = {"image": png(rgb), "mask": png(layer, "RGBA"), "overlay": png(over), "thumb": tb.getvalue()}

    px = int(mask.sum())
    meta = {
        "original_name": Path(f.filename).name[:255], "width": w, "height": h, "tumor_pixels": px,
        "tumor_percent": round(100.0 * px / (w * h), 3), "threshold_val": thr,
        "demo_mode": int(segmenter.demo), "inference_ms": ms,
    }
    result = {**meta, "saved": False}
    try:
        pid = db.insert_prediction(meta, files)
        result.update(id=pid, saved=True, **urls_for(pid))
    except Exception as e:  # the segmentation still works if the database is unreachable
        result["db_error"] = str(e)
        for k in ("image", "mask", "overlay", "thumb"):
            result[f"{k}_url"] = data_uri(files[k], MIME[k])
    return jsonify(result)


@app.get("/api/history")
def history():
    try:
        rows = db.list_predictions(min(int(request.args.get("limit", 50)), 200))
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    return jsonify([{**r, **urls_for(r["id"])} for r in rows])


@app.get("/api/history/<int:pid>")
def history_one(pid):
    try:
        r = db.get_prediction(pid)
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    return jsonify({**r, **urls_for(pid)}) if r else err("Not found.", 404)


@app.delete("/api/history/<int:pid>")
def history_delete(pid):
    try:
        if not db.get_prediction(pid):
            return err("Not found.", 404)
        db.delete_prediction(pid)
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    return jsonify({"deleted": pid})


@app.get("/files/<kind>/<int:pid>")
def files(kind, pid):
    if kind not in MIME:
        return err("Not found.", 404)
    try:
        data = db.get_file(pid, kind)
    except Exception as e:
        return err(f"Database unavailable: {e}", 503)
    if data is None:
        return err("Not found.", 404)
    return Response(data, mimetype=MIME[kind], headers={"Cache-Control": "public, max-age=86400"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
