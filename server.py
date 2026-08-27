"""
server.py — واجهة ويب لتجربة النظارات، لتشغيل المحرك من متصفح على جهاز آخر
غير الذي يشغّل بايثون (مثلاً لابتوب فيه كاميرا، بينما محرك المعالجة على جهاز
بلا كاميرا).

الفكرة: المتصفح يفتح الكاميرا محليًا عبر getUserMedia (كاميرا حقيقية على جهاز
المستخدم)، يرسل كل فريم كصورة JPEG إلى /process، والسيرفر يشغّل نفس محرك
app.py (TryOn) ويرجع الفريم بعد تركيب النظارة.

هذا غلاف حول app.py فقط — لا يعيد أي منطق قياس أو تركيب، فقط ينقل الفريمات
عبر HTTP بدل V4L/DirectShow المحليين.

    python server.py --asset demo_rect --port 5000
"""
from __future__ import annotations
import argparse
import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify

import config
from app import TryOn, load_asset

app = Flask(__name__)
_sessions: dict[str, TryOn] = {}


def get_tryon(asset_id: str) -> TryOn:
    t = _sessions.get(asset_id)
    if t is None:
        t = TryOn(load_asset(asset_id), static=False)
        _sessions[asset_id] = t
    return t


def decode_data_url(data_url: str) -> np.ndarray | None:
    _, _, b64 = data_url.partition(",")
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/assets")
def list_assets():
    items = []
    for p in sorted(config.CATALOG.glob("*.json")):
        a = load_asset(p.stem)
        items.append({"id": a.id, "name": a.name})
    return jsonify(items)


@app.post("/process")
def process():
    data = request.get_json(force=True)
    asset_id = data.get("asset") or "demo_rect"
    frame = decode_data_url(data["image"])
    if frame is None:
        return jsonify({"error": "صورة غير صالحة"}), 400

    try:
        t = get_tryon(asset_id)
    except SystemExit as e:
        return jsonify({"error": str(e)}), 400

    out, rep = t.process(frame, debug=False, smooth=True, with_fit=True)
    fit = rep.pop("fit", None)

    resp = {"image": encode_jpeg(out)}
    if "error" in rep:
        resp["status"] = "لا يوجد وجه في الإطار"
    elif "blocked" in rep:
        resp["status"] = " | ".join(rep["blocked"])
    elif fit is not None:
        resp["status"] = fit.verdict
        resp["fit"] = fit.as_text()
    return jsonify(resp)


INDEX_HTML = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>تجربة النظارات</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; min-height: 100vh; background: #0b0d12; color: #e8eaf0;
    font-family: -apple-system, "Segoe UI", Tahoma, Arial, sans-serif;
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    padding: 16px; box-sizing: border-box;
  }
  h1 { font-size: 18px; margin: 4px 0; font-weight: 600; }
  select, button {
    font-size: 15px; padding: 8px 14px; border-radius: 8px;
    border: 1px solid #333a47; background: #171a21; color: #e8eaf0;
  }
  button { cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .stage {
    position: relative; width: 100%; max-width: 640px; aspect-ratio: 4/3;
    background: #14161c; border-radius: 12px; overflow: hidden;
    border: 1px solid #262a34;
  }
  video, img#out {
    position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
  }
  video { transform: scaleX(-1); }
  #status {
    min-height: 20px; font-size: 14px; color: #ffb454; text-align: center;
    max-width: 640px;
  }
  #fit {
    white-space: pre-wrap; font-size: 13px; color: #9fe6a0;
    background: #12151b; border: 1px solid #24352a; border-radius: 8px;
    padding: 10px 14px; max-width: 640px; width: 100%; box-sizing: border-box;
    display: none;
  }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }
  .hint { font-size: 12px; color: #7d8494; }
</style>
</head>
<body>
  <h1>👓 تجربة النظارات الافتراضية — offline</h1>
  <div class="row">
    <select id="asset"></select>
    <button id="start">تشغيل الكاميرا</button>
    <button id="stop" disabled>إيقاف</button>
  </div>
  <div class="stage">
    <video id="video" autoplay playsinline muted></video>
    <img id="out" style="display:none">
  </div>
  <div id="status"></div>
  <pre id="fit"></pre>
  <div class="hint">المعالجة تتم على السيرفر؛ الكاميرا تُفتح محليًا في متصفحك فقط.</div>

<script>
const video = document.getElementById('video');
const out = document.getElementById('out');
const statusEl = document.getElementById('status');
const fitEl = document.getElementById('fit');
const assetSel = document.getElementById('asset');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');

let stream = null, running = false;
const canvas = document.createElement('canvas');

fetch('/assets').then(r => r.json()).then(items => {
  assetSel.innerHTML = items.map(a => `<option value="${a.id}">${a.name} (${a.id})</option>`).join('');
});

async function loop() {
  if (!running) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
  try {
    const r = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: dataUrl, asset: assetSel.value})
    });
    const j = await r.json();
    if (j.image) { out.src = j.image; out.style.display = 'block'; }
    statusEl.textContent = j.status || '';
    if (j.fit) { fitEl.textContent = j.fit; fitEl.style.display = 'block'; }
    else { fitEl.style.display = 'none'; }
  } catch (e) {
    statusEl.textContent = 'خطأ اتصال بالسيرفر: ' + e;
  }
  if (running) requestAnimationFrame(() => setTimeout(loop, 30));
}

startBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
  } catch (e) {
    statusEl.textContent = 'تعذّر فتح الكاميرا: ' + e;
    return;
  }
  video.srcObject = stream;
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  loop();
};

stopBtn.onclick = () => {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  out.style.display = 'none';
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = '';
  fitEl.style.display = 'none';
};
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ns = ap.parse_args()
    app.run(host="0.0.0.0", port=ns.port, threaded=True)


if __name__ == "__main__":
    main()
