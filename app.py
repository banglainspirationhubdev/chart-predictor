""" Image-based UP/DOWN predictor — MVP

A FastAPI service that accepts a chart/screenshot image and returns a naive UP/DOWN probability estimate. This is a starter you can run today and then swap in a better ML model later.

⚠️ Limitations (read this):

Works best on candlestick/bar charts with green (up) and red (down) candles.

Theme-dependent. If your platform uses different colors, update COLOR_RANGES.

This MVP uses simple computer-vision heuristics on the rightmost candles.

For real accuracy, replace score_with_cv_heuristic with a trained model.


How to run

1. Python 3.10+


2. pip install -r requirements.txt


3. uvicorn app:app --reload --port 8000


4. POST an image to http://127.0.0.1:8000/predict (or open http://127.0.0.1:8000 in your browser for a simple upload form)



requirements.txt (install these): fastapi==0.111.0 uvicorn[standard]==0.30.1 pillow==10.4.0 opencv-python==4.10.0.84 numpy==1.26.4 scikit-learn==1.4.2 python-multipart==0.0.9 """

from fastapi import FastAPI, File, UploadFile, Form from fastapi.responses import JSONResponse, HTMLResponse from fastapi.middleware.cors import CORSMiddleware from PIL import Image import numpy as np import cv2 from io import BytesIO from typing import Tuple, Dict

app = FastAPI(title="Image-based UP/DOWN Predictor (MVP)") app.add_middleware( CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"], )

======== Tunable parameters ========

HSV ranges for green/red candles (typical trading UIs). Adjust to your theme.

COLOR_RANGES = { "green": [ ((40, 40, 40), (90, 255, 255)),     # bright green range ((30, 30, 30), (70, 255, 255)),     # alt green ], "red": [ ((0, 40, 40), (10, 255, 255)),      # low-hue reds ((170, 40, 40), (180, 255, 255)),   # wrap-around reds ], }

Portion of the image (from right edge) to analyze as the "recent candles" region

RIGHT_STRIP_RATIO = 0.28  # 28% of width TOP_CROP_RATIO = 0.06     # crop UI headers BOTTOM_CROP_RATIO = 0.06  # crop footers

Smoothing kernel for noise reduction

BLUR_KSIZE = (3, 3)

def load_image_to_bgr(data: bytes) -> np.ndarray: img = Image.open(BytesIO(data)).convert("RGB") arr = np.array(img) bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) return bgr

def crop_recent_region(bgr: np.ndarray) -> np.ndarray: h, w = bgr.shape[:2] x0 = int(w * (1.0 - RIGHT_STRIP_RATIO)) y0 = int(h * TOP_CROP_RATIO) y1 = int(h * (1.0 - BOTTOM_CROP_RATIO)) roi = bgr[y0:y1, x0:w] return roi

def mask_color_hsv(bgr: np.ndarray, ranges: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray: hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) (lo, hi) = ranges lo = np.array(lo, dtype=np.uint8) hi = np.array(hi, dtype=np.uint8) mask = cv2.inRange(hsv, lo, hi) return mask

def count_colored_pixels(bgr: np.ndarray) -> Dict[str, int]: bgr = cv2.GaussianBlur(bgr, BLUR_KSIZE, 0) counts = {"green": 0, "red": 0} for color, ranges in COLOR_RANGES.items(): total = 0 for r in ranges: total += int(np.sum(mask_color_hsv(bgr, r) > 0)) counts[color] = total return counts

def edge_slope_score(bgr: np.ndarray) -> float: """Estimate general slope in the recent region using edges + Hough lines. Returns positive for upward bias, negative for downward. """ gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) gray = cv2.GaussianBlur(gray, (3, 3), 0) edges = cv2.Canny(gray, 50, 150) lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=20, maxLineGap=8) if lines is None: return 0.0 slopes = [] for x1, y1, x2, y2 in lines[:, 0, :]: dx = x2 - x1 dy = y2 - y1 if abs(dx) < 2: continue slope = -dy / dx  # screen y grows downward # Keep only moderately horizontal lines (price action), filter out vertical gridlines if 0.1 < abs(slope) < 3.0: slopes.append(slope) if not slopes: return 0.0 return float(np.median(slopes))

def score_with_cv_heuristic(bgr: np.ndarray) -> Dict[str, float]: """ Combine (1) green-vs-red pixel ratio and (2) slope estimate to produce a simple probability-like score for UP/DOWN. """ roi = crop_recent_region(bgr) counts = count_colored_pixels(roi) green = counts["green"] red = counts["red"] total = max(green + red, 1) color_bias = (green - red) / total  # -1..+1

slope_bias = edge_slope_score(roi)  # roughly -ve -> down, +ve -> up
# Normalize slope into -1..+1 using tanh-like squashing
slope_bias = float(np.tanh(slope_bias))

# Weighted blend — tune as you test on your screenshots
up_score = 0.65 * max(0.0, (color_bias + 1) / 2) + 0.35 * max(0.0, (slope_bias + 1) / 2)
down_score = 1.0 - up_score

# Calibrate to [0.05, 0.95] to avoid overconfidence
up_prob = 0.05 + 0.90 * up_score
down_prob = 0.05 + 0.90 * down_score

# Normalize again just in case
s = up_prob + down_prob
up_prob /= s
down_prob /= s

return {
    "up": round(float(up_prob), 4),
    "down": round(float(down_prob), 4),
    "debug": {
        "green_pixels": int(green),
        "red_pixels": int(red),
        "color_bias": round(float(color_bias), 4),
        "slope_bias": round(float(slope_bias), 4),
    },
}

@app.get("/") def index(): return HTMLResponse( """ <html> <head><title>UP/DOWN Predictor (MVP)</title></head> <body style="font-family: system-ui; max-width: 720px; margin: 32px auto;"> <h2>Image-based UP/DOWN Predictor (MVP)</h2> <p>Upload a chart screenshot (PNG/JPG). The server will analyze the rightmost ~30% and guess UP/DOWN.</p> <form action="/predict" method="post" enctype="multipart/form-data"> <input type="file" name="file" accept="image/*" required> <br><br> <label>Right strip ratio (0.15 - 0.5):</label> <input type="number" step="0.01" name="right_strip" value="0.28"> <br><br> <button type="submit">Predict</button> </form> <p style="margin-top:16px; font-size: 14px; color: #666;">Tip: If your theme uses different candle colors, update COLOR_RANGES in the code.</p> </body> </html> """ )

@app.post("/predict") async def predict(file: UploadFile = File(...), right_strip: float = Form(0.28)): try: global RIGHT_STRIP_RATIO RIGHT_STRIP_RATIO = float(np.clip(right_strip, 0.15, 0.5))

data = await file.read()
    bgr = load_image_to_bgr(data)
    result = score_with_cv_heuristic(bgr)

    # Select label
    label = "UP" if result["up"] >= result["down"] else "DOWN"
    response = {
        "label": label,
        "probabilities": result,
        "notes": "Heuristic MVP. For production, replace with a trained classifier.",
    }
    return JSONResponse(response)
except Exception as e:
    return JSONResponse({"error": str(e)}, status_code=400)

======== Optional: stub for a real ML model ========

You can later train a CNN/ViT and load it here.

class DummyModel: def predict_proba(self, features: np.ndarray) -> Tuple[float, float]: # placeholder return (0.5, 0.5)

ml_model = DummyModel()

def extract_features_from_image(bgr: np.ndarray) -> np.ndarray: # TODO: Implement robust features (e.g., TA from reconstructed line, or CNN embeddings) roi = crop_recent_region(bgr) gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) gray = cv2.resize(gray, (96, 96)) gray = gray.astype(np.float32) / 255.0 return gray.flatten()[None, :]

def predict_with_ml(bgr: np.ndarray) -> Dict[str, float]: X = extract_features_from_image(bgr) up, down = ml_model.predict_proba(X) s = up + down up /= s down /= s return {"up": float(up), "down": float(down)}

if name == "main": import uvicorn uvicorn.run(app, host="0.0.0.0", port=8000)

