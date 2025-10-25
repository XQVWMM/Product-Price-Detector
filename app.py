from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil, os, cv2, numpy as np, tempfile, math
from detection_functions import predict_total, akaze, kmeans, reg

app = FastAPI(title="Cashierless Product Price Detector")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0
    return interArea / union

def nms(boxes, scores, iou_thresh=0.3):
    if not boxes:
        return []
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_thresh]
    return keep


def process_single_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return [], 0.0, np.zeros((100, 100, 3), np.uint8)

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    MIN_AREA = 5000
    MIN_EXTENT = 0.3
    MIN_ASPECT = 0.2
    MAX_ASPECT = 5.0
    MIN_KP_ABS = 20
    MIN_KP_DENSITY = 0.0008

    detector = cv2.AKAZE_create()
    candidate_boxes, candidate_scores = [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        bbox_area = w * h
        if bbox_area < MIN_AREA:
            continue
        extent = area / float(bbox_area)
        if extent < MIN_EXTENT:
            continue
        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(orig.shape[1], x + w + pad)
        y2 = min(orig.shape[0], y + h + pad)
        crop = orig[y1:y2, x1:x2]

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kps = detector.detect(gray_crop, None)
        num_kp = len(kps)
        area_crop = max(1, (x2 - x1) * (y2 - y1))
        kp_density = num_kp / float(area_crop)

        if not (num_kp >= MIN_KP_ABS or kp_density >= MIN_KP_DENSITY):
            continue

        score = math.log(area_crop + 1) * (1 + kp_density * 1000)
        candidate_boxes.append((x1, y1, x2 - x1, y2 - y1))
        candidate_scores.append(score)


    keep_idxs = nms(candidate_boxes, candidate_scores, iou_thresh=0.5)
    kept_boxes = [candidate_boxes[i] for i in keep_idxs]
    merged_boxes = []
    for box in kept_boxes:
        merged_flag = False
        for j, m in enumerate(merged_boxes):
            if iou(box, m) > 0.4:  
                x1 = min(box[0], m[0])
                y1 = min(box[1], m[1])
                x2 = max(box[0] + box[2], m[0] + m[2])
                y2 = max(box[1] + box[3], m[1] + m[3])
                merged_boxes[j] = (x1, y1, x2 - x1, y2 - y1)
                merged_flag = True
                break
        if not merged_flag:
            merged_boxes.append(box)

    kept_boxes = merged_boxes
    results = []
    total_price = 0.0

    for (x, y, w, h) in kept_boxes:
        crop = orig[y:y+h, x:x+w]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, crop)

        try:
            pred_results, pred_total = predict_total(tmp_path, akaze, kmeans, reg)
        except Exception as e:
            print(f"[WARN] Predict failed: {e}")
            pred_results, pred_total = [], 0.0

        os.remove(tmp_path)

        if not pred_results:
            continue

        for name, price in pred_results:
            results.append((name, price))
            total_price += price

     
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"${pred_total:,.2f}"
        cv2.putText(image, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"TOTAL: ${total_price:,.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    return results, total_price, image


@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>Cashierless Product Price Detector</title>
        </head>
        <body>
            <h2>Upload Beberapa Gambar Produk Sekaligus</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <p><i>Tip:</i> Tekan <b>Ctrl</b> (atau <b>Shift</b>) saat memilih file untuk upload beberapa gambar sekaligus.</p>
                <input type="submit" value="Prediksi Harga">
            </form>
        </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(files: List[UploadFile] = File(...)):
    all_results = []
    grand_total = 0.0
    all_images = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results, total, result_image = process_single_image(file_path)

        result_file = os.path.join(RESULT_DIR, f"result_{file.filename}")
        cv2.imwrite(result_file, result_image)
        all_images.append(os.path.basename(result_file))

        all_results.append((file.filename, results, total))
        grand_total += total

    html = "<html><head><title>Hasil Multi Upload</title></head><body>"
    html += f"<h2>Hasil Prediksi ({len(files)} gambar)</h2>"

    for idx, (fname, results, total) in enumerate(all_results):
        html += f"<h3>{fname}</h3>"
        html += f'<img src="/results/{all_images[idx]}" style="max-width:600px;"><ul>'
        for name, price in results:
            html += f"<li>{name}: ${price:,.2f}</li>"
        html += f"</ul><b>Total: ${total:,.2f}</b><hr>"

    html += f"<h2>Grand Total: ${grand_total:,.2f}</h2>"
    html += '<br><a href="/">Upload Gambar Lain</a></body></html>'
    return HTMLResponse(content=html)
