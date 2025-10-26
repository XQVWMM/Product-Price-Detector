from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil, os, cv2, numpy as np, tempfile
from detection_functions import predict_total, akaze, kmeans, reg

app = FastAPI(title="Cashierless Product Price Detector")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
template_files = [f for f in os.listdir(TEMPLATE_FOLDER) if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not template_files:
    raise FileNotFoundError("Tidak ada file gambar di folder template")
TEMPLATE_PATH = os.path.join(TEMPLATE_FOLDER, template_files[0])
akaze_template = cv2.AKAZE_create()

def load_template(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

tar_gray, tar_img = load_template(TEMPLATE_PATH)
tar_key, tar_desc = akaze_template.detectAndCompute(tar_gray, None)
tar_desc = np.float32(tar_desc)

def resize_with_aspect_ratio(image, target_size=(800, 800)):
    if image is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = max(1,int(w*scale)), max(1,int(h*scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    top = (target_size[1] - new_h) // 2
    left = (target_size[0] - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB-xA)
    interH = max(0, yB-yA)
    interArea = interW * interH
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    union = boxAArea + boxBArea - interArea
    if union == 0: return 0
    return interArea / union

def nms(boxes, scores, iou_thresh=0.4):
    if not boxes: return []
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_thresh]
    return keep

def is_product_template(crop_img, threshold=5):
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.equalizeHist(gray_crop)
    gray_crop = cv2.GaussianBlur(gray_crop, (3,3), 0)
    kp, desc = akaze_template.detectAndCompute(gray_crop, None)
    if desc is None or len(kp) < 5:
        return False
    desc = np.float32(desc)
    FLANN = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))
    matches = FLANN.knnMatch(tar_desc, desc, k=2)
    good_matches = 0
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches += 1
    return good_matches >= threshold

def process_single_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return [], 0.0, np.zeros((100,100,3), np.uint8)
    image = resize_with_aspect_ratio(image)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA=2000; MIN_EXTENT=0.2; MIN_ASPECT=0.2; MAX_ASPECT=5.0
    detector = cv2.AKAZE_create()
    candidate_boxes, candidate_scores = [], []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area <= 0: continue
        bbox_area = w*h
        if bbox_area < MIN_AREA: continue
        extent = area / float(bbox_area)
        if extent < MIN_EXTENT: continue
        aspect = w/float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT: continue
        pad = 6
        x1 = max(0,x-pad); y1 = max(0,y-pad)
        x2 = min(orig.shape[1], x+w+pad); y2 = min(orig.shape[0], y+h+pad)
        crop = orig[y1:y2, x1:x2]
        kps = detector.detect(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
        num_kp = len(kps)
        area_crop = max(1,(x2-x1)*(y2-y1))
        kp_density = num_kp / float(area_crop)
        color_std = np.std(crop.reshape(-1,3), axis=0).mean()
        texture_std = np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        contrast = crop.max() - crop.min()
        mean_color = np.mean(crop.reshape(-1,3), axis=0)
        color_diff = np.max(mean_color) - np.min(mean_color)
        score = min(num_kp/50,1.0) + min(kp_density/0.002,1.0) + min(color_std/40,1.0) + min(texture_std/30,1.0) + min(contrast/50,1.0) + min(color_diff/40,1.0)
        if score < 2.5: continue
        if not is_product_template(crop): continue
        candidate_boxes.append((x1,y1,x2-x1,y2-y1))
        candidate_scores.append(score)
    keep_idxs = nms(candidate_boxes, candidate_scores, iou_thresh=0.4)
    kept_boxes = [candidate_boxes[i] for i in keep_idxs]
    merged_boxes = []
    for box in kept_boxes:
        merged_flag = False
        for j,m in enumerate(merged_boxes):
            if iou(box,m) > 0.4:
                x1 = min(box[0], m[0]); y1 = min(box[1], m[1])
                x2 = max(box[0]+box[2], m[0]+m[2]); y2 = max(box[1]+box[3], m[1]+m[3])
                merged_boxes[j] = (x1,y1,x2-x1,y2-y1)
                merged_flag = True
                break
        if not merged_flag:
            merged_boxes.append(box)
    kept_boxes = merged_boxes
    results = []; total_price = 0.0
    for (x,y,w,h) in kept_boxes:
        crop = orig[y:y+h, x:x+w]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, crop)
        try:
            pred_results, pred_total = predict_total(tmp_path, akaze, kmeans, reg)
        except:
            pred_results, pred_total = [],0.0
        os.remove(tmp_path)
        if not pred_results: continue
        for name, price in pred_results:
            results.append((name, price))
            total_price += price
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        label = f"${pred_total:,.2f}"
        cv2.putText(image, label, (x,max(15,y-6)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(image, f"TOTAL: ${total_price:,.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
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
                <p><i>Tip:</i> Tekan <b>Ctrl</b> atau <b>Shift</b> untuk memilih beberapa gambar.</p>
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
