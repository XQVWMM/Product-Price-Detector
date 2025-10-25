from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil, os
from detection_functions import predict_total, akaze, kmeans, reg
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Product Price Detector")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
        <head>
            <title>Product Price Detector</title>
        </head>
        <body>
            <h2>Upload Gambar Produk</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Prediksi Harga">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(file_path)
    if image is None:
        return HTMLResponse(content="Gagal membaca gambar.", status_code=400)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.AKAZE_create()
    keypoints = detector.detect(gray, None)

    threshold = 50 
    if len(keypoints) < threshold:
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Gagal Prediksi</title></head>
                <body style="font-family:Arial; text-align:center; margin-top:50px;">
                    <h2>Gambar tidak terdeteksi sebagai produk</h2>
                    <p>Jumlah fitur terdeteksi terlalu sedikit ({len(keypoints)} &lt; {threshold}).</p>
                    <p>Pastikan gambar jelas dan menampilkan produk dengan pencahayaan cukup.</p>
                    <br>
                    <a href="/">Kembali ke halaman upload</a>
                </body>
            </html>
            """,
            status_code=400
        )

    results, total = predict_total(file_path, akaze, kmeans, reg)

    y0 = 30
    for idx, (name, price) in enumerate(results):
        text = f"{name}: ${price:,.2f}"
        cv2.putText(image, text, (10, y0 + idx * 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Total: ${total:,.2f}", (10, y0 + (len(results) + 1) * 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    result_file = os.path.join(RESULT_DIR, f"result_{file.filename}")
    cv2.imwrite(result_file, image)

    html_result = f"""
    <html>
        <head><title>Hasil Prediksi</title></head>
        <body>
            <h2>Hasil Prediksi</h2>
            <img src="/results/{os.path.basename(result_file)}" alt="Result">
            <h3>Detail Harga:</h3>
            <ul>
    """
    for name, price in results:
        html_result += f"<li>{name}: ${price:,.2f}</li>"
    html_result += f"</ul><h3>Total: ${total:,.2f}</h3>"
    html_result += '<br><a href="/">Upload Gambar Lain</a></body></html>'

    return HTMLResponse(content=html_result)
