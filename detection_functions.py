import cv2
import joblib
import numpy as np

akaze = cv2.AKAZE_create()
kmeans = joblib.load("dataset/models/kmeans_bovw.joblib")
reg = joblib.load("dataset/models/regressor_rf.joblib")

def build_histogram(descriptors, kmeans_model=kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans_model.n_clusters)
    words = kmeans_model.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(kmeans_model.n_clusters+1))
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def detect_products(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crops = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 5000:
            crop = image[y:y+h, x:x+w]
            crops.append(crop)
    return crops

def predict_price_from_crop(crop, akaze=akaze, kmeans=kmeans, model=reg):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    kps, des = akaze.detectAndCompute(gray, None)
    if des is None or len(des) == 0:
        return None
    hist = build_histogram(des, kmeans)
    return model.predict([hist])[0]

def predict_total(image_path, akaze=akaze, kmeans=kmeans, model=reg):
    import cv2
    image = cv2.imread(image_path)
    crops = detect_products(image)
    total = 0
    results = []
    for idx, crop in enumerate(crops):
        price = predict_price_from_crop(crop, akaze, kmeans, model)
        if price is not None:
            results.append((f"Produk_{idx+1}", price))
            total += price
    return results, total
