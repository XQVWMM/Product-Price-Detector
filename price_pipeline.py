import joblib, cv2, numpy as np
from detection_functions import detect_products, predict_price_from_crop

akaze = cv2.AKAZE_create()
reg = joblib.load("dataset/models/regressor_rf.joblib")
kmeans = joblib.load("dataset/models/kmeans_bovw.joblib")

def predict_prices(image_path):
    image = cv2.imread(image_path)
    crops = detect_products(image)
    results, total = [], 0
    for idx, crop in enumerate(crops):
        price = predict_price_from_crop(crop, akaze, kmeans, reg)
        if price:
            results.append({"name": f"Product_{idx+1}", "price": float(price)})
            total += price
    return {"results": results, "total": float(total)}
