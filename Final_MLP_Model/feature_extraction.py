import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

# === Feature Functions ===
def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return np.mean(gx**2 + gy**2)

def perceptual_blur_metric(image, threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(dx**2 + dy**2)
    coords = np.column_stack(np.where(edges > 0))
    edge_widths = [1.0 / magnitude[y, x] for (y, x) in coords if magnitude[y, x] > threshold]
    return np.mean(edge_widths) if edge_widths else 0

def average_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def histogram_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return np.mean(hist), np.var(hist)

def extract_exif(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None, None
        exif = {TAGS.get(tag): val for tag, val in exif_data.items() if tag in TAGS}
        iso = exif.get("ISOSpeedRatings", None)
        exposure = exif.get("ExposureTime", None)
        if isinstance(exposure, tuple):
            exposure = exposure[0] / exposure[1]
        return iso, exposure
    except Exception:
        return None, None

# === Main Function ===
def extract_optimized_features(image_folder, limit=2000):
    iso_list = []
    ss_list = []
    total_checked = 0
    total_skipped_unreadable = 0

    all_images = []
    for root, _, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, filename))

    np.random.shuffle(all_images)
    selected_images = all_images[:limit]

    for filepath in selected_images:
        filename = os.path.basename(filepath)
        image = cv2.imread(filepath)
        if image is None:
            total_skipped_unreadable += 1
            print(f"❌ Failed: {filename}")
            continue

        iso, shutter = extract_exif(filepath)
        if iso is None or shutter is None:
            continue

        total_checked += 1

        brightness = average_brightness(image)
        edge = edge_density(image)
        pbm = perceptual_blur_metric(image)
        mean_hist, var_hist = histogram_stats(image)

        # === ISO Feature Set (updated) ===
        iso_features = {
            "Image": os.path.relpath(filepath, image_folder),
            "feat_brightness": brightness,
            "feat_hist_mean": mean_hist,
            "feat_hist_var": var_hist,
            "feat_edge_density": edge,
            "feat_pbm": pbm,
            "ISO": iso
        }
        iso_list.append(iso_features)

        # === Shutter Speed Feature Set (unchanged) ===
        ss_features = {
            "Image": os.path.relpath(filepath, image_folder),
            "feat_laplacian": laplacian_variance(image),
            "feat_tenengrad": tenengrad_score(image),
            "feat_pbm": pbm,
            "feat_edge_density": edge,
            "feat_brightness": brightness,
            "feat_hist_mean": mean_hist,
            "feat_hist_var": var_hist,
            "Shutter": shutter
        }
        ss_list.append(ss_features)

        print(f"✅ [{total_checked}/{limit}] {filename}: Features extracted")

    df_iso = pd.DataFrame(iso_list).dropna()
    df_ss = pd.DataFrame(ss_list).dropna()

    df_iso.to_csv("features_iso_only.csv", index=False)
    df_ss.to_csv("features_shutter_only.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"🖼️  Total images processed: {total_checked}")
    print(f"❌ Skipped unreadable: {total_skipped_unreadable}")
    print(f"📊 ISO features: {len(df_iso)}")
    print(f"📊 Shutter features: {len(df_ss)}")
    print("📂 Saved: features_iso_only.csv & features_shutter_only.csv")

# === Run ===
if __name__ == "__main__":
    extract_optimized_features("/home/ab_students/Desktop/Hackathon/MLP/DL_data/SHARP")
