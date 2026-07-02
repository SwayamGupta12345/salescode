

from features import load_image, extract_features
from sklearn.exceptions import InconsistentVersionWarning
import warnings
import joblib
import time
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import pywt
from scipy.fft import fft2, fftshift
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)

WORK_SIZE = 256


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = np.array(img)
    h, w = img.shape[:2]
    scale = WORK_SIZE / min(h, w)
    nh = int(h * scale)
    nw = int(w * scale)
    img = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )
    top = (nh - WORK_SIZE) // 2
    left = (nw - WORK_SIZE) // 2
    img = img[
        top:top + WORK_SIZE,
        left:left + WORK_SIZE
    ]
    return img.astype(np.uint8)


def color_features(img):
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    mn = img.min(axis=(0, 1))
    mx = img.max(axis=(0, 1))
    return np.concatenate([mean, std, mn, mx])


def hsv_features(img):
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )
    mean = hsv.mean(axis=(0, 1))
    std = hsv.std(axis=(0, 1))
    return np.concatenate([mean, std])


def channel_correlation(img):
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    rg = np.corrcoef(r, g)[0, 1]
    rb = np.corrcoef(r, b)[0, 1]
    gb = np.corrcoef(g, b)[0, 1]
    return [rg, rb, gb]


def blue_features(img):
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    return [
        b.mean(),
        b.std(),
        (b / (r + 1)).mean(),
        (b / (g + 1)).mean()
    ]


def sharpness(gray):

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return [

        lap.var(),

        np.mean(np.abs(lap))

    ]


def edge_density(gray):

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    return [

        edges.mean() / 255,

        np.sum(edges > 0)

    ]


def gradient_features(gray):

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1
    )

    mag = np.sqrt(gx ** 2 + gy ** 2)

    return [

        mag.mean(),

        mag.std(),

        np.percentile(mag, 90),

        np.percentile(mag, 99)

    ]


def edge_orientation(gray):

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1
    )

    angle = np.arctan2(
        gy,
        gx
    )

    hist, _ = np.histogram(

        angle,

        bins=12,

        range=(-np.pi, np.pi),

        density=True

    )

    return hist


def noise_features(gray):

    blur = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )

    noise = gray.astype(np.float32) - blur.astype(np.float32)

    return [

        noise.mean(),

        noise.std(),

        np.mean(np.abs(noise))

    ]


def multi_lbp(gray):

    features = []

    settings = [
        (1, 8),
        (2, 16),
        (3, 24)
    ]

    for radius, points in settings:

        lbp = local_binary_pattern(
            gray,
            points,
            radius,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, points + 3),
            density=True
        )

        features.extend(hist)

    return features


def glcm_features(gray):

    gray = (gray // 8).astype(np.uint8)

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    features = []

    props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in props:

        values = graycoprops(glcm, prop)[0]

        features.append(values.mean())
        features.append(values.std())

    return features


def wavelet_features(gray):

    coeffs = pywt.wavedec2(
        gray,
        "haar",
        level=2
    )

    features = []

    for level in coeffs[1:]:

        for band in level:

            band = band.astype(np.float32)

            features.extend([
                band.mean(),
                band.std(),
                np.mean(np.abs(band)),
                np.var(band)
            ])

    return features


def moire_features(gray):

    f = fft2(gray.astype(np.float32))

    f = fftshift(f)

    mag = np.log(np.abs(f) + 1)

    h, w = mag.shape

    cy = h // 2
    cx = w // 2

    y, x = np.indices((h, w))

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    band = (r > 10) & (r < 110)

    values = mag[band]

    median = np.median(values)

    peak = values.max()

    p99 = np.percentile(values, 99)

    threshold = median + 3 * values.std()

    peaks = np.sum(values > threshold)

    outer = mag[(r > 90) & (r < 120)]

    inner = mag[r < 25]

    return [

        peak / (median + 1e-6),

        p99 / (median + 1e-6),

        values.std(),

        outer.mean() / (inner.mean() + 1e-6),

        peaks

    ]


def jpeg_block_features(gray):

    h, w = gray.shape

    vertical = []

    horizontal = []

    for i in range(8, w, 8):

        diff = np.abs(
            gray[:, i].astype(np.float32)
            - gray[:, i - 1].astype(np.float32)
        )

        vertical.append(diff.mean())

    for i in range(8, h, 8):

        diff = np.abs(
            gray[i, :].astype(np.float32)
            - gray[i - 1, :].astype(np.float32)
        )

        horizontal.append(diff.mean())

    return [

        np.mean(vertical),

        np.std(vertical),

        np.mean(horizontal),

        np.std(horizontal)

    ]


def glare_features(gray):

    _, mask = cv2.threshold(
        gray,
        240,
        255,
        cv2.THRESH_BINARY
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    areas = []

    for i in range(1, num_labels):
        areas.append(stats[i, cv2.CC_STAT_AREA])

    if len(areas) == 0:

        return [
            0,
            0,
            0,
            0
        ]

    return [

        len(areas),

        np.mean(areas),

        np.max(areas),

        np.sum(mask > 0) / mask.size

    ]


def local_noise(gray):

    patches = []

    size = 32

    for y in range(0, gray.shape[0], size):

        for x in range(0, gray.shape[1], size):

            patch = gray[y:y + size, x:x + size]

            if patch.shape[0] != size or patch.shape[1] != size:
                continue

            blur = cv2.GaussianBlur(
                patch,
                (5, 5),
                0
            )

            noise = patch.astype(np.float32) - blur.astype(np.float32)

            patches.append(noise.std())

    patches = np.array(patches)

    return [

        patches.mean(),

        patches.std(),

        patches.max(),

        patches.min()

    ]


def extract_features(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    features = []

    features.extend(color_features(img))

    features.extend(hsv_features(img))

    features.extend(channel_correlation(img))

    features.extend(blue_features(img))

    features.extend(sharpness(gray))

    features.extend(edge_density(gray))

    features.extend(gradient_features(gray))

    features.extend(edge_orientation(gray))

    features.extend(noise_features(gray))

    features.extend(local_noise(gray))

    features.extend(glare_features(gray))

    features.extend(multi_lbp(gray))

    features.extend(glcm_features(gray))

    features.extend(wavelet_features(gray))

    features.extend(moire_features(gray))

    features.extend(jpeg_block_features(gray))

    return np.array(features, dtype=np.float32)


# Hide version warning (prediction still works)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "real_fake_lgbm_model.pkl"
)

# Load model only once
model = joblib.load(MODEL_PATH)


def predict(image_path):
    img = load_image(image_path)
    features = extract_features(img).reshape(1, -1)

    probability = model.predict_proba(features)[0, 1]
    prediction = "FAKE" if probability >= 0.5 else "REAL"

    return prediction, float(probability)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    start = time.perf_counter()

    prediction, probability = predict(image_path)

    end = time.perf_counter()

    print(f"Prediction : {prediction}")
    print(f"Probability: {probability:.4f}")
    print(f"Time       : {(end-start)*1000:.2f} ms")
