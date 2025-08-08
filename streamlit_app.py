import io
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# -----------------------------
# Utility functions
# -----------------------------

def load_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def download_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()

# -----------------------------
# Denoising methods
# -----------------------------

def denoise_gaussian(img_bgr: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    k = max(3, ksize | 1)
    return cv2.GaussianBlur(img_bgr, (k, k), sigmaX=sigma)


def denoise_median(img_bgr: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, ksize | 1)
    return cv2.medianBlur(img_bgr, k)


def denoise_bilateral(img_bgr: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    d = max(1, int(d))
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def denoise_nlm_colored(img_bgr: np.ndarray, h: float, hColor: float, template: int, search: int) -> np.ndarray:
    template = max(3, template | 1)
    search = max(3, search | 1)
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=h, hColor=hColor, templateWindowSize=template, searchWindowSize=search)

# -----------------------------
# Pipeline steps
# -----------------------------

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def hist_equalize(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def erode(gray: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
    k = max(3, ksize | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(gray, kernel, iterations=max(1, iterations))


def binarize(gray: np.ndarray, mode: str, thresh: int, block_size: int, C: int) -> np.ndarray:
    if mode == "Otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "Adaptive-mean":
        bs = max(3, block_size | 1)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, C)
    elif mode == "Adaptive-gaussian":
        bs = max(3, block_size | 1)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return bw


def watershed_segment(img_bgr: np.ndarray, bw: np.ndarray, fg_ratio: float, bg_dilate: int):
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(bw, kernel, iterations=max(1, bg_dilate))
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)
    overlay = img_bgr.copy()
    overlay[markers == -1] = [0, 0, 255]
    return markers, overlay, dist

# -----------------------------
# Template matching helpers
# -----------------------------

def nms_rects(rects: List[tuple], scores: List[float], iou_thr: float) -> List[int]:
    if not rects:
        return []
    idxs = list(range(len(rects)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(rects[i], rects[j]) < iou_thr]
    return keep


def run_template_matching(src_img: np.ndarray, tmpl_img: np.ndarray, method: int) -> np.ndarray:
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if src_img.ndim == 3 else src_img
    tmp_gray = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY) if tmpl_img.ndim == 3 else tmpl_img
    res = cv2.matchTemplate(src_gray, tmp_gray, method)
    res = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return res

# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="Image Denoise Lab", page_icon="🧽", layout="wide")
st.title("🧽 Image Denoise Lab")
st.caption("Upload a photo, pick a method, tune sliders, and download the cleaned result. Includes a processing pipeline: Histogram Equalization → Grayscale → Erosion → Threshold → Watershed. Now with Template Matching.")

with st.sidebar:
    st.header("Denoise Controls")
    method = st.selectbox("Denoise method", ("Non-Local Means (colored)","Bilateral","Median","Gaussian"))
    if method == "Non-Local Means (colored)":
        h = st.slider("h (luminance strength)", 1.0, 30.0, 10.0, 0.5)
        hColor = st.slider("hColor (chrominance strength)", 1.0, 30.0, 10.0, 0.5)
        template = st.slider("Template window size (odd)", 3, 15, 7, 2)
        search = st.slider("Search window size (odd)", 9, 35, 21, 2)
        denoise_params = (h, hColor, template, search)
    elif method == "Bilateral":
        d = st.slider("Neighborhood diameter (px)", 1, 25, 9, 1)
        sigma_color = st.slider("SigmaColor", 1.0, 200.0, 75.0, 1.0)
        sigma_space = st.slider("SigmaSpace", 1.0, 200.0, 75.0, 1.0)
        denoise_params = (d, sigma_color, sigma_space)
    elif method == "Median":
        ksize = st.slider("Kernel size (odd)", 3, 21, 5, 2)
        denoise_params = (ksize,)
    else:
        ksize = st.slider("Kernel size (odd)", 3, 51, 9, 2)
        sigma = st.slider("Sigma", 0.0, 25.0, 0.0, 0.5)
        denoise_params = (ksize, sigma)

    st.header("Pipeline Controls")
    use_pipeline = st.checkbox("Run Histogram Equalization → Grayscale → Erosion → Threshold → Watershed", value=True)
    er_k = st.slider("Erosion kernel (odd)", 3, 21, 3, 2)
    er_iter = st.slider("Erosion iterations", 1, 10, 1, 1)
    th_mode = st.selectbox("Threshold mode", ("Otsu","Adaptive-mean","Adaptive-gaussian","Fixed"), index=0)
    th_value = st.slider("Fixed threshold (when Fixed)", 0, 255, 128, 1)
    th_block = st.slider("Adaptive block size (odd)", 3, 51, 21, 2)
    th_C = st.slider("Adaptive C (bias)", -20, 20, 2, 1)
    ws_fg_ratio = st.slider("Watershed FG threshold ratio", 0.1, 0.9, 0.45, 0.01)
    ws_bg_dilate = st.slider("Watershed background dilate iters", 1, 20, 6, 1)

    st.header("Template Matching")
    enable_tm = st.checkbox("Enable template matching", value=False)
    tm_method_name = st.selectbox(
        "Method",
        ("TM_CCOEFF_NORMED","TM_CCORR_NORMED","TM_SQDIFF_NORMED"),
        index=0,
    )
    tm_thresh = st.slider("Match threshold", 0.1, 0.99, 0.8, 0.01)
    tm_max_det = st.slider("Max detections", 1, 130, 30, 1)
    tm_iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.3, 0.05)
    tm_source = st.selectbox("Search on", ("Original","Denoised","Pipeline Binary"), index=1)

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","tif","tiff"])
if uploaded is None:
    st.stop()

pil_in = load_image(uploaded)
img_bgr_in = pil_to_cv(pil_in)
if method == "Non-Local Means (colored)":
    img_bgr = denoise_nlm_colored(img_bgr_in, *denoise_params)
elif method == "Bilateral":
    img_bgr = denoise_bilateral(img_bgr_in, *denoise_params)
elif method == "Median":
    img_bgr = denoise_median(img_bgr_in, *denoise_params)
else:
    img_bgr = denoise_gaussian(img_bgr_in, *denoise_params)

pil_denoised = cv_to_pil(img_bgr)
left, right = st.columns(2)
with left: st.subheader("Original"); st.image(pil_in)
with right: st.subheader("Denoised"); st.image(pil_denoised)

# Pipeline
bw = None
if use_pipeline:
    gray_init = to_gray(img_bgr)
    gray_eq = hist_equalize(gray_init)
    gray = gray_eq
    gray_eroded = erode(gray, er_k, er_iter)
    bw = binarize(gray_eroded, th_mode, th_value, th_block, th_C)
    markers, ws_overlay, dist = watershed_segment(img_bgr, bw, ws_fg_ratio, ws_bg_dilate)

    c1, c2, c3 = st.columns(3)
    with c1: st.subheader("Hist Equalized"); st.image(gray_eq, clamp=True)
    with c2: st.subheader("Erosion"); st.image(gray_eroded, clamp=True)
    with c3: st.subheader("Binary"); st.image(bw, clamp=True)
    c4, c5 = st.columns(2)
    with c4:
        st.subheader("Distance Transform")
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st.image(dist_vis, clamp=True)
    with c5: st.subheader("Watershed Overlay"); st.image(cv_to_pil(ws_overlay))

# -----------------------------
# Template Matching UI + Logic
# -----------------------------
if enable_tm:
    tmpl_files = st.file_uploader(
        "Upload template image(s)",
        type=["png","jpg","jpeg","webp","tif","tiff"],
        key="tmpl",
        accept_multiple_files=True,
    )
    if tmpl_files:
        # Choose search image
        if tm_source == "Original":
            search_img = img_bgr_in.copy()
        elif tm_source == "Denoised":
            search_img = img_bgr.copy()
        else:
            if bw is None:
                st.warning("Pipeline Binary not available yet; run pipeline above.")
                search_img = img_bgr.copy()
            else:
                search_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        method_map = {
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        }
        tm_method = method_map[tm_method_name]

        st.subheader("Templates Preview")
        for f in tmpl_files:
            st.image(load_image(f), width=120)

        # Collect detections across all templates
        all_rects: List[tuple] = []
        all_scores: List[float] = []
        all_tids: List[int] = []

        heatmaps: List[np.ndarray] = []
        sizes: List[tuple] = []
        for tidx, f in enumerate(tmpl_files):
            tmpl_pil = load_image(f)
            tmpl_bgr = pil_to_cv(tmpl_pil)
            res = run_template_matching(search_img, tmpl_bgr, tm_method)
            heatmaps.append(res)
            h, w = tmpl_bgr.shape[:2]
            sizes.append((w, h))

            rects, scores = [], []
            if tm_method == cv2.TM_SQDIFF_NORMED:
                loc = np.where(res <= (1.0 - tm_thresh))
                for pt in zip(*loc[::-1]):
                    x1, y1, x2, y2 = pt[0], pt[1], pt[0] + w, pt[1] + h
                    rects.append((x1, y1, x2, y2))
                    scores.append(1.0 - float(res[pt[1], pt[0]]))
            else:
                loc = np.where(res >= tm_thresh)
                for pt in zip(*loc[::-1]):
                    x1, y1, x2, y2 = pt[0], pt[1], pt[0] + w, pt[1] + h
                    rects.append((x1, y1, x2, y2))
                    scores.append(float(res[pt[1], pt[0]]))

            # Per-template NMS then collect
            keep = nms_rects(rects, scores, tm_iou)
            for k in keep:
                all_rects.append(rects[k])
                all_scores.append(scores[k])
                all_tids.append(tidx)

        # Global top-K cap (max 130)
        order = list(range(len(all_rects)))
        order.sort(key=lambda i: all_scores[i], reverse=True)
        order = order[:tm_max_det]

        # Draw with per-template colors
        palette = [
            (0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
            (128,255,0),(255,128,0),(0,128,255),(128,0,255),(255,0,128),(0,255,128)
        ]
        draw = search_img.copy()
        for i in order:
            x1, y1, x2, y2 = all_rects[i]
            tid = all_tids[i]
            color = palette[tid % len(palette)]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f"T{tid}", (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        st.subheader("Detections (multi-templates)")
        st.image(cv_to_pil(draw), use_column_width=True)

        # Optional: show heatmaps for first few templates
        with st.expander("Match Heatmaps (normalized)"):
            for tidx, res in enumerate(heatmaps):
                st.markdown(f"**Template #{tidx}**")
                st.image(res, clamp=True, use_column_width=True)

        fmt = st.selectbox("Download annotated image format", ("PNG","JPEG"), index=0, key="dl_tm")
        dl_bytes = download_bytes(cv_to_pil(draw), fmt=fmt)
        st.download_button("⬇️ Download annotated", data=dl_bytes, file_name=f"template_matches.{fmt.lower()}", mime=f"image/{fmt.lower()}")
    else:
        st.info("Upload one or more templates to run matching.")
