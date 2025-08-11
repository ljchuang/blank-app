import io
import time
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Sequence, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# =============================
# Utility
# =============================

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

# =============================
# Denoise
# =============================

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

# =============================
# Pipeline
# =============================

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

# =============================
# Template matching
# =============================

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

# =============================
# YOLO helpers + auto split (with annotated previews)
# =============================

def rect_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def simple_train_val_split(n_items: int, train_ratio: float, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_train = int(round(train_ratio * n_items))
    split = ["val"] * n_items
    for i in idx[:n_train]:
        split[i] = "train"
    return split


def _yaml_names_block(class_names: List[str]) -> str:
    lines = ["names:"]
    for n in class_names:
        lines.append(f"- {n}")
    return "\n".join(lines)


def build_yolo_zip_auto(
    image_entries: List[Tuple[Image.Image, str, List[tuple], List[int], Optional[Image.Image]]],
    class_names: List[str],
    per_entry_split: Sequence[str],  # "train" or "val"
    img_format: str = "JPEG",
    quality: int = 95,
) -> bytes:
    """
    image_entries: list of tuples -> (export_pil, original_name, det_rects, det_tids, annotated_preview_pil)
    Writes images, labels, dataset.yaml, and annotated previews into the ZIP.
    Previews are saved under dataset/previews/<split>/.
    """
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        for (pil_img, orig_name, det_rects, det_tids, ann_pil), sp in zip(image_entries, per_entry_split):
            stem = Path(orig_name).stem or f"img_{int(time.time()*1000)}"

            # --- image bytes ---
            img_bytes = io.BytesIO()
            if img_format.upper() == "JPEG":
                pil_img.save(img_bytes, format="JPEG", quality=quality)
                img_ext = ".jpg"
            else:
                pil_img.save(img_bytes, format="PNG")
                img_ext = ".png"
            img_bytes.seek(0)

            img_zip_path = f"dataset/images/{sp}/{stem}{img_ext}"
            z.writestr(img_zip_path, img_bytes.read())

            # --- label txt ---
            W, H = pil_img.size
            lines = []
            for rect, tid in zip(det_rects, det_tids):
                x1, y1, x2, y2 = rect
                cx, cy, w, h = rect_to_yolo(x1, y1, x2, y2, W, H)
                cls_id = int(tid)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label_zip_path = f"dataset/labels/{sp}/{stem}.txt"
            z.writestr(label_zip_path, "\n".join(lines))

            # --- annotated preview (optional) ---
            if ann_pil is not None:
                prev_bytes = io.BytesIO()
                # Save previews using same format for simplicity
                if img_format.upper() == "JPEG":
                    ann_pil.save(prev_bytes, format="JPEG", quality=min(95, quality))
                    prev_ext = ".jpg"
                else:
                    ann_pil.save(prev_bytes, format="PNG")
                    prev_ext = ".png"
                prev_bytes.seek(0)
                prev_zip_path = f"dataset/previews/{sp}/{stem}{prev_ext}"
                z.writestr(prev_zip_path, prev_bytes.read())

        # --- dataset.yaml ---
        yaml_lines = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: ",
            _yaml_names_block(class_names),
        ]
        z.writestr("dataset/dataset.yaml", "\n".join(yaml_lines))

    zbuf.seek(0)
    return zbuf.read()

# =============================
# App
# =============================

st.set_page_config(page_title="Image Denoise Lab", page_icon="🧽", layout="wide")
st.title("🧽 Image Denoise Lab — Auto train/val split + Annotated previews in ZIP")

with st.sidebar:
    st.header("Denoise")
    method = st.selectbox("Method", ("Non-Local Means (colored)","Bilateral","Median","Gaussian"))
    if method == "Non-Local Means (colored)":
        h = st.slider("h (luma)", 1.0, 30.0, 10.0, 0.5)
        hColor = st.slider("hColor (chroma)", 1.0, 30.0, 10.0, 0.5)
        template = st.slider("Template window (odd)", 3, 15, 7, 2)
        search = st.slider("Search window (odd)", 9, 35, 21, 2)
        denoise_params = (h, hColor, template, search)
    elif method == "Bilateral":
        d = st.slider("Diameter", 1, 25, 9, 1)
        sigma_color = st.slider("SigmaColor", 1.0, 200.0, 75.0, 1.0)
        sigma_space = st.slider("SigmaSpace", 1.0, 200.0, 75.0, 1.0)
        denoise_params = (d, sigma_color, sigma_space)
    elif method == "Median":
        ksize = st.slider("Kernel (odd)", 3, 21, 5, 2)
        denoise_params = (ksize,)
    else:
        ksize = st.slider("Kernel (odd)", 3, 51, 9, 2)
        sigma = st.slider("Sigma", 0.0, 25.0, 0.0, 0.5)
        denoise_params = (ksize, sigma)

    st.header("Pipeline")
    use_pipeline = st.checkbox("HistEq → Gray → Erode → Threshold → Watershed", value=True)
    er_k = st.slider("Erode kernel (odd)", 3, 21, 3, 2)
    er_iter = st.slider("Erode iters", 1, 10, 1, 1)
    th_mode = st.selectbox("Threshold", ("Otsu","Adaptive-mean","Adaptive-gaussian","Fixed"), index=0)
    th_value = st.slider("Fixed threshold", 0, 255, 128, 1)
    th_block = st.slider("Adaptive block (odd)", 3, 51, 21, 2)
    th_C = st.slider("Adaptive C", -20, 20, 2, 1)
    ws_fg_ratio = st.slider("WS FG ratio", 0.1, 0.9, 0.45, 0.01)
    ws_bg_dilate = st.slider("WS BG dilate", 1, 20, 6, 1)

    st.header("Template Matching")
    enable_tm = st.checkbox("Enable template matching", value=False)
    tm_method_name = st.selectbox("Method", ("TM_CCOEFF_NORMED","TM_CCORR_NORMED","TM_SQDIFF_NORMED"), index=0)
    tm_thresh = st.slider("Match threshold", 0.1, 0.99, 0.8, 0.01)
    tm_max_det = st.slider("Max detections/image", 1, 130, 30, 1)
    tm_iou = st.slider("NMS IoU", 0.0, 1.0, 0.3, 0.05)
    tm_source = st.selectbox("Search on", ("Original","Denoised","Pipeline Binary"), index=1)

# Upload images
uploaded_files = st.file_uploader("Upload images", type=["png","jpg","jpeg","webp","tif","tiff"], accept_multiple_files=True)
if not uploaded_files:
    st.stop()

# Load + process
images_info: List[Dict] = []
for f in uploaded_files:
    pil_in = load_image(f)
    img_bgr_in = pil_to_cv(pil_in)
    if method == "Non-Local Means (colored)":
        img_bgr = denoise_nlm_colored(img_bgr_in, *denoise_params)
    elif method == "Bilateral":
        img_bgr = denoise_bilateral(img_bgr_in, *denoise_params)
    elif method == "Median":
        img_bgr = denoise_median(img_bgr_in, *denoise_params)
    else:
        img_bgr = denoise_gaussian(img_bgr_in, *denoise_params)

    entry = {
        "name": getattr(f, 'name', f"image_{len(images_info)}.jpg"),
        "pil_in": pil_in,
        "img_bgr_in": img_bgr_in,
        "img_bgr": img_bgr,
        "pil_denoised": cv_to_pil(img_bgr),
        "bw": None,
        "ws_overlay": None,
        "dist": None,
    }
    if use_pipeline:
        gray_init = to_gray(img_bgr)
        gray_eq = hist_equalize(gray_init)
        gray_eroded = erode(gray_eq, er_k, er_iter)
        bw = binarize(gray_eroded, th_mode, th_value, th_block, th_C)
        markers, ws_overlay, dist = watershed_segment(img_bgr, bw, ws_fg_ratio, ws_bg_dilate)
        entry.update({"gray_eq": gray_eq, "gray_eroded": gray_eroded, "bw": bw, "ws_overlay": ws_overlay, "dist": dist})
    images_info.append(entry)

# Quick preview
st.subheader("Preview (first 6)")
cols = st.columns(6)
for i, info in enumerate(images_info[:6]):
    with cols[i % 6]:
        st.image(info["pil_in"], caption=info["name"], use_column_width=True)

# Template matching
all_results: Dict[str, Dict] = {}
if enable_tm:
    tmpl_files = st.file_uploader("Upload templates", type=["png","jpg","jpeg","webp","tif","tiff"], key="tmpl", accept_multiple_files=True)
    if tmpl_files:
        method_map = {"TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED, "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED, "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED}
        tm_method = method_map[tm_method_name]

        st.subheader("Templates Preview")
        for f in tmpl_files:
            st.image(load_image(f), width=120)

        palette = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255),(255,0,128),(0,255,128)]

        for info in images_info:
            if tm_source == "Original":
                search_img = info["img_bgr_in"].copy()
                export_pil_img = info["pil_in"]
            elif tm_source == "Denoised":
                search_img = info["img_bgr"].copy()
                export_pil_img = info["pil_denoised"]
            else:
                if info.get("bw") is None:
                    search_img = info["img_bgr"].copy()
                    export_pil_img = info["pil_denoised"]
                else:
                    search_img = cv2.cvtColor(info["bw"], cv2.COLOR_GRAY2BGR)
                    export_pil_img = cv_to_pil(search_img)

            all_rects: List[tuple] = []
            all_scores: List[float] = []
            all_tids: List[int] = []

            for tidx, f in enumerate(tmpl_files):
                tmpl_pil = load_image(f)
                tmpl_bgr = pil_to_cv(tmpl_pil)
                res = run_template_matching(search_img, tmpl_bgr, tm_method)
                h, w = tmpl_bgr.shape[:2]

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

                keep = nms_rects(rects, scores, tm_iou)
                for k in keep:
                    all_rects.append(rects[k])
                    all_scores.append(scores[k])
                    all_tids.append(tidx)

            order = list(range(len(all_rects)))
            order.sort(key=lambda i: all_scores[i], reverse=True)
            order = order[:tm_max_det]

            draw = search_img.copy()
            for i in order:
                x1, y1, x2, y2 = all_rects[i]
                tid = all_tids[i]
                color = palette[tid % len(palette)]
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"T{tid}", (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            all_results[info["name"]] = {
                "rects": [all_rects[i] for i in order],
                "scores": [all_scores[i] for i in order],
                "tids": [all_tids[i] for i in order],
                "annotated_pil": cv_to_pil(draw),
                "export_pil": export_pil_img,
            }

        st.subheader("Detections (annotated previews)")
        cols = st.columns(3)
        for i, (name, res) in enumerate(all_results.items()):
            with cols[i % 3]:
                st.image(res["annotated_pil"], caption=name, use_column_width=True)

        # =============================
        # AUTO TRAIN/VAL SPLIT EXPORT
        # =============================
        with st.expander("📦 Export YOLO dataset with AUTO train/val split"):
            # Class names for templates
            class_names: List[str] = []
            for tidx in range(len(tmpl_files)):
                default_name = f"class_{tidx}"
                cls = st.text_input(f"Class name for template #{tidx}", value=default_name, key=f"clsname_auto_{tidx}")
                class_names.append(cls)

            train_ratio = st.slider("Train ratio", 0.1, 0.95, 0.8, 0.05)
            seed = st.number_input("Random seed", value=42, step=1)
            img_fmt = st.selectbox("Image format", ("JPEG","PNG"), index=0, key="yolo_imgfmt_auto")
            q = 95
            if img_fmt == "JPEG":
                q = st.slider("JPEG quality", 60, 100, 95, 1, key="yolo_jpgq_auto")

            # Prepare entries
            entries: List[Tuple[Image.Image, str, List[tuple], List[int], Optional[Image.Image]]] = []
            names = []
            for info in images_info:
                name = info["name"]
                names.append(name)
                if name not in all_results:
                    export_pil = info["pil_denoised"] if tm_source == "Denoised" else info["pil_in"]
                    entries.append((export_pil, name, [], [], None))
                else:
                    res = all_results[name]
                    entries.append((res["export_pil"], name, res["rects"], res["tids"], res.get("annotated_pil")))

            per_split = simple_train_val_split(len(entries), train_ratio, int(seed))

            if st.button("Build YOLO dataset zip (auto split)", type="primary"):
                try:
                    yolo_zip = build_yolo_zip_auto(
                        image_entries=entries,
                        class_names=class_names,
                        per_entry_split=per_split,
                        img_format=img_fmt,
                        quality=q,
                    )
                    st.download_button(
                        "⬇️ Download dataset_auto_split.zip",
                        data=yolo_zip,
                        file_name="yolo_dataset_auto_split.zip",
                        mime="application/zip",
                    )
                    st.success("YOLO dataset ready with automatic train/val split, plus annotated previews under dataset/previews/. Train via data=dataset/dataset.yaml.")
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")
    else:
        st.info("Upload one or more templates to run matching.")
