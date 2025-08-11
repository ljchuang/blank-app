import io
import time
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Sequence
from contextlib import contextmanager

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
    # Try OpenCV's C++ NMS first for speed
    try:
        boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in rects]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_thr))
        if idxs is None or len(idxs) == 0:
            return []
        if isinstance(idxs, np.ndarray):
            keep = idxs.flatten().astype(int).tolist()
        else:
            # OpenCV may return list of lists
            keep = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]
        keep.sort(key=lambda i: scores[i], reverse=True)
        return keep
    except Exception:
        pass

    # Fallback: vectorized NumPy NMS
    boxes = np.asarray(rects, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = sc.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remain = np.where(iou < iou_thr)[0]
        order = order[remain + 1]
    return keep


def run_template_matching(src_img: np.ndarray, tmpl_img: np.ndarray, method: int) -> np.ndarray:
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if src_img.ndim == 3 else src_img
    tmp_gray = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY) if tmpl_img.ndim == 3 else tmpl_img
    res = cv2.matchTemplate(src_gray, tmp_gray, method)
    res = cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return res

# =============================
# YOLO helpers + auto split + annotated export
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


def build_yolo_zip_with_ann(
    image_entries: List[Tuple[Image.Image, str, List[tuple], List[int], Image.Image]],
    class_names: List[str],
    per_entry_split: Sequence[str],
    img_format: str = "JPEG",
    quality: int = 95,
) -> bytes:
    """Create a YOLO dataset zip including annotated previews.
    image_entries: list of (export_pil, filename, rects, tids, annotated_pil)
    """
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        for (pil_img, orig_name, det_rects, det_tids, annotated_pil), sp in zip(image_entries, per_entry_split):
            stem = Path(orig_name).stem or f"img_{int(time.time()*1000)}"

            # Save original image (export target)
            img_bytes = io.BytesIO()
            if img_format.upper() == "JPEG":
                pil_img.save(img_bytes, format="JPEG", quality=quality)
                img_ext = ".jpg"
            else:
                pil_img.save(img_bytes, format="PNG")
                img_ext = ".png"
            img_bytes.seek(0)
            z.writestr(f"dataset/images/{sp}/{stem}{img_ext}", img_bytes.read())

            # Save annotated preview (always JPEG inside zip for compactness)
            ann_bytes = io.BytesIO()
            annotated_pil.save(ann_bytes, format="JPEG", quality=92)
            ann_bytes.seek(0)
            z.writestr(f"dataset/annotated/{sp}/{stem}.jpg", ann_bytes.read())

            # Save YOLO label
            W, H = pil_img.size
            lines = []
            for rect, tid in zip(det_rects, det_tids):
                x1, y1, x2, y2 = rect
                cx, cy, w, h = rect_to_yolo(x1, y1, x2, y2, W, H)
                cls_id = int(tid)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            z.writestr(f"dataset/labels/{sp}/{stem}.txt", "\n".join(lines))

        yaml = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: ",
            f"names: {class_names}",
        ]
        z.writestr("dataset/dataset.yaml", "\n".join(yaml))

    zbuf.seek(0)
    return zbuf.read()

# =============================
# App
# =============================

st.set_page_config(page_title="Image Denoise Lab", page_icon="🧽", layout="wide")
st.title("🧽 Image Denoise Lab — Single Tuning & Batch Export")

# ---- Sidebar controls
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose workflow", ("Single image (tune params)", "Batch export (use current params)"), index=0)

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

    st.header("Performance")
    show_perf = st.checkbox("Show performance metrics", value=True)


# =============================
# Perf helpers
# =============================

@contextmanager
def time_section(section_name: str, perf_stats: List[Tuple[str, float]]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        perf_stats.append((section_name, time.perf_counter() - t0))


def render_perf_stats(title: str, perf_stats: List[Tuple[str, float]]):
    if not perf_stats:
        return
    total = sum(dt for _, dt in perf_stats)
    rows = [
        {"Section": name, "Time (ms)": round(dt * 1000.0, 2), "%": round((dt / total) * 100.0, 1)}
        for name, dt in perf_stats
    ]
    st.markdown(f"**{title}** — total {round(total*1000.0,2)} ms")
    st.table(rows)

# ---- Session helpers for carrying templates & class names from tuning to batch
if "tm_templates" not in st.session_state:
    st.session_state.tm_templates = None  # list of (name, bytes)
if "tm_class_names" not in st.session_state:
    st.session_state.tm_class_names = None  # list[str]
if "tm_settings" not in st.session_state:
    st.session_state.tm_settings = None  # dict of thresholds etc.

# =============================
# SINGLE IMAGE MODE (TUNE)
# =============================
if mode == "Single image (tune params)":
    uploaded = st.file_uploader("Upload ONE image", type=["png","jpg","jpeg","webp","tif","tiff"], accept_multiple_files=False)
    if uploaded is None:
        st.stop()

    single_perf: List[Tuple[str, float]] = []

    with time_section("load_image", single_perf):
        pil_in = load_image(uploaded)
    with time_section("pil_to_cv", single_perf):
        img_bgr_in = pil_to_cv(pil_in)
    with time_section(f"denoise:{method}", single_perf):
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

    bw = None
    if use_pipeline:
        with time_section("pipeline:to_gray", single_perf):
            gray_init = to_gray(img_bgr)
        with time_section("pipeline:hist_equalize", single_perf):
            gray_eq = hist_equalize(gray_init)
        with time_section("pipeline:erode", single_perf):
            gray_eroded = erode(gray_eq, er_iter=er_iter, ksize=er_k) if False else erode(gray_eq, er_k, er_iter)
        with time_section("pipeline:binarize", single_perf):
            bw = binarize(gray_eroded, th_mode, th_value, th_block, th_C)
        with time_section("pipeline:watershed", single_perf):
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

    tuned_templates = []
    class_names = []
    if enable_tm:
        tmpl_files = st.file_uploader("Upload template image(s)", type=["png","jpg","jpeg","webp","tif","tiff"], key="tmpl_single", accept_multiple_files=True)
        if tmpl_files:
            # pick search image
            if tm_source == "Original":
                search_img = img_bgr_in.copy(); export_pil_img = pil_in
            elif tm_source == "Denoised":
                search_img = img_bgr.copy(); export_pil_img = pil_denoised
            else:
                if bw is None:
                    st.warning("Pipeline Binary not available; using Denoised.")
                    search_img = img_bgr.copy(); export_pil_img = pil_denoised
                else:
                    search_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                    export_pil_img = cv_to_pil(search_img)

            method_map = {"TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED, "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED, "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED}
            tm_method = method_map[tm_method_name]

            st.subheader("Templates Preview")
            for f in tmpl_files:
                p = load_image(f)
                st.image(p, width=120)
                tuned_templates.append((getattr(f, 'name', 'tmpl'), download_bytes(p, fmt="PNG")))

            # matching
            all_rects: List[tuple] = []
            all_scores: List[float] = []
            all_tids: List[int] = []
            palette = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255),(255,0,128),(0,255,128)]

            for tidx, (nm, b) in enumerate(tuned_templates):
                with time_section(f"tm[{tidx}]:decode_template", single_perf):
                    tmpl_pil = Image.open(io.BytesIO(b)).convert("RGB")
                    tmpl_bgr = pil_to_cv(tmpl_pil)
                with time_section(f"tm[{tidx}]:matchTemplate", single_perf):
                    res = run_template_matching(search_img, tmpl_bgr, tm_method)
                h, w = tmpl_bgr.shape[:2]

                rects, scores = [], []
                with time_section(f"tm[{tidx}]:collect_candidates", single_perf):
                    is_sqdiff = (tm_method == cv2.TM_SQDIFF_NORMED)
                    res_for_max = (1.0 - res) if is_sqdiff else res
                    thr = (1.0 - tm_thresh) if is_sqdiff else tm_thresh
                    # local maxima filter (dilate) + threshold
                    ks = 5
                    kernel = np.ones((ks, ks), np.uint8)
                    res_dil = cv2.dilate(res_for_max, kernel)
                    peak_mask = (res_for_max >= (res_dil - 1e-12)) & (res_for_max >= thr)
                    ys, xs = np.where(peak_mask)
                    vals = res_for_max[ys, xs]
                    # top-K prune
                    K = 800
                    if vals.size > K:
                        kth = np.partition(vals, -K)[-K]
                        sel = vals >= kth
                        xs, ys, vals = xs[sel], ys[sel], vals[sel]
                    rects, scores = [], []
                    for x, y, v in zip(xs, ys, vals):
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        rects.append((x1, y1, x2, y2))
                        scores.append(float(v))

                with time_section(f"tm[{tidx}]:nms", single_perf):
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

            st.subheader("Detections (preview)")
            annotated_pil = cv_to_pil(draw)
            st.image(annotated_pil, use_column_width=True)

            st.markdown("**Assign class names** (per template)")
            for tidx in range(len(tuned_templates)):
                class_names.append(st.text_input(f"Class name for template #{tidx}", value=f"class_{tidx}", key=f"cls_tune_{tidx}"))

            if st.button("Save templates & settings for batch"):
                st.session_state.tm_templates = tuned_templates
                st.session_state.tm_class_names = class_names
                st.session_state.tm_settings = {
                    "tm_method_name": tm_method_name,
                    "tm_thresh": tm_thresh,
                    "tm_max_det": tm_max_det,
                    "tm_iou": tm_iou,
                    "tm_source": tm_source,
                }
                st.success("Saved! Switch to 'Batch export' and upload many images.")

    if show_perf:
        with st.expander("⏱ Performance (single image)", expanded=False):
            render_perf_stats("Single-image timings", single_perf)

# =============================
# BATCH MODE (EXPORT)
# =============================
else:
    # Templates: prefer tuned ones from session; fallback to new upload
    tuned_templates = st.session_state.tm_templates
    class_names = st.session_state.tm_class_names
    tm_settings = st.session_state.tm_settings

    st.subheader("Templates for batch")
    if tuned_templates is None:
        st.info("No saved templates from tuning. Upload templates now.")
        up_tmpls = st.file_uploader("Upload template image(s)", type=["png","jpg","jpeg","webp","tif","tiff"], key="tmpl_batch", accept_multiple_files=True)
        tuned_templates = []
        if up_tmpls:
            for f in up_tmpls:
                p = load_image(f)
                tuned_templates.append((getattr(f, 'name', 'tmpl'), download_bytes(p, fmt="PNG")))
        if class_names is None and tuned_templates:
            class_names = [f"class_{i}" for i in range(len(tuned_templates))]
    else:
        # show previews
        cols = st.columns(6)
        for i, (nm, b) in enumerate(tuned_templates[:6]):
            with cols[i % 6]:
                st.image(Image.open(io.BytesIO(b)), caption=nm, use_column_width=True)

    # Allow editing class names
    if tuned_templates:
        st.markdown("**Class names (editable)**")
        new_names = []
        for i in range(len(tuned_templates)):
            default = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
            new_names.append(st.text_input(f"Class #{i}", value=default, key=f"cls_batch_{i}"))
        class_names = new_names

    # Batch images
    uploaded_files = st.file_uploader("Upload multiple images", type=["png","jpg","jpeg","webp","tif","tiff"], accept_multiple_files=True)
    if not uploaded_files:
        st.stop()

    # Use tm_settings if available; else current sidebar values
    _tm = tm_settings or {
        "tm_method_name": tm_method_name,
        "tm_thresh": tm_thresh,
        "tm_max_det": tm_max_det,
        "tm_iou": tm_iou,
        "tm_source": tm_source,
    }

    method_map = {"TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED, "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED, "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED}

    batch_perf_overall: List[Tuple[str, float]] = []
    batch_rows: List[Tuple[str, List[Tuple[str, float]]]] = []

    # Pre-decode templates once for batch to avoid repeated work
    decoded_templates = None
    if tuned_templates:
        decoded_templates = []
        for nm, b in tuned_templates:
            t0 = time.perf_counter()
            tmpl_pil = Image.open(io.BytesIO(b)).convert("RGB")
            tmpl_bgr = pil_to_cv(tmpl_pil)
            decoded_templates.append((nm, tmpl_bgr))
        # Note: decoding time is not counted per-image; it is one-off

    # Process images
    images_info: List[Dict] = []
    for f in uploaded_files:
        per_image_perf: List[Tuple[str, float]] = []
        with time_section("load_image", per_image_perf):
            pil_in = load_image(f)
        with time_section("pil_to_cv", per_image_perf):
            img_bgr_in = pil_to_cv(pil_in)
        with time_section(f"denoise:{method}", per_image_perf):
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
        }
        if use_pipeline:
            with time_section("pipeline:to_gray", per_image_perf):
                gray_init = to_gray(img_bgr)
            with time_section("pipeline:hist_equalize", per_image_perf):
                gray_eq = hist_equalize(gray_init)
            with time_section("pipeline:erode", per_image_perf):
                gray_eroded = erode(gray_eq, er_k, er_iter)
            with time_section("pipeline:binarize", per_image_perf):
                bw = binarize(gray_eroded, th_mode, th_value, th_block, th_C)
            entry.update({"bw": bw})
        images_info.append(entry)
        batch_rows.append((entry["name"], per_image_perf))

    # Run TM per image
    palette = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255),(255,0,128),(0,255,128)]
    all_results: Dict[str, Dict] = {}

    if not tuned_templates:
        st.warning("No templates provided. Export will include empty labels.")

    for info in images_info:
        # choose search source
        src_choice = _tm["tm_source"]
        if src_choice == "Original":
            search_img = info["img_bgr_in"].copy(); export_pil_img = info["pil_in"]
        elif src_choice == "Denoised":
            search_img = info["img_bgr"].copy(); export_pil_img = info["pil_denoised"]
        else:
            if info.get("bw") is None:
                search_img = info["img_bgr"].copy(); export_pil_img = info["pil_denoised"]
            else:
                search_img = cv2.cvtColor(info["bw"], cv2.COLOR_GRAY2BGR)
                export_pil_img = cv_to_pil(search_img)

        all_rects: List[tuple] = []
        all_scores: List[float] = []
        all_tids: List[int] = []

        if tuned_templates:
            tm_method = method_map[_tm["tm_method_name"]]
            # Use pre-decoded templates
            for tidx, (nm, tmpl_bgr) in enumerate(decoded_templates or []):
                tmp_perf: List[Tuple[str, float]] = []
                with time_section(f"tm[{tidx}]:matchTemplate", tmp_perf):
                    res = run_template_matching(search_img, tmpl_bgr, tm_method)
                h, w = tmpl_bgr.shape[:2]

                rects, scores = [], []
                with time_section(f"tm[{tidx}]:collect_candidates", tmp_perf):
                    is_sqdiff = (tm_method == cv2.TM_SQDIFF_NORMED)
                    res_for_max = (1.0 - res) if is_sqdiff else res
                    thr = (1.0 - _tm["tm_thresh"]) if is_sqdiff else _tm["tm_thresh"]
                    ks = 5
                    kernel = np.ones((ks, ks), np.uint8)
                    res_dil = cv2.dilate(res_for_max, kernel)
                    peak_mask = (res_for_max >= (res_dil - 1e-12)) & (res_for_max >= thr)
                    ys, xs = np.where(peak_mask)
                    vals = res_for_max[ys, xs]
                    K = 800
                    if vals.size > K:
                        kth = np.partition(vals, -K)[-K]
                        sel = vals >= kth
                        xs, ys, vals = xs[sel], ys[sel], vals[sel]
                    rects, scores = [], []
                    for x, y, v in zip(xs, ys, vals):
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        rects.append((x1, y1, x2, y2))
                        scores.append(float(v))

                with time_section(f"tm[{tidx}]:nms", tmp_perf):
                    keep = nms_rects(rects, scores, _tm["tm_iou"]) 
                for k in keep:
                    all_rects.append(rects[k])
                    all_scores.append(scores[k])
                    all_tids.append(tidx)
                # accumulate tm perf into per-image perf rows for visibility
                batch_rows.append((f"{info['name']}::tm[{tidx}]", tmp_perf))

        order = list(range(len(all_rects)))
        order.sort(key=lambda i: all_scores[i], reverse=True)
        order = order[:_tm["tm_max_det"]]

        draw = search_img.copy()
        for i in order:
            x1, y1, x2, y2 = all_rects[i]
            tid = all_tids[i] if all_tids else 0
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

    # ---- Export section
    with st.expander("📦 Export YOLO dataset (auto train/val split) + annotated previews"):
        if not tuned_templates:
            st.warning("You have no templates. Labels will be empty unless you go back to Single mode and save templates.")
        if not class_names and tuned_templates:
            class_names = [f"class_{i}" for i in range(len(tuned_templates))]

        # editable class names
        if tuned_templates:
            st.markdown("**Class names (editable)**")
            new_names = []
            for i in range(len(tuned_templates)):
                default = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                new_names.append(st.text_input(f"Class #{i}", value=default, key=f"cls_export_{i}"))
            class_names = new_names

        train_ratio = st.slider("Train ratio", 0.1, 0.95, 0.8, 0.05)
        seed = st.number_input("Random seed", value=42, step=1)
        img_fmt = st.selectbox("Image format", ("JPEG","PNG"), index=0, key="yolo_imgfmt_export")
        q = 95
        if img_fmt == "JPEG":
            q = st.slider("JPEG quality", 60, 100, 95, 1, key="yolo_jpgq_export")

        # entries with annotated previews
        entries: List[Tuple[Image.Image, str, List[tuple], List[int], Image.Image]] = []
        for info in images_info:
            name = info["name"]
            if name not in all_results:
                export_pil = info["pil_denoised"] if _tm["tm_source"] == "Denoised" else info["pil_in"]
                entries.append((export_pil, name, [], [], export_pil))
            else:
                res = all_results[name]
                entries.append((res["export_pil"], name, res["rects"], res["tids"], res["annotated_pil"]))

        per_split = simple_train_val_split(len(entries), train_ratio, int(seed))

        if st.button("Build YOLO dataset zip (with annotated previews)", type="primary"):
            try:
                yolo_zip = build_yolo_zip_with_ann(
                    image_entries=entries,
                    class_names=class_names if class_names else [],
                    per_entry_split=per_split,
                    img_format=img_fmt,
                    quality=q,
                )
                st.download_button(
                    "⬇️ Download yolo_dataset_with_annotated.zip",
                    data=yolo_zip,
                    file_name="yolo_dataset_with_annotated.zip",
                    mime="application/zip",
                )
                st.success("Dataset ready. data=dataset/dataset.yaml (annotated previews in dataset/annotated/...).")
            except Exception as e:
                st.error(f"Failed to build dataset: {e}")

    if show_perf and batch_rows:
        with st.expander("⏱ Performance (batch)", expanded=False):
            # Show last N images' perf summaries for brevity
            max_rows = 8
            for name, stats in batch_rows[-max_rows:]:
                render_perf_stats(str(name), stats)
