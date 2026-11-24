# -*- coding: utf-8 -*-
"""MediCheck – demo Streamlit (NHÓM 03, Học sâu, ICTU).

Chạy:  python -m streamlit run app.py
Luồng: ảnh đơn thuốc → DBNet → VietOCR → PhoBERT → M5 (tập mã ứng viên C_đơn)
       ảnh viên thuốc → YOLO11 → ConvNeXt-T + ArcFace → so với prototype của C_đơn, ngưỡng τ
       danh sách hoạt chất → tra cặp tương tác trong DDInter 2.0
Giao diện:
  - Trang đầu giới thiệu hệ thống làm gì (3 bước) và cho chọn dữ liệu ngay giữa màn hình: ca mẫu có nhãn thật hoặc tải ảnh lên.
  - Thiếu ảnh đơn hoặc ảnh viên thì báo rõ hệ thống sẽ làm được gì và cần tải thêm gì.
  - Ca mẫu: sau khi nhận dạng, so với nhãn thật (meta.json) để tính độ chính xác từng nhánh.
  - Kết luận tổng hợp ở đầu phần kết quả, ba thẻ chi tiết bên dưới; số liệu kỹ thuật để trong các mục thu gọn.
"""
import contextlib
import hashlib
import html
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("MEDICHECK_MODELS", HERE / "models"))
DATA = HERE / "data"
SAMPLES = DATA / "samples"

st.set_page_config(page_title="MediCheck – Kiểm tra thuốc theo đơn", page_icon="💊", layout="wide")

OK, WARN, BAD, INFO, MUTED = (22, 163, 74), (217, 119, 6), (220, 38, 38), (3, 105, 161), (100, 116, 139)
GT_COLOR = (37, 99, 235)
FIELD_COLOR = {"drugname": (214, 39, 40), "quantity": (255, 127, 14), "usage": (44, 160, 44),
               "diagnose": (148, 103, 189), "date": (31, 119, 180), "other": (150, 150, 150)}
FIELD_VI = {"drugname": "tên thuốc", "quantity": "số lượng", "usage": "cách dùng",
            "diagnose": "chẩn đoán", "date": "ngày", "other": "khác"}
# Trạng thái nội bộ (giống medicheck/pill.py, thêm "chưa đối chiếu" khi không có đơn / đơn không có thuốc nào)
STATUS = {"khớp": ("🟢", "Đúng thuốc trong đơn", OK),
          "tin cậy thấp": ("🟠", "Cần kiểm tra lại", WARN),
          "ngoài đơn": ("🔴", "Không có trong đơn", BAD),
          "chưa đối chiếu": ("⚪", "Chưa đối chiếu được", MUTED)}
LEVEL = {"Major": ("Nghiêm trọng", BAD), "Moderate": ("Trung bình", WARN), "Minor": ("Nhẹ", INFO),
         "Unknown": ("Chưa rõ mức độ", MUTED)}
OUT_OF_PRES = 107          # nhãn VAIPE của viên ngoài đơn
IOU_MATCH = 0.5            # ghép khung dự đoán với khung nhãn thật

CSS = """<style>
.block-container{padding-top:1.2rem;max-width:1440px}
.mc-hero{border-radius:14px;background:linear-gradient(120deg,#0f766e 0%,#0e7490 100%);color:#fff;padding:16px 22px;margin-bottom:12px;text-align:center}
.mc-hero .t{font-size:1.45rem;font-weight:700;letter-spacing:.2px}
.mc-hero .s{font-size:.95rem;opacity:.92;margin-top:2px}
.mc-run{display:flex;justify-content:center;flex-wrap:wrap;gap:6px 8px;margin-top:10px;align-items:center}
.mc-run .p{background:rgba(255,255,255,.92);color:#0f172a;border-radius:999px;padding:3px 12px;font-size:.85rem}
.mc-run .p.off{background:rgba(255,255,255,.2);color:#fff}
.mc-run .arr{opacity:.8}
.mc-sec{font-size:1.12rem;font-weight:700;color:#0f172a;margin:2px 0 6px}
.mc-sec small{font-weight:500;color:#64748b;font-size:.85rem;margin-left:6px}
.mc-up-h{font-weight:700;color:#0f172a;font-size:1rem}
.mc-up-d{font-size:.84rem;color:#64748b;margin:0 0 4px}
.mc-card{font-size:.86rem;color:#334155;margin:4px 0 6px}
.mc-card b{color:#0f172a}
.mc-case{color:#475569;font-size:.88rem;margin:2px 2px 10px}
.mc-verdict{border:1px solid;border-left-width:6px;border-radius:12px;padding:14px 18px;margin:0 0 14px}
.mc-verdict.ok{background:#f0fdf4;border-color:#86efac;border-left-color:#16a34a;color:#14532d}
.mc-verdict.warn{background:#fffbeb;border-color:#fcd34d;border-left-color:#d97706;color:#78350f}
.mc-verdict.bad{background:#fef2f2;border-color:#fca5a5;border-left-color:#dc2626;color:#7f1d1d}
.mc-verdict.info{background:#f0f9ff;border-color:#7dd3fc;border-left-color:#0284c7;color:#0c4a6e}
.mc-verdict .t{font-weight:700;font-size:1.1rem}
.mc-verdict ul{margin:6px 0 0 1.1rem;padding:0}
.mc-verdict li{margin:2px 0}
.mc-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 12px}
@media (max-width:900px){.mc-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.mc-kpi{border:1px solid #e2e8f0;border-radius:12px;padding:12px 16px;background:#fff}
.mc-kpi .v{font-size:1.7rem;font-weight:700;line-height:1.15;color:#0f172a;font-variant-numeric:tabular-nums}
.mc-kpi .l{font-size:.82rem;color:#64748b;margin-top:2px}
.mc-kpi .x{font-size:.76rem;color:#94a3b8;margin-top:2px}
.mc-kpi.bad .v{color:#b91c1c}.mc-kpi.warn .v{color:#b45309}.mc-kpi.ok .v{color:#15803d}
.mc-eval{border:1px solid #bfdbfe;background:#eff6ff;border-radius:12px;padding:12px 16px 4px;margin:0 0 14px}
.mc-eval .t{font-weight:700;color:#1e3a8a;font-size:1.05rem}
.mc-eval .s{font-size:.84rem;color:#475569;margin:2px 0 10px}
.mc-eval .mc-kpi{border-color:#dbeafe}
.mc-chip{display:inline-block;padding:1px 10px;border:1px solid;border-radius:999px;font-size:.78rem;font-weight:600;margin:2px 4px 2px 0;white-space:nowrap}
.mc-drug{border:1px solid #e2e8f0;border-left:5px solid #0f766e;border-radius:10px;padding:10px 14px;margin:0 0 10px;background:#fff}
.mc-drug .h{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;flex-wrap:wrap}
.mc-drug .n{font-weight:700;font-size:1.02rem;color:#0f172a}
.mc-drug .row{font-size:.9rem;color:#1e293b;margin-top:3px}
.mc-drug .k{display:inline-block;min-width:84px;color:#64748b}
.mc-drug .ocr{font-size:.78rem;color:#64748b;margin-top:5px}
.mc-ddi{border:1px solid #e2e8f0;border-left:5px solid;border-radius:10px;padding:10px 14px;margin:0 0 10px}
.mc-ddi .p{font-weight:700;color:#0f172a;margin-left:2px}
.mc-ddi .i{font-size:.86rem;color:#475569;margin-top:3px}
.mc-side-title{font-size:1.35rem;font-weight:700;color:#0f766e;margin-bottom:-4px}
.stTabs [data-baseweb="tab"] p{font-size:1rem;font-weight:600}
</style>"""


# ----------------------------------------------------------------------------- tải mô hình (một lần)
@st.cache_resource(show_spinner="Đang nạp mô hình đọc đơn thuốc (DBNet, VietOCR, PhoBERT)…")
def load_pres():
    from medicheck.pres import PrescriptionReader
    return PrescriptionReader(MODELS)


@st.cache_resource(show_spinner="Đang nạp mô hình nhận dạng viên thuốc (YOLO11, ConvNeXt + ArcFace)…")
def load_pill():
    from medicheck.pill import PillMatcher
    return PillMatcher(MODELS)


@st.cache_resource
def load_ddi():
    from medicheck.ddi import DDIChecker, DrugInfo
    info = DrugInfo(DATA / "drug_info.csv")
    return info, DDIChecker(DATA / "ddinter_vaipe.csv", info)


def missing_models():
    need = ["dbnet_vaipe.pt", "vietocr_vaipe.pth", "vietocr_config.yml", "phobert_out/field_cls.pt",
            "phobert_out/config.json", "drug_dict.json", "yolo_pill_1cls.pt", "metrics_match.json",
            "embed_arcface_imagenet/encoder.pt", "embed_arcface_imagenet/gallery.npy"]
    return [n for n in need if not (MODELS / n).exists()]


# ----------------------------------------------------------------------------- tiện ích
def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_image(data):
    """Mở ảnh từ bytes và xoay đúng chiều theo EXIF (ảnh chụp bằng điện thoại). Ảnh hỏng → None."""
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None


def digest(data):
    return hashlib.md5(data).hexdigest()[:16]


def esc(t):
    return html.escape(str(t))


def rgb(c, a=None):
    return f"rgb({c[0]},{c[1]},{c[2]})" if a is None else f"rgba({c[0]},{c[1]},{c[2]},{a})"


def chip(text, c):
    dark = tuple(int(v * 0.7) for v in c)          # chữ đậm hơn nền để đủ tương phản
    return (f"<span class='mc-chip' style='background:{rgb(c, .10)};border-color:{rgb(c, .45)};"
            f"color:{rgb(dark)}'>{esc(text)}</span>")


def show_html(s):
    st.markdown(s, unsafe_allow_html=True)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def kpi(value, label, cls="", extra=""):
    return (f"<div class='mc-kpi {cls}'><div class='v'>{value}</div><div class='l'>{esc(label)}</div>"
            + (f"<div class='x'>{esc(extra)}</div>" if extra else "") + "</div>")


@contextlib.contextmanager
def out_of_memory_guard():
    """Máy thiếu RAM: torch báo 'not enough memory' hoặc oneDNN báo 'could not create a primitive'.
    Hiện hướng dẫn dễ hiểu thay cho traceback."""
    try:
        yield
    except RuntimeError as e:
        msg = str(e)
        if "not enough memory" not in msg and "could not create a primitive" not in msg:
            raise
        st.error("**Máy không đủ RAM để chạy mô hình.** App cần khoảng 3 GB RAM trống. Hãy tắt bớt chương trình "
                 "(Docker Desktop, WSL: `wsl --shutdown`, SQL Server, bớt tab Chrome…) rồi bấm **R** để chạy lại.\n\n"
                 f"Chi tiết: `{msg[:160]}`")
        st.stop()


@st.cache_data
def thumbnail(path, w=520, ratio=0.55):
    """Ảnh thu nhỏ phần trên của đơn thuốc cho thẻ chọn ca mẫu."""
    im = Image.open(path).convert("RGB")
    h = min(im.height, int(im.width * ratio))
    im = im.crop((0, 0, im.width, h))
    return im.resize((w, int(w * h / im.width)))


# ----------------------------------------------------------------------------- so với nhãn thật (ca mẫu)
def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pill_correct(p, g_cls):
    """Viên đúng nếu: nhãn là viên ngoài đơn và hệ thống báo 'không có trong đơn';
    hoặc nhãn là thuốc trong đơn, hệ thống chấp nhận (khớp / cần kiểm tra lại) và đúng mã thuốc."""
    if g_cls == OUT_OF_PRES:
        return p["status"] == "ngoài đơn"
    return p["status"] in ("khớp", "tin cậy thấp") and p["best_cls"] == g_cls


def eval_pills(pills, gt, scale=1.0):
    """Ghép khung dự đoán với khung nhãn thật (IoU ≥ 0,5, tham lam theo IoU giảm dần).
    Trả về: pred_gt[i] = chỉ số nhãn ghép với viên dự đoán i (hoặc None), danh sách nhãn bị bỏ sót, gt đã đổi tỉ lệ."""
    gt = [dict(g, box=[v * scale for v in g["box"]]) for g in gt]
    pairs = sorted(((iou(p["box"], g["box"]), i, j) for i, p in enumerate(pills) for j, g in enumerate(gt)), reverse=True)
    pred_gt, used = [None] * len(pills), set()
    for v, i, j in pairs:
        if v < IOU_MATCH:
            break
        if pred_gt[i] is None and j not in used:
            pred_gt[i] = j
            used.add(j)
    missed = [j for j in range(len(gt)) if j not in used]
    return pred_gt, missed, gt


# ----------------------------------------------------------------------------- vẽ
def draw_pres(img, lines, hide_other):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for ln in lines:
        if hide_other and ln["field"] == "other":
            continue
        c = FIELD_COLOR.get(ln["field"], (0, 0, 0))
        b = ln["box"]
        d.rectangle([b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], outline=c, width=3 if ln["field"] != "other" else 1)
    return im


def draw_pills(img, pills, info, gt=None, missed=()):
    im = img.copy()
    d = ImageDraw.Draw(im)
    f = font(max(14, im.width // 45))
    lw = max(3, im.width // 250)
    if gt is not None:                               # khung nhãn thật (xanh dương, mảnh) để so sánh
        for j, g in enumerate(gt):
            d.rectangle(g["box"], outline=GT_COLOR, width=max(2, lw - 1))
            if j in missed:
                t = "bỏ sót"
                tw = d.textlength(t, font=f)
                x, y = g["box"][0], g["box"][3] + 2
                d.rectangle([x, y, x + tw + 8, y + f.size + 6], fill=GT_COLOR)
                d.text((x + 4, y + 2), t, font=f, fill="white")
    for i, p in enumerate(pills, 1):
        c = STATUS[p["status"]][2]
        b = p["box"]
        d.rectangle(b, outline=c, width=lw)
        if p["status"] == "ngoài đơn":
            t = f"#{i} không có trong đơn"
        elif p["status"] == "chưa đối chiếu":
            t = f"#{i} ? {info.name(p['global_cls'])}"
        else:
            t = f"#{i} {info.name(p['best_cls'])}"
        tw = d.textlength(t, font=f)
        x = max(0, min(b[0], im.width - tw - 8))
        y = b[1] - f.size - 8 if b[1] > f.size + 8 else b[3] + 2
        d.rectangle([x, y, x + tw + 8, y + f.size + 6], fill=c)
        d.text((x + 4, y + 2), t, font=f, fill="white")
    return im


def hero(statuses=None):
    """Đầu trang gọn: tên + một dòng mô tả. statuses: [(tên bước, kết quả, đã chạy?)] → một hàng trạng thái ngắn."""
    row = ""
    if statuses:
        row = "<div class='mc-run'>" + "<span class='arr'>➜</span>".join(
            f"<span class='p{'' if ok else ' off'}'><b>{n}</b> {esc(t)}</span>" for n, t, ok in statuses) + "</div>"
    show_html("<div class='mc-hero'><div class='t'>💊 MediCheck — Kiểm tra viên thuốc có đúng với đơn không</div>"
              "<div class='s'>Đọc đơn thuốc · Đối chiếu từng viên với đơn · Cảnh báo tương tác thuốc</div>" + row + "</div>")


# ----------------------------------------------------------------------------- dữ liệu ca mẫu
show_html(CSS)
sample_meta = {}
if SAMPLES.exists():
    for p in sorted(SAMPLES.iterdir()):
        if (p / "meta.json").exists():
            sample_meta[p.name] = json.loads((p / "meta.json").read_text(encoding="utf-8"))

ss = st.session_state
hero_slot = st.empty()          # điền sau khi biết đã có dữ liệu / kết quả hay chưa

# ----------------------------------------------------------------------------- BƯỚC CHỌN DỮ LIỆU (giữa màn hình)
MODE_SAMPLE, MODE_UPLOAD = "🧪 Dùng ca mẫu có sẵn (có nhãn thật)", "📤 Tải ảnh của bạn lên"
meta, sid, key, pres_bytes, pill_src = None, None, None, None, []
bad_files = []
# Đã chọn ca mẫu thì thu gọn khung chọn để kết quả lên cao; chế độ tải ảnh luôn mở để còn tải thêm ảnh
_picked = (ss.get("src_mode") or MODE_SAMPLE) == MODE_SAMPLE and ss.get("sample") in sample_meta
_label = (f"📂 Đang xem ca mẫu {list(sample_meta).index(ss.sample) + 1} — bấm để chọn ca khác hoặc tải ảnh của bạn"
          if _picked else "📂 Chọn dữ liệu để kiểm tra")
_mid = st.columns([1, 6, 1])[1]          # khung chọn dữ liệu nằm giữa màn hình
with _mid, st.expander(_label, expanded=not _picked):
    show_html("<div class='mc-sec'>Chọn dữ liệu để kiểm tra"
              "<small>dùng ca mẫu để xem nhanh, hoặc tải ảnh đơn thuốc và ảnh viên thuốc của bạn</small></div>")
    mode = st.segmented_control("Nguồn dữ liệu", [MODE_SAMPLE, MODE_UPLOAD], default=MODE_SAMPLE, key="src_mode",
                                label_visibility="collapsed") or MODE_SAMPLE
    if mode == MODE_SAMPLE and sample_meta:
        st.caption("Ba ca lấy từ tập test của nhóm (VAIPE, không dùng khi huấn luyện). Vì có nhãn thật nên sau khi "
                   "nhận dạng, hệ thống tự tính độ chính xác.")
        cols = st.columns(len(sample_meta), gap="medium")
        for n, (c, (s, m)) in enumerate(zip(cols, sample_meta.items()), 1):
            n_img = len(list((SAMPLES / s).glob("anh_vien_*.jpg")))
            n_pill = sum(len(v) for v in m.get("anh_vien", {}).values())
            with c:
                st.image(thumbnail(str(SAMPLES / s / "don_thuoc.png")), width="stretch")
                show_html(f"<div class='mc-card'><b>Ca {n}</b> · {len(m.get('thuoc_trong_don', []))} thuốc trong đơn · "
                          f"{n_img} ảnh viên ({n_pill} viên)<br>{esc(m.get('ghi_chu', ''))}</div>")
                chosen = ss.get("sample") == s
                if st.button("✔ Đang xem ca này" if chosen else "Xem ca này", key=f"pick_{s}",
                             type="primary" if chosen else "secondary", width="stretch"):
                    ss.sample = s
                    st.rerun()
        sid = ss.get("sample")
        if sid in sample_meta:
            meta = sample_meta[sid]
            sd = SAMPLES / sid
            pres_bytes = (sd / "don_thuoc.png").read_bytes()
            key = meta.get("id") or sid
            pill_src = [(p.name, p.read_bytes()) for p in sorted(sd.glob("anh_vien_*.jpg"))]
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            show_html("<div class='mc-up-h'>📄 Ảnh đơn thuốc</div>"
                      "<div class='mc-up-d'>1 ảnh, chụp thẳng, đủ sáng, thấy rõ các dòng tên thuốc.</div>")
            up = st.file_uploader("Ảnh đơn thuốc", type=["png", "jpg", "jpeg"], key="up_pres", label_visibility="collapsed")
        with c2:
            show_html("<div class='mc-up-h'>💊 Ảnh viên thuốc</div>"
                      "<div class='mc-up-d'>Có thể chọn nhiều ảnh; mỗi ảnh có thể có nhiều viên, đặt trên nền trơn.</div>")
            ups = st.file_uploader("Ảnh viên thuốc", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
                                   key="up_pills", label_visibility="collapsed")
        if up:
            pres_bytes = up.getvalue()
            key = "upload-" + digest(pres_bytes)        # băm toàn bộ file: hai đơn khác nhau không bao giờ dùng chung kết quả
        pill_src = [(u.name, u.getvalue()) for u in ups or []]

# Thông báo khi có ảnh mới (toast góc phải) — chỉ báo một lần cho mỗi file
seen = ss.setdefault("seen_files", set())
if pres_bytes is not None and mode == MODE_UPLOAD and key not in seen:
    seen.add(key)
    st.toast("Đã nhận ảnh đơn thuốc", icon="📄")
new_pills = [n for n, b in pill_src if f"{n}:{digest(b)}" not in seen]
if new_pills and mode == MODE_UPLOAD:
    seen.update(f"{n}:{digest(b)}" for n, b in pill_src)
    st.toast(f"Đã nhận {len(new_pills)} ảnh viên thuốc", icon="💊")

# Ảnh hỏng → bỏ ra và báo
pres_img = None
if pres_bytes is not None:
    pres_img = load_image(pres_bytes)
    if pres_img is None:
        bad_files.append("ảnh đơn thuốc")
        pres_bytes, key = None, None
pill_imgs = []
for n, b in pill_src:
    im = load_image(b)
    if im is None:
        bad_files.append(n)
    else:
        pill_imgs.append(dict(name=n, key=f"{n}:{digest(b)}", img=im))
has_pres, has_pills = pres_img is not None, bool(pill_imgs)

# Thông báo thiếu ảnh: hệ thống làm được gì, cần tải thêm gì
if bad_files:
    _mid.error("Không mở được: " + ", ".join(bad_files) + ". Hãy chọn ảnh PNG hoặc JPG khác.", icon="🚫")
if mode == MODE_UPLOAD:
    if not has_pres and not has_pills:
        _mid.info("Hãy tải **ảnh đơn thuốc** vào ô bên trái và **ảnh viên thuốc** vào ô bên phải. Có thể tải từng loại; "
                "kết quả hiện ngay bên dưới sau khi tải.", icon="👆")
    elif has_pres and not has_pills:
        _mid.info("**Đã có ảnh đơn thuốc, chưa có ảnh viên thuốc.** Hệ thống sẽ đọc đơn và kiểm tra tương tác giữa các thuốc "
                "trong đơn. Tải thêm ảnh viên thuốc ở ô bên phải để đối chiếu từng viên với đơn.", icon="📄")
    elif has_pills and not has_pres:
        _mid.warning("**Đã có ảnh viên thuốc, chưa có ảnh đơn thuốc.** Không có đơn thì chưa biết viên nào đúng đơn: hệ thống "
                   "chỉ đoán tên gần nhất trong 107 loại thuốc của VAIPE để tham khảo, không kết luận đúng/sai. "
                   "Tải thêm ảnh đơn thuốc ở ô bên trái để đối chiếu.", icon="💊")
    else:
        _mid.success(f"Đã có ảnh đơn thuốc và {len(pill_imgs)} ảnh viên thuốc — kết quả ở bên dưới.", icon="✅")

STEPS = [("Đọc đơn thuốc", "Tìm các dòng chữ, đọc chữ tiếng Việt, phân loại dòng nào là tên thuốc rồi chuẩn hóa "
          "về danh mục thuốc.", ["DBNet", "VietOCR", "PhoBERT", "So khớp tên"]),
         ("Đối chiếu từng viên", "Tìm từng viên trong ảnh, trích đặc trưng và so với các thuốc <b>có trong đơn</b>; viên "
          "không giống thuốc nào trong đơn bị báo là viên lạ.", ["YOLO11s", "ConvNeXt-T + ArcFace", "Ngưỡng τ"]),
         ("Cảnh báo tương tác", "Lấy hoạt chất của các thuốc trong đơn và tra mọi cặp trong cơ sở dữ liệu tương tác thuốc.",
          ["DDInter 2.0", "342 cặp · 39 hoạt chất"])]

if not has_pres and not has_pills:
    with hero_slot.container():
        hero()
    st.stop()

# ----------------------------------------------------------------------------- thanh bên: cài đặt
with st.sidebar:
    show_html("<div class='mc-side-title'>💊 MediCheck</div>")
    st.caption("Demo đồ án Học sâu — NHÓM 03 (ICTU). Huấn luyện và đánh giá trên bộ dữ liệu VAIPE; chạy hoàn toàn trên CPU.")
    with st.expander("⚙️ Cài đặt nâng cao"):
        pill_tau = st.slider("Ngưỡng chấp nhận τ (cosin)", 0.10, 0.80, 0.36, 0.01,
                             help="Độ tương đồng tối thiểu giữa viên thuốc và thuốc trong đơn. Dưới τ → 'Không có trong đơn'. "
                                  "τ = 0,36 chọn trên tập val (notebook 06).")
        margin = st.slider("Vùng 'cần kiểm tra lại' (τ … τ + x)", 0.0, 0.20, 0.05, 0.01)
        hide_other = st.checkbox("Ẩn khung dòng 'khác' trên ảnh đơn", value=True)
        use_true_cdon = bool(meta) and st.checkbox(
            "Dùng danh sách thuốc đúng (nhãn) thay cho kết quả đọc đơn", value=False, disabled=not meta,
            help="Chỉ có ở ca mẫu. Dùng để tách riêng sai số của nhánh viên thuốc khỏi sai số đọc đơn.")
    st.warning("Công cụ hỗ trợ học tập, nghiên cứu — **không thay thế** đánh giá của bác sĩ, dược sĩ.", icon="⚠️")

miss = missing_models()
if miss:
    st.error("Chưa có đủ trọng số trong thư mục `models/`. Hãy chạy `python setup_models.py` (đọc README). Thiếu: "
             + ", ".join(miss))
    st.stop()
info, ddi = load_ddi()

# ----------------------------------------------------------------------------- chạy mô hình
pres = None
if has_pres:
    if ss.get("pres_key") != key:
        try:
            reader = load_pres()
        except OSError as e:
            st.error("Không nạp được bộ tách từ PhoBERT (`vinai/phobert-base`). Lần chạy đầu cần mạng; "
                     "hoặc chép `vocab.txt` và `bpe.codes` vào `models/phobert_tokenizer/`. Chi tiết: " + str(e)[:300])
            st.stop()
        with st.spinner("Bước 1 · Đang đọc đơn thuốc (CPU: khoảng 10–20 giây)…"), out_of_memory_guard():
            ss.pres = reader.read(pres_img)
        ss.pres_key = key
        ss.pop("pill_res", None)
    pres = ss.pres
drugs = pres["drugs"] if pres else []

cand = list(meta["c_don_that"]) if use_true_cdon else (pres["c_don_cand"] if pres else [])
no_context = not cand          # không có đơn hoặc đơn không có thuốc nào → không thể đối chiếu, không được báo "khớp"


def judge(score):
    if no_context:
        return "chưa đối chiếu"
    if score < pill_tau:
        return "ngoài đơn"
    return "tin cậy thấp" if score < pill_tau + margin else "khớp"


# Phân tích mọi ảnh viên một lần (tính điểm với τ = 0, áp τ ở trên để kéo thanh trượt không phải chạy lại mô hình)
ctx = key or "no-pres"
cache = ss.setdefault("pill_res", {})
todo = [p for p in pill_imgs if (ctx, p["key"], tuple(cand)) not in cache]
if todo:
    matcher = load_pill()
    with st.spinner(f"Bước 2 · Đang kiểm tra {len(todo)} ảnh viên thuốc…"), out_of_memory_guard():
        for p in todo:
            cache[(ctx, p["key"], tuple(cand))] = matcher.analyze(p["img"], cand, tau=0.0, low_margin=0.0)
pill_views = []
for p in pill_imgs:
    res = cache[(ctx, p["key"], tuple(cand))]
    pill_views.append(dict(name=p["name"], res=res, src_w=p["img"].width,
                           pills=[dict(q, status=judge(q["score"])) for q in res["pills"]]))
all_pills = [q for v in pill_views for q in v["pills"]]
n_ok = sum(q["status"] == "khớp" for q in all_pills)
n_low = sum(q["status"] == "tin cậy thấp" for q in all_pills)
n_out = sum(q["status"] == "ngoài đơn" for q in all_pills)
# Mã viên đã được xác nhận trong các ảnh của đơn hiện tại, với đúng tập C_đơn đang dùng
seen_cls = {q["best_cls"] for q in all_pills if q["status"] in ("khớp", "tin cậy thấp")}
empty_views = [v["name"] for v in pill_views if not v["pills"]]


def drug_label(dr):
    m = dr["match"]
    return info.name(m["single"]) if m["single"] is not None else dr["text"]


# Tương tác: mỗi thuốc trong đơn ứng với nhiều mã viên; nếu ảnh viên (hoặc nhãn đúng) đã xác nhận mã nào thì chỉ giữ mã đó
sure = set(meta["c_don_that"]) if use_true_cdon else seen_cls
groups = []
for dr in drugs:
    cs = dr["match"]["cand"]
    if set(cs) & sure:
        cs = sorted(set(cs) & sure)
    if cs:
        groups.append((drug_label(dr), cs))
found, nodata = ddi.check(groups) if len(groups) >= 2 else ([], [])
known = [r for r in found if r["level"] != "Unknown"]
unk = [r for r in found if r["level"] == "Unknown"]
n_major = sum(r["level"] == "Major" for r in known)
n_mod = sum(r["level"] == "Moderate" for r in known)
n_minor = sum(r["level"] == "Minor" for r in known)
unmatched = [dr for dr in drugs if not dr["match"]["cand"]]

# ----------------------------------------------------------------------------- so với nhãn thật (chỉ ca mẫu)
ev = None
if meta:
    gt_cls = list(meta["c_don_that"])
    got = set(pres["c_don_cand"]) if pres else set()
    pres_hit = [c for c in gt_cls if c in got]
    n_gt = n_det = n_right = n_fp = 0
    img_all_ok = 0
    for v in pill_views:
        gt = meta.get("anh_vien", {}).get(v["name"])
        if gt is None:
            continue
        scale = v["res"]["image"].width / v["src_w"]      # nhãn ghi trên ảnh mẫu; YOLO có thể đã thu nhỏ ảnh
        pred_gt, missed, gt_s = eval_pills(v["pills"], gt, scale)
        right = [j is not None and pill_correct(p, gt_s[j]["cls"]) for p, j in zip(v["pills"], pred_gt)]
        v.update(gt=gt_s, pred_gt=pred_gt, missed=missed, right=right)
        n_gt += len(gt_s)
        n_det += len(gt_s) - len(missed)
        n_right += sum(right)
        n_fp += sum(j is None for j in pred_gt)
        img_all_ok += int(not missed and all(right) and len(v["pills"]) == len(gt_s))
    n_eval_img = sum("gt" in v for v in pill_views)
    ev = dict(pres_hit=pres_hit, n_drug=len(gt_cls), extra=sorted(got - set(gt_cls)), n_gt=n_gt, n_det=n_det,
              n_right=n_right, n_fp=n_fp, img_ok=img_all_ok, n_img=n_eval_img)

# ----------------------------------------------------------------------------- đầu trang (có trạng thái từng bước)
t_pres = sum(pres["timing"].values()) if pres else 0
t_pill = sum(t for v in pill_views for t in v["res"]["timing"].values())
s1 = (f"✓ {len(drugs) - len(unmatched)}/{len(drugs)} tên thuốc nhận ra · {t_pres:.1f} s", False) if pres else \
     ("Chưa có ảnh đơn thuốc", True)
s2 = (f"✓ {len(all_pills)} viên · {t_pill:.1f} s" + (" · chưa có đơn để đối chiếu" if no_context else ""), False) \
    if pill_imgs else ("Chưa có ảnh viên thuốc", True)
s3 = ((f"✓ {len(known)} cặp có mức độ · {len(unk)} cặp chưa rõ", False) if len(groups) >= 2 else
      ("Cần ít nhất 2 thuốc nhận ra trong đơn", True))
with hero_slot.container():
    hero([(f"{k}. {h}", stt[0], not stt[1]) for k, ((h, _, _), stt) in enumerate(zip(STEPS, (s1, s2, s3)), 1)])

# ----------------------------------------------------------------------------- kết luận tổng hợp
st.divider()
show_html("<div class='mc-sec'>Kết quả kiểm tra</div>")
if meta:
    case = f"Ca mẫu <b>{esc(key)}</b>"
else:
    case = "Ảnh tải lên"
show_html(f"<div class='mc-case'>{case} · {'1 ảnh đơn' if pres else 'chưa có ảnh đơn'} · {len(pill_imgs)} ảnh viên thuốc"
          + (" · <b>đang dùng danh sách thuốc đúng (nhãn)</b>" if use_true_cdon else "") + "</div>")

bad, warn = [], []
if pres and not drugs:
    bad.append("Không tìm thấy dòng tên thuốc nào trong ảnh đơn. Có thể ảnh không phải đơn thuốc, bị mờ hoặc chụp nghiêng.")
elif pres and no_context:
    bad.append("Không khớp được tên thuốc nào trong đơn với danh mục — chưa thể đối chiếu viên thuốc.")
elif unmatched:
    warn.append(f"{len(unmatched)} dòng tên thuốc không khớp danh mục: "
                + ", ".join(f"<i>{esc(d['text'])}</i>" for d in unmatched) + ".")
if n_out:
    bad.append(f"<b>{n_out} viên không có trong đơn</b> — cần kiểm tra lại trước khi uống / cấp phát.")
if n_major:
    bad.append(f"<b>{n_major} cặp tương tác mức nghiêm trọng</b> — xem thẻ Tương tác thuốc.")
if n_low:
    warn.append(f"{n_low} viên có độ tương đồng thấp, nên kiểm tra lại bằng mắt.")
if n_mod:
    warn.append(f"{n_mod} cặp tương tác mức trung bình.")
if empty_views:
    warn.append("Không tìm thấy viên nào trong: " + ", ".join(esc(n) for n in empty_views)
                + ". Kiểm tra lại đó có đúng là ảnh viên thuốc không.")
if pres and not pill_imgs:
    warn.append("Chưa có ảnh viên thuốc nên chưa đối chiếu được viên nào với đơn.")
elif pres and not no_context:
    not_seen = [drug_label(dr) for dr in drugs if dr["match"]["cand"] and not set(dr["match"]["cand"]) & seen_cls]
    if not_seen:
        warn.append("Chưa thấy viên nào của: " + ", ".join(esc(x) for x in not_seen) + ".")

if not pres:
    tone, title = "info", "ℹ️ Mới nhận dạng thử, chưa đối chiếu với đơn"
    items = [f"Tìm thấy {len(all_pills)} viên trong {len(pill_imgs)} ảnh. Tên ghi trên từng viên chỉ là thuốc giống nhất "
             "trong 107 loại của VAIPE.", "Tải thêm ảnh đơn thuốc để biết viên nào đúng đơn, viên nào không."] + warn
elif bad:
    tone, title, items = "bad", "⛔ Cần kiểm tra ngay", bad + warn
elif warn:
    tone, title, items = "warn", "⚠️ Cần lưu ý", warn
else:
    tone, title = "ok", "✅ Không phát hiện vấn đề"
    items = [f"Cả {len(all_pills)} viên đều đúng thuốc trong đơn."]
    items.append(f"{n_minor} cặp tương tác mức nhẹ." if n_minor else "Không có tương tác đáng lưu ý trong DDInter.")
show_html(f"<div class='mc-verdict {tone}'><div class='t'>{title}</div><ul>"
          + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>")

n_drug_ok = len(drugs) - len(unmatched)
show_html("<div class='mc-kpis'>"
          + kpi(f"{n_drug_ok}/{len(drugs)}" if pres else "—", "Thuốc trong đơn nhận ra được",
                "bad" if pres and no_context else "warn" if unmatched else "")
          + kpi(len(all_pills) if pill_imgs else "—", "Viên thuốc đã kiểm tra")
          + kpi("—" if no_context or not pill_imgs else n_out, "Viên không có trong đơn", "bad" if n_out else "")
          + kpi(n_major + n_mod if len(groups) >= 2 else "—", "Tương tác cần lưu ý",
                "bad" if n_major else "warn" if n_mod else "")
          + "</div>")

# Độ chính xác so với nhãn thật (ca mẫu)
if ev:
    def cls_of(a, b):
        return "" if not b else "ok" if a == b else "warn" if a >= 0.8 * b else "bad"
    show_html("<div class='mc-eval'><div class='t'>🎯 Độ chính xác so với nhãn thật</div>"
              "<div class='s'>Ca mẫu có nhãn thật trong VAIPE: so kết quả của mô hình với đáp án. Viên được tính là đúng khi "
              "tìm ra viên (IoU ≥ 0,5 với khung nhãn) và kết luận đúng: đúng mã thuốc nếu viên có trong đơn, "
              "hoặc báo <i>không có trong đơn</i> nếu là viên lạ.</div><div class='mc-kpis'>"
              + kpi(f"{len(ev['pres_hit'])}/{ev['n_drug']}", "Thuốc trong đơn đọc đúng",
                    cls_of(len(ev["pres_hit"]), ev["n_drug"]),
                    f"{len(ev['extra'])} mã ứng viên thừa" if ev["extra"] else "không có mã thừa")
              + kpi(f"{ev['n_det']}/{ev['n_gt']}", "Viên được tìm thấy (YOLO)", cls_of(ev["n_det"], ev["n_gt"]),
                    f"{ev['n_fp']} khung thừa" if ev["n_fp"] else "không có khung thừa")
              + kpi(pct(ev["n_right"], ev["n_gt"]), "Viên nhận dạng đúng", cls_of(ev["n_right"], ev["n_gt"]),
                    f"{ev['n_right']}/{ev['n_gt']} viên")
              + kpi(f"{ev['img_ok']}/{ev['n_img']}", "Ảnh đúng hoàn toàn", cls_of(ev["img_ok"], ev["n_img"]),
                    "mọi viên trong ảnh đều đúng")
              + "</div></div>")

tab1, tab2, tab3 = st.tabs(["📋 Bước 1 · Đơn thuốc", "💊 Bước 2 · Đối chiếu viên thuốc", "⚠️ Bước 3 · Tương tác thuốc"])

# ----------------------------------------------------------------------------- ① đơn thuốc
with tab1:
    if not pres:
        st.info("Chưa có ảnh đơn thuốc. Tải ảnh đơn ở ô **Ảnh đơn thuốc** phía trên để hệ thống đọc tên thuốc.", icon="📄")
    else:
        c1, c2 = st.columns([5, 6], gap="large")
        with c1:
            st.image(draw_pres(pres_img, pres["lines"], hide_other), width="stretch")
            show_html(" ".join(chip(v, FIELD_COLOR[k]) for k, v in FIELD_VI.items()))
            st.caption("Khung là dòng chữ DBNet tìm thấy, màu khung là loại thông tin PhoBERT phân loại.")
        with c2:
            st.markdown("#### Thuốc kê trong đơn")
            if not drugs:
                st.warning("Không tìm thấy dòng nào được phân loại là tên thuốc.")
            gt_set = set(meta["c_don_that"]) if meta else set()
            cards = []
            for i, dr in enumerate(drugs, 1):
                m = dr["match"]
                chips = []
                if not m["cand"]:
                    chips.append(chip("Không khớp danh mục", BAD))
                elif pill_imgs and not no_context:
                    chips.append(chip("Đã thấy viên", OK) if set(m["cand"]) & seen_cls else chip("Chưa thấy viên", MUTED))
                if meta:
                    chips.append(chip("✓ Đúng nhãn", GT_COLOR) if set(m["cand"]) & gt_set else chip("✗ Không có trong nhãn", BAD))
                rows = "".join(f"<div class='row'><span class='k'>{k}</span>{esc('; '.join(v))}</div>"
                               for k, v in (("Số lượng", dr["quantity"]), ("Cách dùng", dr["usage"])) if v)
                border = rgb(BAD) if not m["cand"] else "#0f766e"
                cards.append(f"<div class='mc-drug' style='border-left-color:{border}'><div class='h'>"
                             f"<div class='n'>{i}. {esc(drug_label(dr))}</div><div>{''.join(chips)}</div></div>"
                             f"{rows}<div class='ocr'>Đơn ghi (VietOCR đọc): “{esc(dr['text'])}”</div></div>")
            show_html("".join(cards))
            if meta:
                with st.expander("🎯 So với nhãn thật của đơn", expanded=bool(ev and len(ev["pres_hit"]) < ev["n_drug"])):
                    got = set(pres["c_don_cand"])
                    st.dataframe(pd.DataFrame([{"Dòng trong đơn (nhãn)": t["dong"], "Thuốc đúng": f"{info.name(t['cls'])} [mã {t['cls']}]",
                                                "Kết quả": "✅ Đọc đúng" if t["cls"] in got else "❌ Bỏ sót"}
                                               for t in meta.get("thuoc_trong_don", [])]), hide_index=True, width="stretch")
                    if ev["extra"]:
                        st.caption("Mã ứng viên thừa (cùng tên thuốc nhưng khác nhà sản xuất / hình dáng viên): "
                                   + ", ".join(f"{info.name(c)} [mã {c}]" for c in ev["extra"])
                                   + ". Ảnh viên sẽ quyết định mã cụ thể.")
            with st.expander("Chi tiết kỹ thuật"):
                st.dataframe(pd.DataFrame([{"#": i, "Dòng OCR": dr["text"], "Tên khớp (từ điển)": dr["match"]["name"] or "—",
                                            "Điểm khớp": round(dr["match"]["score"]),
                                            "Mã ứng viên": ", ".join(map(str, dr["match"]["cand"])) or "không khớp"}
                                           for i, dr in enumerate(drugs, 1)]), hide_index=True, width="stretch")
                st.markdown(f"Tập ứng viên của đơn **C_đơn** = {{{', '.join(map(str, pres['c_don_cand']))}}}. "
                            "Một tên thuốc có thể ứng với nhiều mã viên trong VAIPE; ảnh viên sẽ quyết định mã cụ thể.")
                st.caption("Thời gian: " + " · ".join(f"{k} {v:.1f}s" for k, v in pres["timing"].items()))
                st.dataframe(pd.DataFrame([{"#": ln["idx"], "Văn bản": ln["text"], "Trường": FIELD_VI.get(ln["field"], ln["field"]),
                                            "Xác suất": round(ln["field_prob"], 2)} for ln in pres["lines"]]),
                             hide_index=True, width="stretch", height=320)

# ----------------------------------------------------------------------------- ② viên thuốc
with tab2:
    if not pill_views:
        st.info("Chưa có ảnh viên thuốc. Tải ảnh ở ô **Ảnh viên thuốc** phía trên (có thể chọn nhiều ảnh) để đối chiếu "
                "từng viên với đơn.", icon="💊")
    else:
        if no_context:
            st.warning(("Chưa có ảnh đơn thuốc" if not pres else "Chưa đọc được thuốc nào trong đơn")
                       + " nên **không thể đối chiếu**. Tên thuốc dưới đây chỉ là dự đoán gần nhất trong toàn bộ "
                       "107 loại thuốc của VAIPE, không phải kết luận.")

        def view_label(i):
            v = pill_views[i]
            ps = v["pills"]
            if "right" in v:
                flag = "✅ " if not v["missed"] and all(v["right"]) and len(ps) == len(v["gt"]) else "❌ "
            else:
                flag = "🔴 " if any(q["status"] == "ngoài đơn" for q in ps) else \
                       "🟠 " if any(q["status"] == "tin cậy thấp" for q in ps) else ""
            return f"{flag}Ảnh {i + 1} · {len(ps)} viên"

        vi = st.radio("Ảnh viên thuốc", list(range(len(pill_views))), format_func=view_label, horizontal=True,
                      key="pill_view")
        if vi is None or vi >= len(pill_views):
            vi = 0
        v = pill_views[vi]
        pills = v["pills"]
        has_gt = "gt" in v
        c1, c2 = st.columns([5, 6], gap="large")
        with c1:
            show_gt = has_gt and st.toggle("Hiện khung nhãn thật (xanh dương)", value=True, key="show_gt")
            st.image(draw_pills(v["res"]["image"], pills, info, v["gt"] if show_gt else None, v.get("missed", ())),
                     width="stretch", caption=v["name"])
            shown = [s for s in STATUS if s != "chưa đối chiếu" or no_context]
            show_html(" ".join(chip(STATUS[s][1], STATUS[s][2]) for s in shown)
                      + (" " + chip("Khung nhãn thật", GT_COLOR) if show_gt else ""))
        with c2:
            st.markdown("#### Kết quả từng viên")
            cnt = {s: sum(q["status"] == s for q in pills) for s in STATUS}
            show_html(chip(f"{len(pills)} viên", INFO) + " "
                      + " ".join(chip(f"{n} {STATUS[s][1].lower()}", STATUS[s][2]) for s, n in cnt.items() if n))
            if not pills:
                st.warning("Không tìm thấy viên thuốc nào trong ảnh. Hãy kiểm tra đây có đúng là ảnh viên thuốc không "
                           "(có thể đã tải nhầm ảnh đơn vào ô viên thuốc), hoặc chụp lại gần hơn, trên nền trơn.")
            rows = []
            for i, p in enumerate(pills, 1):
                name = info.name(p["best_cls"])
                if p["status"] == "ngoài đơn":
                    name = f"— (gần nhất trong đơn: {name})"
                elif p["status"] == "chưa đối chiếu":
                    name = f"? {info.name(p['global_cls'])}"
                ic, lab, _ = STATUS[p["status"]]
                row = {"Viên": f"#{i}", "Kết luận": f"{ic} {lab}"}
                if has_gt:
                    j = v["pred_gt"][i - 1]
                    if j is None:
                        row["So với nhãn"] = "➕ Khung thừa"
                    else:
                        g = v["gt"][j]["cls"]
                        truth = "viên ngoài đơn" if g == OUT_OF_PRES else info.name(g)
                        row["So với nhãn"] = ("✅ Đúng" if v["right"][i - 1] else "❌ Sai") + f" · nhãn: {truth}"
                row.update({"Thuốc nhận ra": name, "Độ tương đồng": round(p["score"], 3)})
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                    "Độ tương đồng": st.column_config.ProgressColumn(
                        "Độ tương đồng", min_value=0.0, max_value=1.0, format="%.2f",
                        help=f"Cosin giữa viên và thuốc gần nhất trong đơn. Ngưỡng chấp nhận τ = {pill_tau:.2f}.")})
            if has_gt:
                nr = sum(v["right"])
                msg = f"Ảnh này: {nr}/{len(v['gt'])} viên nhận dạng đúng"
                if v["missed"]:
                    msg += f" · {len(v['missed'])} viên bị bỏ sót (khung xanh dương ghi 'bỏ sót')"
                (st.success if nr == len(v["gt"]) and not v["missed"] and len(pills) == len(v["gt"]) else st.warning)(msg)
            if pills:
                with st.expander("Ảnh cắt từng viên"):
                    cols = st.columns(4)
                    for i, p in enumerate(pills):
                        with cols[i % 4]:
                            st.image(p["crop"], caption=f"#{i + 1} · {STATUS[p['status']][1]}", width="stretch")
                            st.caption("<br>".join(f"{esc(info.name(c))} [{c}]: {s:.2f}" for c, s in p["top3"]),
                                       unsafe_allow_html=True)
            with st.expander("Chi tiết kỹ thuật"):
                if pills:
                    st.dataframe(pd.DataFrame([{"Viên": f"#{i}", "Mã trong đơn": p["best_cls"], "Cosin": round(p["score"], 3),
                                                "Gần nhất (mọi mã)": f"{info.name(p['global_cls'])} [mã {p['global_cls']}] "
                                                                      f"{p['global_score']:.2f}",
                                                "YOLO conf": round(p["det_conf"], 2)} for i, p in enumerate(pills, 1)]),
                                 hide_index=True, width="stretch")
                st.caption(f"τ = {pill_tau:.2f}, vùng cần kiểm tra lại [{pill_tau:.2f}; {pill_tau + margin:.2f}) · thời gian: "
                           + " · ".join(f"{k} {t:.1f}s" for k, t in v["res"]["timing"].items()))

# ----------------------------------------------------------------------------- ③ tương tác
with tab3:
    if not pres:
        st.info("Chưa có ảnh đơn thuốc nên chưa biết các thuốc cần tra tương tác. Tải ảnh đơn ở phía trên.", icon="📄")
    elif len(groups) < 2:
        st.info("Cần ít nhất 2 thuốc đã nhận ra trong đơn để kiểm tra tương tác.")
    else:
        if known:
            show_html("".join(
                f"<div class='mc-ddi' style='border-left-color:{rgb(LEVEL[r['level']][1])};"
                f"background:{rgb(LEVEL[r['level']][1], .05)}'>{chip(LEVEL[r['level']][0], LEVEL[r['level']][1])}"
                f"<span class='p'>{esc(r['thuoc_1'])} × {esc(r['thuoc_2'])}</span>"
                f"<div class='i'>Hoạt chất: {esc(r['hoat_chat_1'])} – {esc(r['hoat_chat_2'])}</div></div>"
                for r in known))
        else:
            st.success("Không thấy cặp tương tác đã phân mức (nghiêm trọng / trung bình / nhẹ) trong DDInter.")
        st.caption("Mức độ (phân loại của DDInter): nghiêm trọng · trung bình · nhẹ. Cần đối chiếu thêm với tài liệu "
                   "chuyên môn trước khi quyết định điều chỉnh đơn.")
        with st.expander("Hoạt chất của từng thuốc trong đơn"):
            st.dataframe(pd.DataFrame([{"Thuốc": g, "Hoạt chất": "; ".join(sorted({info.ingredient(c) for c in cs})),
                                        "Tên trong DDInter": "; ".join(sorted({n for c in cs for n in info.ddinter(c)})) or "— (chưa có)"}
                                       for g, cs in groups]), hide_index=True, width="stretch")
            st.caption("Mỗi thuốc có thể ứng với nhiều mã viên. Nếu ảnh viên đã xác nhận mã nào thì chỉ giữ mã đó; "
                       "chưa có ảnh viên thì xét mọi mã ứng viên.")
        if unk:
            with st.expander(f"⚪ {len(unk)} cặp có ghi nhận trong DDInter nhưng chưa rõ mức độ"):
                st.dataframe(pd.DataFrame(unk)[["thuoc_1", "thuoc_2", "hoat_chat_1", "hoat_chat_2"]].rename(columns={
                    "thuoc_1": "Thuốc 1", "thuoc_2": "Thuốc 2", "hoat_chat_1": "Hoạt chất 1", "hoat_chat_2": "Hoạt chất 2"}),
                             hide_index=True, width="stretch")
        if nodata:
            st.caption("Chưa tra được: " + "; ".join(f"{r['thuoc_1']} × {r['thuoc_2']}" for r in nodata)
                       + " (thiếu hoạt chất trong bảng DDInter, thường là thuốc thảo dược).")
    st.caption("Nguồn: DDInter 2.0 (ddinter2.scbdd.com), bản tải về theo nhóm ATC A, B, D, H, L, P, R, V; chỉ giữ 39 hoạt chất "
               "có trong VAIPE nên có thể thiếu cặp. Bảng mã thuốc → hoạt chất do nhóm lập, một số dòng cần kiểm tra lại.")
