"""
prep.py — تحويل صورة نظارة من متجر إلى أصل جاهز للتركيب. بلا شبكة.

الخطوات:
  1) قصّ الخلفية  → BGRA
  2) اكتشاف نقاط الارتساء تلقائيًا (الجسر، المفصلتان، مركزا العدستين)
  3) توليد صورة الذراع
  4) كتابة المانيفست + صورة تحقق مؤشَّرة عليها النقاط

قرار موثّق: صورة منتج واحدة أمامية **لا** تحوي معلومات كافية عن شكل الذراع
(هي مطويّة أو مخفية خلف الواجهة). لذلك الذراع تُولَّد هندسيًا من لون وسماكة
الإطار عند المفصلة، وتُوسم `temple_is_synthetic=True`. البديل الأمين الوحيد هو
صورة جانبية للمنتج — يدعمها الحقل temple_png إن توفرت.
"""
from __future__ import annotations
from pathlib import Path
import argparse

import cv2
import numpy as np

from assetkit.schema import GlassesAsset, Anchors


# --------------------------------------------------------------------- القصّ
def cutout(bgr: np.ndarray, border_frac: float = 0.02) -> np.ndarray:
    """
    فصل النظارة عن الخلفية بلا أي نموذج مُدرَّب.

    منهج: عيّنة من إطار الصورة الخارجي تمثّل الخلفية (صور المنتجات دائمًا
    بخلفية موحّدة)، ثم عتبة على مسافة اللون، ثم GrabCut لتنقية الحواف —
    وهي المرحلة التي تلتقط الأذرع الرفيعة والعدسات نصف الشفافة.
    """
    h, w = bgr.shape[:2]
    b = max(2, int(min(h, w) * border_frac))
    ring = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                           bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    spread = float(np.median(np.abs(ring - bg)))

    dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    thr = max(18.0, 6.0 * spread)
    sure_fg = (dist > thr * 2.0)
    maybe = (dist > thr * 0.6)

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[maybe] = cv2.GC_PR_FGD
    mask[sure_fg] = cv2.GC_FGD
    mask[:b] = cv2.GC_BGD; mask[-b:] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD; mask[:, -b:] = cv2.GC_BGD

    if sure_fg.sum() > 50:
        try:
            bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, mask, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            pass

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # أكبر مكوّن متصل فقط — يتخلص من ظلال المنتج وشعارات المتجر
    n, lab, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha = np.where(lab == k, 255, 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.dstack([bgr, alpha])


def trim(bgra: np.ndarray, pad: int = 2) -> np.ndarray:
    a = bgra[:, :, 3]
    ys, xs = np.where(a > 8)
    if ys.size == 0:
        raise ValueError("القصّ أنتج صورة فارغة — تحقق من خلفية الصورة الأصلية")
    y1, y2 = max(0, ys.min() - pad), min(bgra.shape[0], ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(bgra.shape[1], xs.max() + pad + 1)
    return bgra[y1:y2, x1:x2].copy()


# ------------------------------------------------------------- نقاط الارتساء
def detect_anchors(bgra: np.ndarray) -> tuple[Anchors, dict]:
    """
    اشتقاق النقاط من شكل قناع الألفا نفسه — لا تخمين يدوي.

    المنطق الهندسي:
      • الجسر  = العمود الذي يقل فيه ارتفاع الكتلة عند منتصف الصورة أفقيًا.
      • المفصلة = أقصى نقطة على كل جانب، عند الارتفاع الرأسي لمركز العدسة.
      • مركز العدسة = مركز ثقل أكبر ثقب داخلي في القناع (فتحة الإطار).
    """
    a = (bgra[:, :, 3] > 8).astype(np.uint8)
    h, w = a.shape
    colsum = a.sum(axis=0).astype(np.float32)
    nz = np.where(colsum > 0)[0]
    x_lo, x_hi = int(nz[0]), int(nz[-1])
    width = x_hi - x_lo + 1

    # --- الجسر: منطقة أنحف الأعمدة ضمن الثلث الأوسط.
    # نأخذ مركز كل الأعمدة القريبة من الحد الأدنى (لا أول حد أدنى)، لأن الجسر
    # شريط ممتد لا عمود واحد، و argmin وحده ينحاز لطرفه.
    lo, hi = x_lo + int(width * .35), x_lo + int(width * .65)
    band = np.where(colsum[lo:hi + 1] > 0, colsum[lo:hi + 1], np.inf)
    lowest = band.min()
    near = np.where(band <= lowest * 1.25)[0]
    bx = int(round(lo + near.mean()))
    ys = np.where(a[:, bx] > 0)[0]
    by = float(ys.mean()) if ys.size else h / 2

    # --- مركزا العدستين = الثقوب المغلقة داخل الظل (fill من الحدود الخارجية)
    inv = (a == 0).astype(np.uint8) * 255
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((ff > 0).astype(np.uint8), 8)
    cands = sorted([(stats[i, cv2.CC_STAT_AREA], cent[i]) for i in range(1, n)],
                   key=lambda t: -t[0])[:2]
    if len(cands) == 2:
        c = sorted([tuple(map(float, p)) for _, p in cands], key=lambda p: p[0])
        lens_l, lens_r = c[0], c[1]
        eye_y = (lens_l[1] + lens_r[1]) / 2.0
    else:
        lens_l = lens_r = None
        eye_y = by

    # --- المفصلتان = أقصى نقطتين على يمين ويسار الظل الكامل.
    # الذراع تتصل دائمًا بأعرض نقطة في الواجهة، فالطرف العام أمتن من شريط
    # رأسي قد يفوّت نتوء المفصلة إن كان أعلى أو أسفل مستوى العدسة.
    hx_l, hx_r = x_lo, x_hi
    hy_l = float(np.mean(np.where(a[:, hx_l] > 0)[0]))
    hy_r = float(np.mean(np.where(a[:, hx_r] > 0)[0]))

    anchors = Anchors(bridge_center=(float(bx), by),
                      hinge_left=(float(hx_l), hy_l),
                      hinge_right=(float(hx_r), hy_r),
                      lens_center_left=lens_l, lens_center_right=lens_r)
    meas = dict(cutout_px=[int(w), int(h)], front_span_px=int(width),
                lens_holes_found=int(len(cands)),
                hinge_span_px=int(hx_r - hx_l))
    return anchors, meas


# ------------------------------------------------------------------ الذراع
def synth_temple(bgra: np.ndarray, anchors: Anchors, length_px: int) -> np.ndarray:
    """
    ذراع مولّدة: شريط مستدقّ بلون وسماكة مأخوذين من الإطار عند المفصلة نفسها،
    مع انحناء طرفي بسيط (الجزء الذي يلتف خلف الأذن).
    """
    hx, hy = anchors.hinge_right
    x0 = int(np.clip(hx - 6, 0, bgra.shape[1] - 1))
    y0 = int(np.clip(hy, 0, bgra.shape[0] - 1))
    patch = bgra[max(0, y0 - 4):y0 + 5, max(0, x0 - 4):x0 + 5]
    m = patch[:, :, 3] > 8
    color = patch[:, :, :3][m].mean(axis=0) if m.any() else np.array([40, 40, 40])

    col = a_col = bgra[:, :, 3] > 8
    ys = np.where(a_col[:, x0])[0]
    thick = max(3, int(ys.size)) if ys.size else 5

    L, T = int(length_px), thick * 3
    img = np.zeros((T, L, 4), np.uint8)
    cy = T // 2
    for x in range(L):
        t = x / max(1, L - 1)
        half = max(1.0, thick / 2.0 * (1.0 - 0.35 * t))       # استدقاق للخلف
        drop = (max(0.0, t - 0.78) / 0.22) ** 2 * thick * 2.2  # التفاف خلف الأذن
        c = cy + drop
        y1, y2 = int(round(c - half)), int(round(c + half)) + 1
        y1, y2 = max(0, y1), min(T, y2)
        img[y1:y2, x, :3] = color
        img[y1:y2, x, 3] = 255
    img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (3, 3), 0)
    return img


def derive_lens_centers(front: np.ndarray, anchors: Anchors, lens_mm: float,
                        bridge_mm: float, mm_per_px: float) -> tuple[Anchors, dict]:
    """
    اشتقاق مركزي العدستين حين يفشل كشف الثقوب — حالة النظارة الشمسية.

    العدسة الداكنة المعتمة تُصنَّف ضمن الظل نفسه، فلا يبقى ثقب مغلق يُقاس منه
    المركز. لكن الترقيم المطبوع يعطي المسافة بين المركزين مباشرة:

        المسافة بين مركزي العدستين = عرض العدسة + الجسر

    فيُشتقّ الفصل الأفقي من هذا الرقم (لا تخمين)، ويبقى الموضع الرأسي **مقيسًا
    من القناع**: متوسط ارتفاع الكتلة عند عمود كل عدسة.

    تنبيه أمانة: بعد هذا الاشتقاق يصير فحص schema.validate المتقاطع
    (المسافة مقابل lens+bridge) تحصيل حاصل ولا يعود دليلًا مستقلًا؛ لذلك
    يُوسم الأصل بـ lens_centers_source='derived_from_numbering'.
    """
    a = (front[:, :, 3] > 8).astype(np.uint8)
    h, w = a.shape
    sep_px = (lens_mm + bridge_mm) / mm_per_px
    bx = anchors.bridge_center[0]

    def col_center_y(x: float) -> float:
        xi = int(np.clip(round(x), 0, w - 1))
        ys = np.where(a[:, xi] > 0)[0]
        if ys.size == 0:                       # عمود فارغ → ارجع لمستوى الجسر
            return anchors.bridge_center[1]
        return float(ys.mean())

    xl, xr = bx - sep_px / 2.0, bx + sep_px / 2.0
    left = (float(np.clip(xl, 0, w - 1)), col_center_y(xl))
    right = (float(np.clip(xr, 0, w - 1)), col_center_y(xr))

    new = Anchors(bridge_center=anchors.bridge_center,
                  hinge_left=anchors.hinge_left, hinge_right=anchors.hinge_right,
                  lens_center_left=left, lens_center_right=right)
    info = {"lens_centers_source": "derived_from_numbering",
            "lens_sep_px": round(float(sep_px), 1)}
    return new, info


# -------------------------------------------------------------------- المسار
def prepare(src: Path, out_dir: Path, asset_id: str, name: str,
            lens_mm: float, bridge_mm: float, temple_mm: float,
            lens_h_mm: float, frame_w_mm: float | None,
            source_note: str) -> GlassesAsset:
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(src)

    front = trim(cutout(bgr))
    anchors, meas = detect_anchors(front)

    if frame_w_mm is None:
        # العرض الكلي غير معطى → يُشتق من الترقيم المطبوع على الإطار
        frame_w_mm = 2 * lens_mm + bridge_mm
        meas["frame_width_source"] = "derived_from_lens_bridge"
    else:
        meas["frame_width_source"] = "given"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    front_name = f"{asset_id}_front.png"
    cv2.imwrite(str(out_dir / front_name), front)

    mm_per_px = frame_w_mm / front.shape[1]

    # عدسات داكنة: لا ثقوب مغلقة → يُشتق المركزان بدل الارتداد إلى الجسر،
    # لأن الارتساء على الجسر يُجلس العدسات أخفض من البؤبؤين (انظر README §4).
    if anchors.lens_center_left is None or anchors.lens_center_right is None:
        anchors, info = derive_lens_centers(front, anchors, lens_mm, bridge_mm,
                                            mm_per_px)
        meas.update(info)
    else:
        meas["lens_centers_source"] = "measured_from_holes"

    temple = synth_temple(front, anchors, length_px=int(temple_mm / mm_per_px))
    temple_name = f"{asset_id}_temple.png"
    cv2.imwrite(str(out_dir / temple_name), temple)

    asset = GlassesAsset(
        id=asset_id, name=name,
        frame_width_mm=float(frame_w_mm), lens_width_mm=float(lens_mm),
        bridge_mm=float(bridge_mm), temple_length_mm=float(temple_mm),
        lens_height_mm=float(lens_h_mm),
        anchors=anchors, front_png=front_name, temple_png=temple_name,
        temple_is_synthetic=True, source_note=source_note, measured=meas)

    errs = asset.validate(out_dir)
    if errs:
        raise ValueError("الأصل لم يجتز الفحص:\n  - " + "\n  - ".join(errs))
    asset.save(out_dir)
    _debug_overlay(front, anchors, out_dir / f"{asset_id}_anchors.png")
    return asset


def _debug_overlay(front: np.ndarray, a: Anchors, path: Path):
    """صورة تحقق بصري — تُراجَع بالعين قبل اعتماد الأصل."""
    vis = front[:, :, :3].copy()
    vis[front[:, :, 3] < 8] = (255, 255, 255)
    for pt, col, tag in ((a.bridge_center, (0, 0, 255), "B"),
                         (a.hinge_left, (255, 0, 0), "HL"),
                         (a.hinge_right, (255, 0, 0), "HR"),
                         (a.lens_center_left, (0, 160, 0), "L"),
                         (a.lens_center_right, (0, 160, 0), "R")):
        if pt is None:
            continue
        p = (int(pt[0]), int(pt[1]))
        cv2.drawMarker(vis, p, col, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, tag, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description="تجهيز أصل نظارة من صورة منتج")
    ap.add_argument("image")
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lens", type=float, required=True, help="عرض العدسة مم")
    ap.add_argument("--bridge", type=float, required=True, help="الجسر مم")
    ap.add_argument("--temple", type=float, required=True, help="طول الذراع مم")
    ap.add_argument("--lens-height", type=float, default=40.0)
    ap.add_argument("--frame-width", type=float, default=None)
    ap.add_argument("--out", default="catalog")
    ap.add_argument("--note", default="")
    ns = ap.parse_args()
    a = prepare(Path(ns.image), Path(ns.out), ns.id, ns.name, ns.lens, ns.bridge,
                ns.temple, ns.lens_height, ns.frame_width, ns.note)
    print(f"تم: {a.id}  عرض={a.frame_width_mm}mm  قياسات={a.measured}")


if __name__ == "__main__":
    main()
