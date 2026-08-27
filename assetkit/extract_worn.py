"""
extract_worn.py — استخراج إطار نظارة من صورة شخص يلبسها.

هذا المسار عكس مسار التركيب تمامًا. في التركيب نأخذ صورة مسطّحة ونضعها على
مستوى ثلاثي الأبعاد في فضاء الوجه ثم نُسقطها. هنا نعرف وضعية الرأس، فنعكس
ذلك الإسقاط: نُرجع منطقة النظارة إلى مستوى أمامي مواجه للكاميرا (rectify)،
وعندها تصير الأبعاد داخل الصورة المستوية متناسبة تناسبًا مباشرًا مع
المليمترات — أي أن الصورة المرتدَاة تعطينا **المقياس الفيزيائي مجانًا**،
وهو بالضبط ما لا تعطيه صورة المنتج على خلفية بيضاء.

المفاضلة بين المصدرين:
  صورة منتج : بكسلات نظيفة، حواف حادة، بلا حجب — لكن بلا أي مقياس.
  صورة مرتدَاة: مقياس مقيس — لكن بكسلات ملوّثة بالبشرة والشعر والانعكاسات،
                وجانب واحد مختصر بالمنظور.

حدود موثّقة صراحةً:
  1) المقياس نسبي لوجه اللابس. بلا قزحية مرئية (والعدسات داكنة عادة) نضطر
     لافتراض وجه بحجم النموذج القانوني، فيدخل خطأ ≈±5% من تباين حجم الوجوه.
  2) الجانب البعيد عن الكاميرا مختصر ومعلوماته أقل. نستفيد من كون النظارات
     متماثلة ثنائيًا فنعكس الجانب الأوضح — وهذا **افتراض**، لا قياس، ويُوسَم
     في المانيفست بـ symmetrized=True.
  3) الأذرع لا تُستخرج من صورة أمامية. الصور الجانبية قد تحويها، وهي عمل لاحق.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse

import cv2
import numpy as np

import config
from core import geometry as G
from core.face import FaceAnalyzer, FaceGeometry
from assetkit import prep
from assetkit.schema import GlassesAsset

# نافذة الاستخراج في فضاء الوجه (مم) — أوسع من أي إطار حقيقي بهامش
WIN_W_MM = 170.0
WIN_H_MM = 70.0
OUT_W = 1000                       # دقة الصورة المستوية
PX_PER_MM = OUT_W / WIN_W_MM
OUT_H = int(WIN_H_MM * PX_PER_MM)


@dataclass
class Extraction:
    bgra: np.ndarray
    frame_width_mm: float
    lens_width_mm: float
    bridge_mm: float
    lens_height_mm: float
    symmetrized: bool
    yaw_deg: float
    notes: list[str]


def _window_corners_3d() -> np.ndarray:
    """زوايا نافذة الاستخراج على مستوى النظارة في فضاء الوجه."""
    from core.fitting import _optical_anchor_3d, _front_basis
    o = _optical_anchor_3d(1.0)
    right, up = _front_basis()
    hw, hh = WIN_W_MM / 2.0, WIN_H_MM / 2.0
    return np.array([o - right * hw + up * hh,      # TL
                     o + right * hw + up * hh,      # TR
                     o + right * hw - up * hh,      # BR
                     o - right * hw - up * hh])     # BL


def rectify(frame_bgr: np.ndarray, fg: FaceGeometry) -> np.ndarray:
    """يُرجع منطقة النظارة إلى مستوٍ أمامي بمقياس معلوم (PX_PER_MM)."""
    src = fg.pose.project(_window_corners_3d()).astype(np.float32)
    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)


def _skin_reference(frame_bgr: np.ndarray, fg: FaceGeometry) -> np.ndarray:
    """عيّنة لون بشرة من الوجنتين والجبهة — بعيدًا عن النظارة."""
    ids = [G.IDX['forehead'], 205, 425, G.IDX['philtrum'], 50, 280]
    cols = []
    h, w = frame_bgr.shape[:2]
    for i in ids:
        if i >= len(fg.landmarks):
            continue
        x, y = fg.landmarks[i].astype(int)
        if 3 <= x < w - 3 and 3 <= y < h - 3:
            cols.append(frame_bgr[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3))
    if not cols:
        return np.array([[150, 150, 150]], np.float32)
    return np.concatenate(cols).astype(np.float32)


def segment(rect_bgr: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    فصل الإطار عن البشرة في الصورة المستوية.

    البذرة هندسية لا لونية فقط: الإطار يقع في شريط أفقي حول منتصف النافذة
    (لأننا ثبّتنا النافذة على المركز البصري)، والبشرة تحيط به من فوق ومن تحت.
    نُغذّي GrabCut بهذه المعرفة بدل تركه يخمّن.
    """
    h, w = rect_bgr.shape[:2]
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(skin.reshape(-1, 1, 3).astype(np.uint8),
                            cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    mu = skin_lab.mean(axis=0)
    sd = np.maximum(skin_lab.std(axis=0), 6.0)
    d = np.sqrt((((lab - mu) / sd) ** 2).sum(axis=2))     # مسافة ماهالانوبيس مبسّطة

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    band = slice(int(h * .18), int(h * .82))
    mask[band] = cv2.GC_PR_FGD
    mask[:int(h * .10)] = cv2.GC_BGD          # الحاجبان/الجبهة
    mask[int(h * .92):] = cv2.GC_BGD          # الوجنتان
    mask[(d < 1.6)] = cv2.GC_PR_BGD           # يشبه البشرة بقوة
    strong = np.zeros_like(mask, bool)
    strong[band] = d[band] > 4.0
    mask[strong] = cv2.GC_FGD

    if strong.sum() < 200:
        return np.zeros((h, w), np.uint8)
    try:
        bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(rect_bgr, mask, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass

    a = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # نحتفظ بأفضل مكوّن، وأيضًا بمكوّن ثانٍ كبير على الجهة المقابلة.
    # السبب: جسر الأنف رفيع وأحيانًا أفتح من البشرة فيُفقد، فتنفصل العدستان
    # إلى مكوّنين. الاكتفاء بواحد كان يحذف نصف الإطار.
    n, lab_c, stats, _ = cv2.connectedComponentsWithStats(a, 8)
    scored = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.004 * h * w:
            continue
        crosses = x < w * .5 < x + bw
        scored.append((area * (2.0 if crosses else 1.0) * (bw / w), i,
                       x + bw / 2.0))
    if not scored:
        return np.zeros((h, w), np.uint8)
    scored.sort(reverse=True)
    keep = [scored[0][1]]
    cx0 = scored[0][2]
    for sc, i, cx in scored[1:]:
        if sc > 0.25 * scored[0][0] and (cx - w / 2) * (cx0 - w / 2) < 0:
            keep.append(i)
            break
    return np.where(np.isin(lab_c, keep), 255, 0).astype(np.uint8)


def symmetrize(alpha: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, bool]:
    """
    عند الدوران، الجانب الأقرب للكاميرا أوضح. نعكسه على الآخر.
    نظارات المستهلك متماثلة ثنائيًا، فهذا افتراض معقول لكنه يبقى افتراضًا.
    """
    if abs(yaw_deg) < 8:
        return alpha, False
    w = alpha.shape[1]
    mid = w // 2
    keep_left = yaw_deg < 0            # الوجه ملتفت لليمين → يسار الصورة أقرب
    half = alpha[:, :mid] if keep_left else alpha[:, mid:]
    mirrored = cv2.flip(half, 1)
    out = alpha.copy()
    if keep_left:
        out[:, mid:mid + mirrored.shape[1]] = mirrored
    else:
        out[:, mid - mirrored.shape[1]:mid] = mirrored
    return out, True


def measure_specs(alpha: np.ndarray) -> dict:
    """قياس مقاسات الإطار من الظل المستوي — كل شيء بالمليمتر مباشرة."""
    cols = np.where(alpha.max(axis=0) > 8)[0]
    rows = np.where(alpha.max(axis=1) > 8)[0]
    if cols.size < 2 or rows.size < 2:
        return {}
    total_mm = (cols[-1] - cols[0] + 1) / PX_PER_MM
    height_mm = (rows[-1] - rows[0] + 1) / PX_PER_MM

    # العدستان = أكبر ثقبين مغلقين، أو تقدير من الفجوة الوسطى إن كانت مصمتة
    inv = (alpha == 0).astype(np.uint8) * 255
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((alpha.shape[0] + 2, alpha.shape[1] + 2), np.uint8),
                  (0, 0), 0)
    n, lb, st, ct = cv2.connectedComponentsWithStats((ff > 0).astype(np.uint8), 8)
    holes = sorted([(st[i, cv2.CC_STAT_AREA], st[i]) for i in range(1, n)],
                   key=lambda t: -t[0])[:2]
    if len(holes) == 2:
        a1, a2 = sorted([h[1] for h in holes], key=lambda s: s[cv2.CC_STAT_LEFT])
        lens_mm = ((a1[cv2.CC_STAT_WIDTH] + a2[cv2.CC_STAT_WIDTH]) / 2) / PX_PER_MM
        bridge_mm = (a2[cv2.CC_STAT_LEFT] -
                     (a1[cv2.CC_STAT_LEFT] + a1[cv2.CC_STAT_WIDTH])) / PX_PER_MM
        lens_h_mm = ((a1[cv2.CC_STAT_HEIGHT] + a2[cv2.CC_STAT_HEIGHT]) / 2) / PX_PER_MM
        src = "holes"
    else:
        # عدسات معتمة ملتحمة بالإطار: نقيس الجسر من انخفاض ارتفاع الأعمدة
        # في المنطقة الوسطى — الجسر أنحف بكثير من العدسة رأسيًا.
        colh = (alpha > 8).sum(axis=0).astype(np.float32)
        seg = colh[cols[0]:cols[-1] + 1]
        peak = float(np.percentile(seg, 90))
        mid = len(seg) // 2
        lo = mid
        while lo > 0 and seg[lo] < 0.6 * peak:
            lo -= 1
        hi = mid
        while hi < len(seg) - 1 and seg[hi] < 0.6 * peak:
            hi += 1
        bridge_px = max(1, hi - lo)
        if bridge_px > 0.35 * len(seg):
            raise ValueError("تعذّر تمييز الجسر — الفصل غالبًا ملوّث")
        bridge_mm = bridge_px / PX_PER_MM
        lens_mm = (total_mm - bridge_mm) / 2
        lens_h_mm = height_mm * 0.85
        src = "bridge_from_profile"
    return dict(frame_width_mm=float(total_mm), lens_width_mm=float(lens_mm),
                bridge_mm=float(max(8.0, bridge_mm)),
                lens_height_mm=float(lens_h_mm), spec_source=src)


def extract(image_path: Path, analyzer: FaceAnalyzer) -> Extraction:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    if max(img.shape[:2]) < 700:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    fg = analyzer.analyze(img)
    if fg is None:
        raise ValueError("لم يُكتشف وجه")

    notes: list[str] = []
    if abs(fg.pose.yaw) > 35:
        notes.append(f"دوران {fg.pose.yaw:.0f}° كبير — جودة الجانب البعيد رديئة")

    rect = rectify(img, fg)
    alpha = segment(rect, _skin_reference(img, fg))
    if alpha.max() == 0:
        raise ValueError("فشل الفصل: لم يُعثر على إطار متمايز عن البشرة")

    cov_w = (alpha.max(axis=0) > 8).sum() / alpha.shape[1]
    cov_h = (alpha.max(axis=1) > 8).sum() / alpha.shape[0]
    fill = (alpha > 8).mean()
    if cov_w > 0.96:
        raise ValueError(f"تسرّب أفقي: الظل يملأ {cov_w:.0%} من نافذة {WIN_W_MM:.0f}mm")
    if cov_h > 0.88:
        raise ValueError(f"تسرّب رأسي: الظل يملأ {cov_h:.0%} من ارتفاع النافذة")
    if fill > 0.55:
        raise ValueError(f"الظل مصمت ({fill:.0%}) — بشرة أو شعر مُدمج بالإطار")

    alpha, sym = symmetrize(alpha, fg.pose.yaw)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    bgra = np.dstack([rect, alpha])
    bgra = prep.trim(bgra)

    specs = measure_specs(alpha)
    if not specs:
        raise ValueError("تعذّر قياس المقاسات")
    # حدود المقاسات الحقيقية للنظارات الشمسية — رقم خارجها يعني فشل قياس لا إطارًا غريبًا
    if not (110 <= specs["frame_width_mm"] <= 165):
        raise ValueError(f"عرض {specs['frame_width_mm']:.0f}mm خارج المدى الواقعي")
    if not (35 <= specs["lens_width_mm"] <= 75):
        raise ValueError(f"عرض عدسة {specs['lens_width_mm']:.0f}mm غير معقول")
    if not (10 <= specs["bridge_mm"] <= 35):
        raise ValueError(f"جسر {specs['bridge_mm']:.0f}mm غير معقول")
    if specs["spec_source"] == "estimated_no_holes":
        notes.append("العدسات معتمة وملتحمة بالإطار — تقسيم العدسة/الجسر مقدَّر لا مقيس")
    notes.append("المقياس نسبي لوجه اللابس بافتراض حجم قانوني (خطأ ≈±5%)")

    return Extraction(bgra=bgra, symmetrized=sym, yaw_deg=fg.pose.yaw,
                      notes=notes, **{k: v for k, v in specs.items()
                                      if k != "spec_source"})


def to_asset(ex: Extraction, asset_id: str, name: str,
             out_dir: Path, source: str) -> GlassesAsset:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    front_name = f"{asset_id}_front.png"
    cv2.imwrite(str(out_dir / front_name), ex.bgra)

    anchors, meas = prep.detect_anchors(ex.bgra)
    meas.update(dict(extracted_from_worn_photo=True, symmetrized=ex.symmetrized,
                     yaw_deg=round(ex.yaw_deg, 1), notes=ex.notes))

    mm_per_px = ex.frame_width_mm / ex.bgra.shape[1]
    temple_mm = 140.0            # [تخمين] لا تظهر الذراع في صورة أمامية
    temple = prep.synth_temple(ex.bgra, anchors, int(temple_mm / mm_per_px))
    temple_name = f"{asset_id}_temple.png"
    cv2.imwrite(str(out_dir / temple_name), temple)

    a = GlassesAsset(id=asset_id, name=name,
                     frame_width_mm=round(ex.frame_width_mm, 1),
                     lens_width_mm=round(ex.lens_width_mm, 1),
                     bridge_mm=round(ex.bridge_mm, 1),
                     temple_length_mm=temple_mm,
                     lens_height_mm=round(ex.lens_height_mm, 1),
                     anchors=anchors, front_png=front_name,
                     temple_png=temple_name, temple_is_synthetic=True,
                     source_note=source, measured=meas)
    a.save(out_dir)
    prep._debug_overlay(ex.bgra, anchors, out_dir / f"{asset_id}_anchors.png")
    return a


def main():
    ap = argparse.ArgumentParser(description="استخراج إطار من صورة شخص يلبسه")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="catalog")
    ap.add_argument("--prefix", default="w")
    ns = ap.parse_args()

    fa = FaceAnalyzer(static=True)
    ok = fail = 0
    for i, p in enumerate(ns.images):
        p = Path(p)
        try:
            ex = extract(p, fa)
            aid = f"{ns.prefix}{i:02d}"
            a = to_asset(ex, aid, p.stem[:28], Path(ns.out), f"مستخرج من {p.name}")
            errs = a.validate(Path(ns.out))
            flag = "  ⚠ " + "; ".join(errs) if errs else ""
            print(f"[OK]   {p.name[:44]:<46} {a.frame_width_mm:.0f}mm "
                  f"{a.lens_width_mm:.0f}□{a.bridge_mm:.0f} yaw={ex.yaw_deg:+.0f}°{flag}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {p.name[:44]:<46} {e}")
            fail += 1
    fa.close()
    print(f"\nنجح {ok} / فشل {fail}")


if __name__ == "__main__":
    main()
