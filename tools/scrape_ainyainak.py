"""
scrape_ainyainak.py — جمع نظارات شمسية من متجر عيني عينك (Salla) وتحويلها
إلى أصول كتالوج عبر assetkit.prep.

لماذا هذا المتجر مصدر أفضل من مسار extract_worn:
  • ينشر المقاسات الثلاثة بالمليمتر لكل منتج (قطر العدسة / Bridge / طول الذراع)
    — وهي الأرقام التي يطلبها prep.py، ولا تتوفر أبدًا من صورة منتج وحدها.
  • صور المنتج على خلفية بيضاء موحّدة: بكسلات نظيفة وحواف حادة، لا تلوّث
    ببشرة وشعر وانعكاسات كما في الصور المرتدَاة.

التحدي الوحيد: معظم صور المنتج بزاوية 3/4 لا أمامية، و prep.py يفترض صورة
أمامية (يشتق الجسر من أنحف الأعمدة والمفصلتين من الطرفين). لذلك يُنتقى الإطار
الأمامي آليًا بمقياس **تماثل مرآتي** لقناع الألفا: الواجهة الأمامية متماثلة
حول محورها الرأسي، أما زاوية 3/4 فتُظهر ذراعًا واحدة فتكسر التماثل.

    python -m tools.scrape_ainyainak fetch  --limit 40 --out work/ainy
    python -m tools.scrape_ainyainak build  --work work/ainy --out catalog
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
import time
import urllib.request

import cv2
import numpy as np

BASE = "https://ainyainak.sa"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

IMG_RX = re.compile(r"https://cdn\.salla\.sa/GQOBV/[A-Za-z0-9]+\.(?:jpg|png|webp)")

# المقاسات كما ينشرها المتجر داخل وصف المنتج
RX_LENS = re.compile(r"قطر\s+العدسة\s*[:：]\s*(\d{2,3})")
RX_BRIDGE = re.compile(r"(?:Bridge|الجسر)\s*[:：]\s*(\d{2,3})")
RX_TEMPLE = re.compile(r"(?:طول\s+الذراع|Temple)\s*[:：]\s*(\d{2,3})")


def get2(url: str, tries: int = 3) -> tuple[bytes, str]:
    """يُرجع (المحتوى، الرابط النهائي بعد التحويلات)."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read(), r.geturl()
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"تعذّر الجلب {url}: {last}")


def get(url: str, tries: int = 3) -> bytes:
    return get2(url, tries)[0]


# ------------------------------------------------------------------ الفهرسة
def product_ids(kind: str = "شمسية") -> list[str]:
    xml = get(f"{BASE}/sitemap-2.xml").decode("utf-8", "replace")
    urls = re.findall(r"https://ainyainak\.sa/[^<]+/p\d+", xml)
    return [u.rsplit("/", 1)[1] for u in urls if kind in u]


def store_common_images(ids: list[str]) -> set[str]:
    """الصور التي تتكرر على كل صفحة = شعار وبانرات المتجر، تُطرح من المعرض."""
    sets = []
    for pid in ids[:3]:
        html = get(f"{BASE}/x/{pid}").decode("utf-8", "replace")
        sets.append(set(IMG_RX.findall(html)))
    return set.intersection(*sets) if sets else set()


def parse_product(pid: str, common: set[str]) -> dict | None:
    html = get(f"{BASE}/x/{pid}").decode("utf-8", "replace")

    lens = RX_LENS.search(html)
    bridge = RX_BRIDGE.search(html)
    temple = RX_TEMPLE.search(html)
    if not (lens and bridge and temple):
        return None                     # بلا مقاسات معلنة → لا يُبنى أصل منه

    name = re.search(r'"@type":"Product","name":"([^"]{3,120})"', html)
    brand = re.search(r'"brand":\{"@type":"Brand","name":"([^"]{2,60})"', html)

    imgs = [u for u in dict.fromkeys(IMG_RX.findall(html)) if u not in common]
    if not imgs:
        return None

    return {
        "id": pid,
        "name": (name.group(1) if name else pid).strip(),
        "brand": brand.group(1) if brand else "",
        "lens_mm": float(lens.group(1)),
        "bridge_mm": float(bridge.group(1)),
        "temple_mm": float(temple.group(1)),
        "images": imgs,
        "url": f"{BASE}/x/{pid}",
    }


# ----------------------------------------------- انتقاء الإطار الأمامي آليًا
def _alpha_mask(bgr: np.ndarray) -> np.ndarray | None:
    """قناع سريع بعتبة على مسافة اللون عن خلفية محيط الصورة (بلا GrabCut)."""
    h, w = bgr.shape[:2]
    b = max(2, int(min(h, w) * 0.02))
    ring = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                           bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    spread = float(np.median(np.abs(ring - bg)))
    if spread > 12:
        return None                     # الخلفية غير موحّدة → صورة لايف ستايل
    dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
    m = (dist > max(18.0, 6.0 * spread)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def frontality(bgr: np.ndarray) -> tuple[float, dict]:
    """
    درجة "أماميّة" الصورة = IoU بين قناع النظارة وصورته المرآتية.

    الواجهة الأمامية متماثلة حول محورها الرأسي فتعطي IoU عاليًا؛ أما زاوية
    3/4 فتُظهر ذراعًا ممتدة على جانب واحد فينهار التماثل.
    """
    m = _alpha_mask(bgr)
    if m is None:
        return 0.0, {"reason": "خلفية غير موحّدة"}
    ys, xs = np.where(m > 0)
    if ys.size < 200:
        return 0.0, {"reason": "قناع صغير جدًا"}
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    flip = crop[:, ::-1]
    inter = np.logical_and(crop, flip).sum()
    union = np.logical_or(crop, flip).sum()
    iou = float(inter / union) if union else 0.0

    h, w = crop.shape
    aspect = w / max(1, h)
    # الواجهة الأمامية عريضة ومنخفضة؛ ونطلب ثقبَي عدسة مغلقين
    inv = (crop == 0).astype(np.uint8) * 255
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)
    nh, _, hst, _ = cv2.connectedComponentsWithStats((ff > 0).astype(np.uint8), 8)
    holes = int(sum(1 for i in range(1, nh)
                    if hst[i, cv2.CC_STAT_AREA] > 0.02 * crop.size))
    return iou, {"iou": round(iou, 3), "aspect": round(aspect, 2), "holes": holes}


def pick_front(image_urls: list[str], cache: Path) -> tuple[Path | None, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    best, best_score, report = None, -1.0, []
    for i, u in enumerate(image_urls):
        p = cache / f"{i:02d}{Path(u).suffix}"
        try:
            if not p.exists():
                p.write_bytes(get(u))
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
        except Exception:                                        # noqa: BLE001
            continue
        iou, info = frontality(bgr)
        # الثقبان المغلقان دليل قاطع على رؤية العدستين من الأمام
        score = iou + (0.15 if info.get("holes", 0) == 2 else 0.0)
        info["score"] = round(score, 3)
        report.append({"url": u, **info})
        if score > best_score:
            best, best_score = p, score
    return (best if best_score >= 0.80 else None), {"best_score": round(best_score, 3),
                                                    "candidates": report}


# -------------------------------------------------------------------- أوامر
def cmd_fetch(ns):
    out = Path(ns.out); out.mkdir(parents=True, exist_ok=True)
    ids = product_ids()
    print(f"وُجد {len(ids)} منتج شمسي في خريطة الموقع")
    common = store_common_images(ids)
    print(f"صور المتجر العامة المستبعدة: {len(common)}")

    rows, skipped = [], {"بلا مقاسات": 0, "بلا صورة أمامية": 0, "خطأ": 0}
    for pid in ids[:ns.limit]:
        time.sleep(ns.delay)            # تهذيب تجاه خادم المتجر
        try:
            p = parse_product(pid, common)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {pid}: خطأ {e}")
            skipped["خطأ"] += 1
            continue
        if p is None:
            skipped["بلا مقاسات"] += 1
            continue
        front, rep = pick_front(p["images"], out / "img" / pid)
        if front is None:
            print(f"  {pid}: رُفض — أعلى تماثل {rep['best_score']}")
            skipped["بلا صورة أمامية"] += 1
            continue
        p["front_path"] = str(front)
        p["front_report"] = rep
        rows.append(p)
        print(f"  ✓ {pid} {p['name'][:40]:<42} "
              f"{p['lens_mm']:.0f}□{p['bridge_mm']:.0f} {p['temple_mm']:.0f} "
              f"(تماثل {rep['best_score']})")

    (out / "products.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nنجح: {len(rows)} | مستبعد: {skipped}")
    print(f"كُتب {out / 'products.json'}")


def cmd_build(ns):
    from assetkit.prep import prepare

    work = Path(ns.work)
    rows = json.loads((work / "products.json").read_text(encoding="utf-8"))
    ok, fail = 0, []
    for p in rows:
        aid = "ay_" + p["id"].lstrip("p")[:8]
        try:
            a = prepare(Path(p["front_path"]), Path(ns.out), aid,
                        p["name"][:40], p["lens_mm"], p["bridge_mm"],
                        p["temple_mm"], ns.lens_height, None,
                        f"عيني عينك {p['url']} — مقاسات معلنة من المتجر")
            print(f"  ✓ {aid}  عرض={a.frame_width_mm:.0f}mm  {a.measured}")
            ok += 1
        except Exception as e:                                   # noqa: BLE001
            print(f"  ✗ {aid}: {e}")
            fail.append((aid, str(e)))
    print(f"\nبُني {ok} أصل | فشل {len(fail)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch"); f.set_defaults(fn=cmd_fetch)
    f.add_argument("--limit", type=int, default=40)
    f.add_argument("--out", default="work/ainy")
    f.add_argument("--delay", type=float, default=0.7)

    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--work", default="work/ainy")
    b.add_argument("--out", default="catalog")
    b.add_argument("--lens-height", type=float, default=40.0)

    ns = ap.parse_args()
    ns.fn(ns)


if __name__ == "__main__":
    main()
