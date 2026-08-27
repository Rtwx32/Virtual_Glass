"""
app.py — واجهة التشغيل: وضع الصورة الواحدة ووضع الكاميرا الحية.

    python app.py photo --image me.jpg --asset demo_rect --out result.png
    python app.py live  --asset demo_rect
    python app.py list
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

import config
from core.face import FaceAnalyzer
from core import compositor, measure
from assetkit.schema import GlassesAsset


def load_asset(asset_id: str) -> GlassesAsset:
    p = config.CATALOG / f"{asset_id}.json"
    if not p.exists():
        raise SystemExit(f"لا يوجد أصل بالمعرّف '{asset_id}' في {config.CATALOG}")
    return GlassesAsset.load(p)


class TryOn:
    """
    تنعيم الوضعية زمنيًا في الوضع الحي.

    بلا تنعيم تهتز النظارة بمقدار اهتزاز المعالم نفسه ([مقيس] 0.23–1.22% من
    IPD تحت ضوضاء حسّاس). التنعيم الأُسّي يقلّل الاهتزاز على حساب تأخر بسيط
    في الاستجابة — وهو تبادل مقبول لأن حركة الرأس أبطأ بكثير من معدل الفريمات.
    """

    def __init__(self, asset: GlassesAsset, static: bool):
        self.asset = asset
        self.analyzer = FaceAnalyzer(static=static)
        self._prev = None

    def _smooth(self, fg):
        a = config.TEMPORAL_SMOOTH_ALPHA
        if a <= 0 or self._prev is None:
            self._prev = (fg.pose.rvec.copy(), fg.pose.tvec.copy())
            return fg
        pr, pt = self._prev
        fg.pose.rvec = a * fg.pose.rvec + (1 - a) * pr
        fg.pose.tvec = a * fg.pose.tvec + (1 - a) * pt
        self._prev = (fg.pose.rvec.copy(), fg.pose.tvec.copy())
        return fg

    def process(self, frame, debug=False, smooth=False, with_fit=True):
        fg = self.analyzer.analyze(frame)
        if fg is None:
            return frame, {"error": "NO_FACE"}
        if smooth:
            fg = self._smooth(fg)
        if not fg.ok_for_render:
            return frame, {"blocked": fg.warnings,
                           "pose": {"yaw": round(fg.pose.yaw, 1),
                                    "pitch": round(fg.pose.pitch, 1)}}
        fit = measure.measure(fg, self.asset)
        # نرسم بالمقياس الشخصي فقط إن كان القياس موثوقًا؛ وإلا نرتد للمتوسط
        k = fit.scale_factor_k if fit.reliable else 1.0
        out, rep = compositor.render(frame, fg, self.asset, config.CATALOG,
                                     debug, scale_k=k)
        if with_fit:
            rep["fit"] = fit
        return out, rep

    def close(self):
        self.analyzer.close()


def cmd_photo(ns):
    img = cv2.imread(ns.image)
    if img is None:
        raise SystemExit(f"تعذّرت قراءة الصورة: {ns.image}")
    t = TryOn(load_asset(ns.asset), static=True)
    out, rep = t.process(img, debug=ns.debug)
    t.close()

    fit = rep.pop("fit", None)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if fit is not None:
        print("\n--- تقرير المقاس ---")
        print(fit.as_text())
    if "blocked" not in rep and "error" not in rep:
        Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(ns.out, out)
        print(f"\nكُتبت النتيجة: {ns.out}")


def cmd_live(ns):
    cap = cv2.VideoCapture(ns.camera)
    if not cap.isOpened():
        raise SystemExit(f"تعذّر فتح الكاميرا {ns.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t = TryOn(load_asset(ns.asset), static=False)
    print("q للخروج | d لعرض معلومات التصحيح | m لتقرير المقاس")
    debug, show_fit = ns.debug, False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)          # مرآة، أطبيعية للمستخدم
            out, rep = t.process(frame, debug=debug, smooth=True,
                                 with_fit=show_fit)
            if "blocked" in rep:
                cv2.putText(out, " | ".join(rep["blocked"])[:60], (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            elif show_fit and rep.get("fit") is not None:
                f = rep["fit"]
                txt = (f"{f.verdict}  frame {f.frame_width_mm:.0f}mm  "
                       f"face {f.face_width_mm:.0f}mm")
                cv2.putText(out, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            .6, (0, 220, 0) if f.reliable else (0, 165, 255), 2)
            cv2.imshow("Glasses Try-On", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('d'):
                debug = not debug
            if k == ord('m'):
                show_fit = not show_fit
    finally:
        cap.release()
        cv2.destroyAllWindows()
        t.close()


def cmd_list(_):
    items = sorted(config.CATALOG.glob("*.json"))
    if not items:
        print("الكتالوج فارغ — استعمل: python -m assetkit.prep <صورة> ...")
    for p in items:
        a = GlassesAsset.load(p)
        print(f"{a.id:<16} {a.name:<28} عرض={a.frame_width_mm:.0f}mm  "
              f"{a.lens_width_mm:.0f}□{a.bridge_mm:.0f} {a.temple_length_mm:.0f}"
              f"{'  [ذراع مولّدة]' if a.temple_is_synthetic else ''}")


def main():
    ap = argparse.ArgumentParser(description="تجربة نظارات افتراضية — offline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("photo"); p.set_defaults(fn=cmd_photo)
    p.add_argument("--image", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--out", default="out.png")
    p.add_argument("--debug", action="store_true")

    l = sub.add_parser("live"); l.set_defaults(fn=cmd_live)
    l.add_argument("--asset", required=True)
    l.add_argument("--camera", type=int, default=0)
    l.add_argument("--debug", action="store_true")

    s = sub.add_parser("list"); s.set_defaults(fn=cmd_list)

    ns = ap.parse_args()
    ns.fn(ns)


if __name__ == "__main__":
    main()
