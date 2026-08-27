"""
schema.py — عقد بيانات أصل النظارة.

الأصل ليس صورة واحدة: هو صورة واجهة + ذراعان + نقاط ارتساء + مقاسات فيزيائية.
بدون المقاسات الفيزيائية تصير التجربة "شكل" بلا معنى قياسي.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass
class Anchors:
    """نقاط الارتساء داخل صورة الواجهة، بالبكسل."""
    bridge_center: tuple[float, float]   # منتصف الجسر — يجلس على أنف المستخدم
    hinge_left: tuple[float, float]      # مفصلة الذراع اليسرى (يسار الصورة)
    hinge_right: tuple[float, float]
    lens_center_left: tuple[float, float] | None = None
    lens_center_right: tuple[float, float] | None = None


@dataclass
class GlassesAsset:
    id: str
    name: str

    # --- المقاسات الفيزيائية (مم) ---
    # المصدر: ترقيم الإطار المطبوع على الذراع بصيغة  lens–bridge–temple
    # مثال 52□18 140  =>  lens_width=52، bridge=18، temple_length=140
    frame_width_mm: float                 # العرض الكلي للواجهة
    lens_width_mm: float
    bridge_mm: float
    temple_length_mm: float
    lens_height_mm: float

    anchors: Anchors
    front_png: str                        # مسار نسبي لصورة الواجهة (BGRA)
    temple_png: str | None = None         # ذراع واحدة (تُعكس للجهة الأخرى)
    temple_is_synthetic: bool = True      # هل الذراع مولّدة أم مقصوصة من صورة حقيقية
    lens_opacity: float | None = None     # None => استعمل الافتراضي من config
    source_note: str = ""                 # من وين جاءت المقاسات ومن أين الصورة
    measured: dict = field(default_factory=dict)   # أي أرقام مقيسة أثناء التجهيز

    # ---------------------------------------------------------------- IO
    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{self.id}.json"
        d = asdict(self)
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @staticmethod
    def load(path: Path) -> "GlassesAsset":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["anchors"] = Anchors(**{k: (tuple(v) if v is not None else None)
                                  for k, v in d["anchors"].items()})
        return GlassesAsset(**d)

    # ---------------------------------------------------------------- صور
    def load_front(self, base: Path) -> np.ndarray:
        img = cv2.imread(str(Path(base) / self.front_png), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(self.front_png)
        if img.shape[2] == 3:
            raise ValueError("صورة الواجهة يجب أن تحتوي قناة ألفا (BGRA)")
        return img

    def load_temple(self, base: Path) -> np.ndarray | None:
        if not self.temple_png:
            return None
        return cv2.imread(str(Path(base) / self.temple_png), cv2.IMREAD_UNCHANGED)

    # ---------------------------------------------------------------- مقياس
    def mm_per_px(self, base: Path) -> float:
        """
        معامل تحويل صورة الأصل إلى مليمترات حقيقية.
        يُشتق من العرض الفعلي للبكسلات غير الشفافة مقابل frame_width_mm.
        """
        a = self.load_front(base)[:, :, 3]
        cols = np.where(a.max(axis=0) > 8)[0]
        if cols.size < 2:
            raise ValueError("صورة الواجهة فارغة أو ألفا غير صحيحة")
        return self.frame_width_mm / float(cols[-1] - cols[0] + 1)

    def validate(self, base: Path) -> list[str]:
        """فحوصات تمنع أصلاً معطوبًا من الوصول لمحرك التركيب."""
        errs: list[str] = []
        try:
            front = self.load_front(base)
        except Exception as e:
            return [f"front_png: {e}"]

        h, w = front.shape[:2]
        for nm in ("bridge_center", "hinge_left", "hinge_right"):
            x, y = getattr(self.anchors, nm)
            if not (0 <= x < w and 0 <= y < h):
                errs.append(f"{nm} خارج حدود الصورة")

        hl, hr = self.anchors.hinge_left, self.anchors.hinge_right
        if hl[0] >= hr[0]:
            errs.append("hinge_left يجب أن تكون يسار hinge_right")

        # تحقق تناسق: مجموع المقاسات المطبوعة يقارب العرض الكلي
        expected = 2 * self.lens_width_mm + self.bridge_mm
        if self.frame_width_mm and abs(expected - self.frame_width_mm) > 0.25 * self.frame_width_mm:
            errs.append(f"عدم تناسق مقاسات: 2×lens+bridge={expected:.0f}mm "
                        f"بعيد عن frame_width={self.frame_width_mm:.0f}mm")
        if not (30 <= self.frame_width_mm <= 165):
            errs.append(f"frame_width_mm={self.frame_width_mm} خارج المدى المعقول")

        # تحقق متقاطع بين المقاسات المعلَنة والهندسة المقاسة من الصورة نفسها:
        # المسافة بين مركزي العدستين يجب أن تساوي lens_width + bridge.
        a = self.anchors
        if a.lens_center_left and a.lens_center_right:
            mmpp = self.frame_width_mm / max(1, front.shape[1])
            seen = abs(a.lens_center_right[0] - a.lens_center_left[0]) * mmpp
            declared = self.lens_width_mm + self.bridge_mm
            if abs(seen - declared) > 0.12 * declared:
                errs.append(
                    f"تعارض: المسافة بين مركزي العدستين في الصورة {seen:.0f}mm "
                    f"بينما الترقيم المعلَن يقول {declared:.0f}mm — راجع الأرقام")
        return errs
