"""
measure.py — القياس المِتري: هل هذا الإطار يناسب هذا الوجه فعلاً؟

المشكلة الأساسية: كاميرا واحدة لا تستطيع التفريق بين وجه صغير قريب ووجه كبير
بعيد. حل PnP بنموذج قانوني يفكّ الغموض بافتراض أن وجه المستخدم بحجم المتوسط،
وهذا افتراض دائري لو استعملناه للحكم على المقاس.

الحل هنا: **القزحية كمسطرة**. قطر القزحية الأفقي ثابت تشريحيًا تقريبًا
(≈11.7مم) وتباينه بين البشر أصغر بكثير من تباين حجم الوجه. فإذا:

    Z_pnp   = العمق الذي يفترضه PnP (بافتراض وجه قانوني)
    Z_iris  = العمق الحقيقي المشتق من قطر القزحية المقاس
    k       = Z_iris / Z_pnp

فإن k هو معامل حجم وجه المستخدم نسبةً للمتوسط، وكل مقاس قانوني × k يعطي
مقاس المستخدم الحقيقي.

حدود الطريقة، صراحةً:
  • خطأ قياس القزحية ≈ ±0.5px  →  خطأ نسبي = 0.5 / قطرها بالبكسل
  • التباين البيولوجي لقطر القزحية ≈ ±0.5مم أي ≈ ±4.3%
  • هذان يجتمعان تربيعيًا؛ عند قزحية 30px يصير الخطأ الكلي ≈ ±4.6%
    أي ±7مم على وجه عرضه 150مم. صالح للإرشاد، لا لوصفة طبية.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config
from core import geometry as G
from core.face import FaceGeometry
from assetkit.schema import GlassesAsset

IRIS_BIO_CV = 0.043          # التباين البيولوجي النسبي لقطر القزحية


@dataclass
class FitReport:
    reliable: bool
    scale_factor_k: float          # حجم الوجه نسبةً للنموذج القانوني
    rel_error: float               # الخطأ النسبي المقدَّر
    face_width_mm: float           # عرض ما بين الأذنين
    pupil_distance_mm: float       # PD الحقيقي للمستخدم
    frame_width_mm: float
    frame_pd_mm: float             # مسافة مركزي عدستي الإطار
    width_delta_mm: float          # الإطار − الوجه
    pd_delta_mm: float             # PD الإطار − PD المستخدم
    verdict: str
    notes: list[str]

    def as_text(self) -> str:
        pm = f"±{self.rel_error * 100:.1f}%"
        lines = [
            f"الحكم: {self.verdict}",
            f"عرض وجهك المقدَّر: {self.face_width_mm:.0f}mm ({pm})",
            f"عرض الإطار: {self.frame_width_mm:.0f}mm "
            f"(الفرق {self.width_delta_mm:+.0f}mm)",
            f"PD المقدَّر: {self.pupil_distance_mm:.0f}mm | PD الإطار: "
            f"{self.frame_pd_mm:.0f}mm (الفرق {self.pd_delta_mm:+.0f}mm)",
        ]
        lines += [f"• {n}" for n in self.notes]
        return "\n".join(lines)


def measure(fg: FaceGeometry, asset: GlassesAsset) -> FitReport:
    notes: list[str] = []
    canon = G.canonical_mm()
    fx = fg.pose.K[0, 0]

    d_px = G.iris_diameter_px(fg.landmarks)
    if d_px <= 1.0:
        return _unreliable(asset, "تعذّر قياس القزحية")

    z_iris = config.IRIS_DIAMETER_MM * fx / d_px
    z_pnp = fg.pose.depth_mm
    k = float(z_iris / z_pnp) if z_pnp > 0 else float("nan")

    meas_err = 0.5 / d_px
    rel_err = float(np.hypot(meas_err, IRIS_BIO_CV))

    reliable = True
    if fg.face_px < config.MIN_FACE_PX_MEASURE:
        reliable = False
        notes.append(f"الوجه {fg.face_px:.0f}px < {config.MIN_FACE_PX_MEASURE}px "
                     "المطلوبة للقياس — قرّب الكاميرا")
    if abs(fg.pose.yaw) > 15:
        reliable = False
        notes.append(f"دوران أفقي {fg.pose.yaw:.0f}° يقصّر القزحية ظاهريًا "
                     "ويضخّم المقاس — انظر للكاميرا مباشرة")
    if not (0.80 <= k <= 1.25):
        reliable = False
        notes.append(f"معامل الحجم k={k:.2f} خارج المدى البشري المعقول — "
                     "غالبًا خطأ في معايرة مجال رؤية الكاميرا")

    face_w = float(np.linalg.norm(canon[G.IDX['tragion_l']] -
                                  canon[G.IDX['tragion_r']]) * k)
    pd = float(np.linalg.norm(canon[G.IDX['eye_in_l']] -
                              canon[G.IDX['eye_in_r']]) * 0 +
               _canonical_pd() * k)

    frame_pd = asset.lens_width_mm + asset.bridge_mm
    dw = asset.frame_width_mm - face_w
    dpd = frame_pd - pd

    if not reliable:
        verdict = "غير موثوق — القياس معطّل"
    elif abs(dw) <= 4 and abs(dpd) <= 3:
        verdict = "مقاس مناسب"
    elif dw > 10:
        verdict = "الإطار أعرض من وجهك — سينزلق"
    elif dw < -10:
        verdict = "الإطار أضيق من وجهك — سيضغط على الصدغين"
    elif abs(dpd) > 5:
        verdict = "مركز العدسة بعيد عن بؤبؤك — إزاحة بصرية"
    else:
        verdict = "مقبول مع تحفّظ"

    notes.append("كل الأرقام تقديرية بالقزحية كمرجع، وليست قياسًا بصريًا رسميًا")
    if fg.pose.reproj_rms_px / max(1.0, G.ipd_px(fg.landmarks)) > 0.12:
        notes.append("خطأ إعادة إسقاط مرتفع — ملامحك تختلف عن النموذج القانوني")

    return FitReport(reliable=reliable, scale_factor_k=k, rel_error=rel_err,
                     face_width_mm=face_w, pupil_distance_mm=pd,
                     frame_width_mm=asset.frame_width_mm, frame_pd_mm=frame_pd,
                     width_delta_mm=dw, pd_delta_mm=dpd,
                     verdict=verdict, notes=notes)


def _canonical_pd() -> float:
    """PD في النموذج القانوني: منتصف كل عين من زاويتيها."""
    c = G.canonical_mm()
    l = (c[G.IDX['eye_out_l']] + c[G.IDX['eye_in_l']]) / 2.0
    r = (c[G.IDX['eye_out_r']] + c[G.IDX['eye_in_r']]) / 2.0
    return float(np.linalg.norm(l - r))


def _unreliable(asset: GlassesAsset, why: str) -> FitReport:
    return FitReport(False, float("nan"), float("inf"), float("nan"),
                     float("nan"), asset.frame_width_mm,
                     asset.lens_width_mm + asset.bridge_mm,
                     float("nan"), float("nan"), "غير موثوق", [why])
