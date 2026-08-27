"""
compositor.py — دمج الطبقات على الصورة مع ترتيب عمق صحيح وحجب واقعي.

منطق الحجب: الصورة الأصلية هي الخلفية، فلا يمكن رسم شيء "خلفها". لذلك أي جزء
من النظارة يقع خلف الرأس لا يُرسم أصلاً — نطرح قناع محيط الوجه من ألفا تلك
الطبقة. هذا مكافئ بصريًا للحجب بمخزن العمق، وبلا تكلفة رندر ثلاثي الأبعاد.

ترتيب الرسم يُحدَّد بالعمق المحسوب فعليًا لكل طبقة، لا باصطلاح ثابت — لأن
أي الذراعين أقرب يعتمد على اتجاه دوران الرأس.
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from core import fitting as F
from core import geometry as G
from core.face import FaceGeometry
from assetkit.schema import GlassesAsset


def _alpha_blend(dst_bgr: np.ndarray, layer_bgra: np.ndarray) -> np.ndarray:
    a = (layer_bgra[:, :, 3:4].astype(np.float32) / 255.0)
    return (layer_bgra[:, :, :3].astype(np.float32) * a
            + dst_bgr.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def _match_shading(layer_bgra: np.ndarray, frame_bgr: np.ndarray,
                   face_mask: np.ndarray) -> np.ndarray:
    """
    مطابقة سطوع الأصل لإضاءة المشهد.

    الأصل مصوَّر في استوديو (إضاءة قوية موحّدة) والوجه في إضاءة غرفة. بلا
    مطابقة تبدو النظارة ملصوقة. المعامل = سطوع الوجه الفعلي ÷ سطوع مرجعي،
    ومُقيَّد بمدى ضيّق حتى لا تنقلب النظارة السوداء رمادية أو العكس.
    """
    if not config.SHADING_MATCH:
        return layer_bgra
    face_px = frame_bgr[face_mask > 0]
    if face_px.size == 0:
        return layer_bgra
    lum = float(cv2.cvtColor(face_px.reshape(-1, 1, 3),
                             cv2.COLOR_BGR2GRAY).mean())
    gain = float(np.clip(lum / 128.0, 0.72, 1.28))
    out = layer_bgra.copy()
    out[:, :, :3] = np.clip(out[:, :, :3].astype(np.float32) * gain,
                            0, 255).astype(np.uint8)
    return out


def render(frame_bgr: np.ndarray, fg: FaceGeometry, asset: GlassesAsset,
           base_dir, debug: bool = False,
           scale_k: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    يركّب النظارة على الصورة ويُرجع (الصورة، تقرير).

    scale_k = حجم وجه المستخدم نسبةً للنموذج القانوني. القيمة 1.0 تعني
    "ارسم بحجم الوجه المتوسط" وهي الارتداد الآمن حين يكون القياس غير موثوق.
    """
    h, w = frame_bgr.shape[:2]
    front = asset.load_front(base_dir)
    temple = asset.load_temple(base_dir)
    mm_per_px = asset.mm_per_px(base_dir)

    pl_front = F.place_front(asset, front, mm_per_px, fg.pose, scale_k)
    layers: list[tuple[F.LayerPlacement, np.ndarray]] = [(pl_front, front)]

    if temple is not None:
        t_r = temple
        t_l = cv2.flip(temple, 1)             # الذراع الأخرى معكوسة أفقيًا
        layers.append((F.place_temple(asset, t_l, mm_per_px, fg.pose,
                                      "left", scale_k), t_l))
        layers.append((F.place_temple(asset, t_r, mm_per_px, fg.pose,
                                      "right", scale_k), t_r))

    oval = F.face_oval_mask(fg.landmarks, (h, w))
    front_depth = pl_front.depth_mm

    out = frame_bgr.copy()
    report = {"layers": [], "mm_per_px_asset": round(mm_per_px, 4),
              "scale_k": round(float(scale_k), 3)}

    # من الأبعد إلى الأقرب
    for pl, img in sorted(layers, key=lambda t: -t[0].depth_mm):
        warped = F.warp_layer(img, pl.quad_px, (w, h))

        occluded = pl.name.startswith("temple") and pl.depth_mm > front_depth
        if occluded:
            warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3],
                                              cv2.bitwise_not(oval))

        warped = _match_shading(warped, frame_bgr, oval)
        out = _alpha_blend(out, warped)
        report["layers"].append({"name": pl.name,
                                 "depth_mm": round(pl.depth_mm, 1),
                                 "occluded_by_head": bool(occluded)})

    if debug:
        cv2.polylines(out, [pl_front.quad_px.astype(np.int32)], True, (0, 255, 255), 1)
        for i in (G.IDX['iris_l'], G.IDX['iris_r'], G.IDX['bridge_top'],
                  G.IDX['tragion_l'], G.IDX['tragion_r']):
            cv2.circle(out, tuple(fg.landmarks[i].astype(int)), 2, (0, 0, 255), -1)

    report["pose"] = {"pitch": round(fg.pose.pitch, 1), "yaw": round(fg.pose.yaw, 1),
                      "roll": round(fg.pose.roll, 1),
                      "reproj_rms_px": round(fg.pose.reproj_rms_px, 2),
                      "crosscheck_deg": (None if fg.crosscheck_deg is None
                                         else round(fg.crosscheck_deg, 1))}
    report["warnings"] = fg.warnings
    return out, report
