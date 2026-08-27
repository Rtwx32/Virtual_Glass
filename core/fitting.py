"""
fitting.py — وضع طبقات النظارة في الفضاء ثلاثي الأبعاد للوجه ثم إسقاطها.

الفكرة المركزية: لا نلصق صورة على صورة. نضع كل طبقة على **مستوى ثلاثي الأبعاد**
داخل فضاء الوجه بالمليمتر، ثم نُسقط زواياها الأربع بوضعية الرأس المحسوبة،
ونحسب تحويل منظور (homography) من الصورة الأصلية إلى الرباعي المُسقَط.
هذا ما يعطي المنظور الصحيح عند دوران الرأس بدل الشكل المسطّح.

المستويات:
  • الواجهة : مستوى XY للوجه، مائل حول محور X بزاوية pantoscopic tilt،
              مرتكز على جسر الأنف مع إزاحة للأمام وللأسفل.
  • الذراع  : مستوى موازٍ للمستوى السهمي (sagittal) عند x المفصلة،
              يمتد من المفصلة نحو نقطة أمام الأذن (tragion).
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

import config
from core import geometry as G
from assetkit.schema import GlassesAsset


@dataclass
class LayerPlacement:
    name: str                 # front | temple_left | temple_right
    quad_px: np.ndarray       # (4,2) الرباعي المُسقَط: TL,TR,BR,BL
    depth_mm: float           # عمق مركز الطبقة في فضاء الكاميرا
    visible: bool


def _rot_x(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _eye_center_3d(side: str) -> np.ndarray:
    c = G.canonical_mm()
    a = c[G.IDX['eye_out_l'] if side == "left" else G.IDX['eye_out_r']]
    b = c[G.IDX['eye_in_l'] if side == "left" else G.IDX['eye_in_r']]
    return (a + b) / 2.0


def _optical_anchor_3d(k: float = 1.0) -> np.ndarray:
    """
    نقطة الارتكاز = منتصف مركزَي العينين، مزاحة للأمام بسماكة الإطار.

    اخترنا المركز البصري لا الجسر لأن القاعدة البصرية هي أن يقع مركز العدسة
    على البؤبؤ (fitting height)؛ الارتساء على الجسر يورّث أي انحياز في مكان
    شريط الجسر داخل صورة المنتج، وهو يختلف من إطار لآخر.
    """
    p = (_eye_center_3d("left") + _eye_center_3d("right")) / 2.0 * k
    p[2] += config.BRIDGE_FORWARD_OFFSET_MM
    p[1] -= config.BRIDGE_DROP_MM
    return p


def _front_basis() -> tuple[np.ndarray, np.ndarray]:
    """متجها الاتجاه للواجهة: (يمين، أعلى) بعد تطبيق الميل الرأسي."""
    R = _rot_x(-config.PANTOSCOPIC_TILT_DEG)
    return R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])


def asset_image_anchor(asset: GlassesAsset) -> tuple[float, float]:
    """نقطة الارتساء داخل صورة الأصل: منتصف مركزَي العدستين، وإلا الجسر."""
    a = asset.anchors
    if a.lens_center_left and a.lens_center_right:
        return ((a.lens_center_left[0] + a.lens_center_right[0]) / 2.0,
                (a.lens_center_left[1] + a.lens_center_right[1]) / 2.0)
    return a.bridge_center


def front_pixel_to_face3d(asset: GlassesAsset, mm_per_px: float,
                          uv: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    يحوّل بكسلات صورة الواجهة إلى نقاط ثلاثية الأبعاد في فضاء الوجه القانوني.

    معامل k هو حجم وجه المستخدم نسبةً للنموذج القانوني (من core.measure).
    الرسم يتم في الفضاء القانوني، فجسم حقيقي عرضه W مم على وجه أكبر بـ k
    يظهر في ذلك الفضاء بعرض W/k. تجاهل هذا يجعل النظارة تُرسم بحجم متوسط
    على كل الوجوه بدل حجمها الحقيقي.
    """
    bu, bv = asset_image_anchor(asset)
    right, up = _front_basis()
    origin = _optical_anchor_3d(k)
    s = mm_per_px / max(k, 1e-6)
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    du = (uv[:, 0] - bu) * s
    dv = (uv[:, 1] - bv) * s                  # محور v للأسفل → نطرحه من "أعلى"
    return origin + du[:, None] * right - dv[:, None] * up


def place_front(asset: GlassesAsset, front_img: np.ndarray, mm_per_px: float,
                pose: G.Pose, k: float = 1.0) -> LayerPlacement:
    h, w = front_img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    p3 = front_pixel_to_face3d(asset, mm_per_px, corners, k)
    quad = pose.project(p3)
    cam = (pose.R @ p3.T).T + pose.tvec.reshape(1, 3)
    return LayerPlacement("front", quad, float(cam[:, 2].mean()), True)


def place_temple(asset: GlassesAsset, temple_img: np.ndarray, mm_per_px: float,
                 pose: G.Pose, side: str, k: float = 1.0) -> LayerPlacement:
    """
    side='left' يعني يسار الصورة (= يمين المستخدم في صورة مرآة).
    الذراع تمتد من المفصلة نحو نقطة أمام الأذن ثم تتجاوزها بطول الذراع الحقيقي.
    """
    th, tw = temple_img.shape[:2]
    hinge_uv = (asset.anchors.hinge_left if side == "left"
                else asset.anchors.hinge_right)
    hinge3d = front_pixel_to_face3d(asset, mm_per_px, np.array([hinge_uv]), k)[0]

    trag = G.canonical_mm()[G.IDX['tragion_l'] if side == "left"
                            else G.IDX['tragion_r']].copy()
    trag[1] = hinge3d[1]                      # الذراع أفقية تقريبًا
    direction = trag - hinge3d
    n = np.linalg.norm(direction)
    if n < 1e-6:
        direction = np.array([0.0, 0.0, -1.0])
    else:
        direction = direction / n

    up = np.array([0.0, 1.0, 0.0])
    length_mm = asset.temple_length_mm / max(k, 1e-6)
    height_mm = th * (length_mm / tw)          # حافظ على نسبة صورة الذراع

    # الصورة: u من المفصلة للخلف، v من أعلى لأسفل
    def corner(uu, vv):
        return (hinge3d + direction * (uu / tw * length_mm)
                + up * ((th / 2.0 - vv) / th * height_mm))

    p3 = np.array([corner(0, 0), corner(tw, 0), corner(tw, th), corner(0, th)])
    quad = pose.project(p3)
    cam = (pose.R @ p3.T).T + pose.tvec.reshape(1, 3)
    return LayerPlacement(f"temple_{side}", quad, float(cam[:, 2].mean()), True)


def warp_layer(img_bgra: np.ndarray, quad: np.ndarray,
               out_size: tuple[int, int]) -> np.ndarray:
    """تحويل منظور من صورة الطبقة إلى الرباعي المُسقَط على لوحة بحجم الصورة."""
    h, w = img_bgra.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.warpPerspective(img_bgra, M, out_size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def face_oval_mask(landmarks: np.ndarray, shape: tuple[int, int],
                   dilate_px: int = 2) -> np.ndarray:
    """قناع مصمت لمحيط الوجه — يستعمل لحجب الذراع البعيدة خلف الرأس."""
    poly = landmarks[G.FACE_OVAL].astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(poly), 255)
    if dilate_px:
        m = cv2.dilate(m, np.ones((dilate_px * 2 + 1,) * 2, np.uint8))
    return m
