"""
face.py — استخراج هندسة الوجه، مع تحقق متقاطع بين محرّكين مستقلين.

المحرك الأساسي: MediaPipe FaceMesh (478 نقطة، فيها القزحية).
المحرك الرقيب:  InsightFace 1k3d68 (وضعية محسوبة بطريقة مختلفة كليًا).

سبب وجود الرقيب: لو أخطأ محرك واحد في الوضعية، النظارة تُركّب بزاوية غلط
وتبدو النتيجة "شغالة" وهي خاطئة. اتفاق مصدرين مستقلين هو الفحص الوحيد المتاح
بلا حقيقة أرضية (ground truth).
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import warnings

os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore")

import cv2
import numpy as np

import config
from core import geometry as G


@dataclass
class FaceGeometry:
    landmarks: np.ndarray        # (478,2) بكسل في إحداثيات الصورة الكاملة
    pose: G.Pose
    bbox: tuple[float, float, float, float]
    face_px: float               # عرض صندوق الوجه
    crosscheck_deg: float | None # أكبر فرق زاوي عن المحرك الرقيب
    warnings: list[str]

    @property
    def ok_for_render(self) -> bool:
        return not any(w.startswith("BLOCK") for w in self.warnings)


class MediaPipeBackend:
    """المحرك الأساسي. يعمل بالكامل بلا شبكة (الموديل داخل الحزمة، إصدار مثبّت)."""

    def __init__(self, static: bool = True, max_faces: int = 1):
        from mediapipe.python.solutions import face_mesh as mpfm
        self._mesh = mpfm.FaceMesh(static_image_mode=static,
                                   max_num_faces=max_faces,
                                   refine_landmarks=True,          # يفعّل القزحية
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

    def landmarks(self, bgr: np.ndarray) -> np.ndarray | None:
        res = self._mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        h, w = bgr.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)

    def close(self):
        self._mesh.close()


class InsightFaceRefereeBackend:
    """
    الرقيب. يُحمّل كاشف SCRFD ونموذج المعالم ثلاثية الأبعاد فقط.
    اصطلاح محاوره يختلف: pitch و yaw معكوسا الإشارة عن اصطلاحنا.
    [مقيس] bench_pose.py على 4 وجوه: متوسط الفرق pitch 2.0° / yaw 3.7° / roll 1.6°.
    """
    SIGN = np.array([-1.0, -1.0, 1.0])   # (pitch, yaw, roll)

    def __init__(self):
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(name="buffalo_l",
                                 allowed_modules=["detection", "landmark_3d_68"],
                                 providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def pose_and_bbox(self, bgr: np.ndarray):
        faces = self._app.get(bgr)
        if not faces:
            return None, None
        f = max(faces, key=lambda z: z.bbox[2] - z.bbox[0])
        if getattr(f, "pose", None) is None:
            return None, tuple(float(v) for v in f.bbox)
        return np.asarray(f.pose, np.float64) * self.SIGN, tuple(float(v) for v in f.bbox)


class FaceAnalyzer:
    def __init__(self, static: bool = True, crosscheck: bool | None = None):
        self.primary = MediaPipeBackend(static=static)
        self.crosscheck = config.CROSSCHECK_ENABLED if crosscheck is None else crosscheck
        self.referee = InsightFaceRefereeBackend() if self.crosscheck else None

    def analyze(self, bgr: np.ndarray, hfov_deg: float | None = None) -> FaceGeometry | None:
        h, w = bgr.shape[:2]
        L = self.primary.landmarks(bgr)
        if L is None:
            return None

        pose = G.solve_pose(L, w, h, hfov_deg)
        if pose is None:
            return None

        x1, y1 = L.min(axis=0)
        x2, y2 = L.max(axis=0)
        face_px = float(x2 - x1)
        warns: list[str] = []

        if face_px < config.MIN_FACE_PX_RENDER:
            warns.append(f"BLOCK_FACE_TOO_SMALL:{face_px:.0f}px"
                         f"<{config.MIN_FACE_PX_RENDER}")
        if face_px < config.MIN_FACE_PX_MEASURE:
            warns.append(f"NO_METRIC:{face_px:.0f}px<{config.MIN_FACE_PX_MEASURE}")
        if abs(pose.yaw) > config.MAX_YAW_DEG:
            warns.append(f"BLOCK_YAW:{pose.yaw:.0f}°")
        if abs(pose.pitch) > config.MAX_PITCH_DEG:
            warns.append(f"BLOCK_PITCH:{pose.pitch:.0f}°")

        cc = None
        if self.referee is not None:
            ref_pose, _ = self.referee.pose_and_bbox(bgr)
            if ref_pose is not None:
                mine = np.array([pose.pitch, pose.yaw, pose.roll])
                cc = float(np.max(np.abs(mine - ref_pose)))
                if cc > config.CROSSCHECK_MAX_DEG:
                    warns.append(f"BLOCK_POSE_DISAGREE:{cc:.1f}°")

        return FaceGeometry(landmarks=L, pose=pose, bbox=(x1, y1, x2, y2),
                            face_px=face_px, crosscheck_deg=cc, warnings=warns)

    def close(self):
        self.primary.close()
