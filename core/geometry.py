"""
geometry.py — النموذج القانوني، حل الوضعية 6DoF، والتحويل إلى مقياس مِتري.

كل الإحداثيات ثلاثية الأبعاد بالمليمتر في فضاء الوجه:
  X → يمين المشاهد،  Y → أعلى،  Z → للأمام (خارج الوجه).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

import config

# --- فهارس معالم MediaPipe المستعملة (توبولوجيا 478 نقطة) -------------------
IDX = dict(
    nose_tip=1, chin=152, bridge_top=168, bridge_mid=6, nose_base=2,
    eye_out_l=33, eye_out_r=263, eye_in_l=133, eye_in_r=362,
    iris_l=468, iris_r=473,
    iris_l_ring=(469, 470, 471, 472), iris_r_ring=(474, 475, 476, 477),
    tragion_l=234, tragion_r=454,
    mouth_l=61, mouth_r=291, forehead=10, philtrum=199,
)

# نقاط حل الوضعية: متفرقة ومنتشرة على العمق للحصول على شرطية عددية جيدة.
PNP_IDS = [IDX['nose_tip'], IDX['chin'], IDX['eye_out_l'], IDX['eye_out_r'],
           IDX['eye_in_l'], IDX['eye_in_r'], IDX['bridge_top'],
           IDX['tragion_l'], IDX['tragion_r'], IDX['mouth_l'],
           IDX['mouth_r'], IDX['philtrum']]

# محيط الوجه (face oval) — يُستعمل كقناع حجب للأذرع
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@lru_cache(maxsize=1)
def canonical_mm() -> np.ndarray:
    """رؤوس النموذج القانوني (468×3) بالمليمتر."""
    verts = []
    with open(config.CANONICAL_OBJ) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    if v.shape[0] != 468:
        raise ValueError(f"النموذج القانوني تالف: {v.shape[0]} رأس بدل 468")
    return v * config.CANONICAL_UNIT_TO_MM


def intrinsics(w: int, h: int, hfov_deg: float | None = None) -> np.ndarray:
    """مصفوفة الكاميرا الداخلية من مجال الرؤية الأفقي."""
    hfov = np.radians(hfov_deg if hfov_deg is not None else config.CAMERA_HFOV_DEG)
    fx = (w / 2.0) / np.tan(hfov / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def euler_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """تفكيك ZYX → (pitch=X, yaw=Y, roll=Z) بالدرجات، مطويّة إلى ±90."""
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x, y, z = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0

    def fold(a):
        a = (np.degrees(a) + 180.0) % 360.0 - 180.0
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        return float(a)

    return fold(x), fold(y), fold(z)


@dataclass
class Pose:
    rvec: np.ndarray            # (3,1) متجه الدوران
    tvec: np.ndarray            # (3,1) الإزاحة بالمليمتر
    K: np.ndarray               # (3,3) مصفوفة الكاميرا
    pitch: float
    yaw: float
    roll: float
    reproj_rms_px: float        # خطأ إعادة الإسقاط
    depth_mm: float             # مسافة الوجه عن الكاميرا (تحت فرض الوجه القانوني)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    def project(self, pts3d_mm: np.ndarray) -> np.ndarray:
        """يُسقط نقاطًا من فضاء الوجه (مم) إلى بكسل الصورة."""
        p, _ = cv2.projectPoints(np.asarray(pts3d_mm, np.float64).reshape(-1, 1, 3),
                                 self.rvec, self.tvec, self.K, None)
        return p.reshape(-1, 2)

    def mm_per_px_at_face(self) -> float:
        """مقياس تقريبي عند مستوى الوجه: كم مليمترًا يمثله البكسل الواحد."""
        return float(self.depth_mm / self.K[0, 0])


def solve_pose(landmarks_px: np.ndarray, w: int, h: int,
               hfov_deg: float | None = None) -> Pose | None:
    """
    يحل الوضعية من معالم ثنائية الأبعاد مقابل النموذج القانوني.

    ملاحظة مهمة عن غموض المقياس: العدسة الواحدة لا تفرّق بين وجه صغير قريب
    ووجه كبير بعيد. حل PnP بنموذج ثابت الحجم يفكّ هذا الغموض بـ**افتراض** أن
    وجه المستخدم بأبعاد النموذج القانوني. لذلك depth_mm وأي قياس مِتري مشتق
    منه صالحان للوضع البصري، ولا يُعتمدان كقياس شخصي إلا بعد معايرة
    (انظر core/calibrate.py).
    """
    obj = canonical_mm()[PNP_IDS]
    img = np.ascontiguousarray(landmarks_px[PNP_IDS, :2], dtype=np.float64)
    K = intrinsics(w, h, hfov_deg)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # تحسين بالمربعات الصغرى بعد الحل الابتدائي
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    R = cv2.Rodrigues(rvec)[0]
    pitch, yaw, roll = euler_from_R(R)
    return Pose(rvec=rvec, tvec=tvec, K=K, pitch=pitch, yaw=yaw, roll=roll,
                reproj_rms_px=rms, depth_mm=float(tvec[2, 0]))


def iris_diameter_px(landmarks_px: np.ndarray) -> float:
    """متوسط قطر القزحية الأفقي بالبكسل من حلقتي القزحية."""
    L = landmarks_px
    dl = np.linalg.norm(L[IDX['iris_l_ring'][2]] - L[IDX['iris_l_ring'][0]])
    dr = np.linalg.norm(L[IDX['iris_r_ring'][2]] - L[IDX['iris_r_ring'][0]])
    return float((dl + dr) / 2.0)


def ipd_px(landmarks_px: np.ndarray) -> float:
    """المسافة بين مركزي البؤبؤين بالبكسل (نقاط القزحية الحقيقية)."""
    return float(np.linalg.norm(landmarks_px[IDX['iris_l']] - landmarks_px[IDX['iris_r']]))


def scale_from_iris(landmarks_px: np.ndarray) -> tuple[float, float]:
    """
    مقياس مم/بكسل مشتق من قطر القزحية، مع تقدير الخطأ النسبي.

    القزحية مرساة أفضل من IPD لأن تباينها بين البشر أصغر، لكن قياسها يتحلل
    سريعًا مع صغر الوجه — لذلك تُرجع الدالة الخطأ المتوقع ليقرر المستدعي.
    """
    d_px = iris_diameter_px(landmarks_px)
    if d_px <= 0:
        return float("nan"), float("inf")
    mm_per_px = config.IRIS_DIAMETER_MM / d_px
    rel_err = 0.5 / d_px          # ±0.5px دقة تحديد المعلم
    return mm_per_px, float(rel_err)
