"""
مقارنة تجريبية بين InsightFace (2d106det + 1k3d68) و MediaPipe FaceMesh
على المعايير التي تهم تركيب النظارة تحديدًا:
  1) زمن الاستدلال على CPU (فريم كامل + وجه واحد)
  2) توفّر نقاط الارتساء اللازمة للنظارة
  3) تقدير الوضعية 6DoF واتفاق الاثنين
  4) الاهتزاز (jitter) تحت ضوضاء حسّاس محاكاة
  5) دقة قياس المسافة بين البؤبؤين (IPD)
"""
import os, sys, time, json
import numpy as np
import cv2

os.environ["GLOG_minloglevel"] = "3"
import warnings; warnings.filterwarnings("ignore")

IMG = "/home/claude/lab/t1.png"
OUT = "/home/claude/lab/bench_result.json"
N_JITTER = 12
RNG = np.random.default_rng(1234)

# ---------------------------------------------------------------- InsightFace
from insightface.app import FaceAnalysis
_if_app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"],
                       providers=["CPUExecutionProvider"])
_if_app.prepare(ctx_id=-1, det_size=(640, 640))

# ---------------------------------------------------------------- MediaPipe
from mediapipe.python.solutions import face_mesh as mp_face_mesh
_mp = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                            refine_landmarks=True, min_detection_confidence=0.5)

# نقاط MediaPipe المرجعية (تسميات ثابتة في التوبولوجيا 478)
MP = dict(
    r_iris=473, l_iris=468,          # مركز البؤبؤ (يتطلب refine_landmarks)
    r_eye_out=263, l_eye_out=33,     # الزاوية الخارجية للعين
    r_eye_in=362, l_eye_in=133,      # الزاوية الداخلية
    nose_bridge=168,                 # أعلى جسر الأنف — نقطة ارتكاز النظارة
    nose_tip=1, chin=152,
    r_tragion=454, l_tragion=234,    # أمام الأذن — نقطة تعليق الذراع
)

# نموذج وجه قانوني ثلاثي الأبعاد (مم) لـ solvePnP
CANON = np.array([
    [0.0,    0.0,    0.0],      # nose_bridge
    [0.0,  -33.0,  -12.0],      # nose_tip  (تحت الجسر)
    [0.0, -110.0,  -25.0],      # chin
    [-45.0,  -5.0,  -30.0],     # l_eye_out
    [ 45.0,  -5.0,  -30.0],     # r_eye_out
    [-75.0,  -8.0,  -78.0],     # l_tragion
    [ 75.0,  -8.0,  -78.0],     # r_tragion
], dtype=np.float64)


def mp_landmarks(bgr):
    res = _mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float64)


def solve_pose(pts2d, w, h):
    f = w  # تقريب طول بؤري افتراضي
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(CANON, pts2d, K, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return dict(yaw=yaw, pitch=pitch, roll=roll, rvec=rvec, tvec=tvec, K=K)


def mp_pose(L, w, h):
    p = np.array([L[MP['nose_bridge']][:2], L[MP['nose_tip']][:2], L[MP['chin']][:2],
                  L[MP['l_eye_out']][:2], L[MP['r_eye_out']][:2],
                  L[MP['l_tragion']][:2], L[MP['r_tragion']][:2]])
    return solve_pose(p, w, h)


def noisy(img, k):
    """محاكاة ضوضاء حسّاس + ضغط — لقياس اهتزاز النقاط بين الفريمات."""
    n = RNG.normal(0, 3.0, img.shape)
    out = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    img = cv2.imread(IMG)
    H, W = img.shape[:2]
    report = {"image": f"{W}x{H}", "faces": []}

    # ---- كشف الوجوه مرة واحدة عبر SCRFD
    t0 = time.perf_counter()
    faces = _if_app.get(img)
    t_full = (time.perf_counter() - t0) * 1000
    report["insightface_full_frame_ms"] = round(t_full, 1)
    report["n_faces"] = len(faces)
    faces = sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]))[:4]

    for idx, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox
        bw, bh = x2 - x1, y2 - y1
        m = 0.6
        cx1, cy1 = int(max(0, x1 - bw * m)), int(max(0, y1 - bh * m))
        cx2, cy2 = int(min(W, x2 + bw * m)), int(min(H, y2 + bh * m))
        crop = img[cy1:cy2, cx1:cx2].copy()
        ch, cw = crop.shape[:2]

        rec = {"idx": idx, "bbox_px": [round(float(bw), 1), round(float(bh), 1)],
               "det_score": round(float(f.det_score), 3)}

        # ---- InsightFace: وضعية جاهزة من 1k3d68
        rec["if_pose_builtin"] = [round(float(v), 1) for v in f.pose] \
            if getattr(f, "pose", None) is not None else None

        # زمن نقاط 106 لوجه واحد
        t0 = time.perf_counter()
        for _ in range(3):
            _if_app.models['landmark_2d_106'].get(img, f)
        rec["if_106_ms"] = round((time.perf_counter() - t0) * 1000 / 3, 1)

        # ---- MediaPipe على نفس القصّة
        t0 = time.perf_counter()
        L = mp_landmarks(crop)
        rec["mp_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["mp_found"] = L is not None

        if L is not None:
            pose = mp_pose(L, cw, ch)
            rec["mp_pose"] = [round(pose['yaw'], 1), round(pose['pitch'], 1),
                              round(pose['roll'], 1)] if pose else None
            ipd_iris = np.linalg.norm(L[MP['l_iris']][:2] - L[MP['r_iris']][:2])
            rec["mp_ipd_iris_px"] = round(float(ipd_iris), 2)

        # IPD من نقاط InsightFace الخمس (مركز العين التقريبي)
        kps = f.kps
        ipd_kps = np.linalg.norm(kps[0] - kps[1])
        rec["if_ipd_kps_px"] = round(float(ipd_kps), 2)

        # ---- اهتزاز النقاط تحت ضوضاء
        if_j, mp_j = [], []
        for k in range(N_JITTER):
            nz = noisy(crop, k)
            fs = _if_app.get(nz)
            if fs:
                g = max(fs, key=lambda z: (z.bbox[2] - z.bbox[0]))
                if_j.append(np.array([g.kps[0], g.kps[1]]).ravel())
            Ln = mp_landmarks(nz)
            if Ln is not None:
                mp_j.append(np.array([Ln[MP['l_iris']][:2],
                                      Ln[MP['r_iris']][:2]]).ravel())

        def jit(arr, ipd):
            if len(arr) < 4 or ipd <= 0:
                return None
            a = np.stack(arr)
            return round(float(np.mean(a.std(axis=0))) / ipd * 100, 3)

        rec["if_jitter_pct_ipd"] = jit(if_j, ipd_kps)
        rec["mp_jitter_pct_ipd"] = jit(mp_j, rec.get("mp_ipd_iris_px", 0) or 0)
        rec["if_jitter_n"] = len(if_j)
        rec["mp_jitter_n"] = len(mp_j)

        report["faces"].append(rec)
        print(json.dumps(rec, ensure_ascii=False, default=float))

    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    print("\nSAVED", OUT)


if __name__ == "__main__":
    main()
