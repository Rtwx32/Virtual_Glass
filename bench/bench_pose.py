"""
قياس الوضعية بمرجع مِتري حقيقي (canonical_face_model, وحدة سم) بدل نموذج تقديري.
يقارن: اتفاق الوضعية بين المصدرين، خطأ إعادة الإسقاط، وفرق قياس IPD.
"""
import os, json, time
import numpy as np, cv2
os.environ["GLOG_minloglevel"] = "3"
import warnings; warnings.filterwarnings("ignore")

from insightface.app import FaceAnalysis
from mediapipe.python.solutions import face_mesh as mpfm

V = np.array([[float(x) for x in l.split()[1:4]]
              for l in open('/home/claude/lab/canonical_face_model.obj')
              if l.startswith('v ')], dtype=np.float64)          # سم

# نقاط ارتساء متفرقة ومنتشرة (أفضل شرطية لـ solvePnP)
PNP_IDS = [1, 152, 33, 263, 133, 362, 168, 234, 454, 61, 291, 199]
OBJ = V[PNP_IDS] * 10.0                                          # سم -> مم

IRIS_L, IRIS_R = 468, 473
EYE_OUT_L, EYE_OUT_R = 33, 263
BRIDGE, TRAG_L, TRAG_R = 168, 234, 454


def euler(R):
    """تفكيك ZYX قياسي: pitch=X, yaw=Y, roll=Z (بالدرجات)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x, y, z = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0
    return np.degrees([x, y, z])


def wrap(a):
    """يطوي الزاوية إلى [-90,90] لإزالة انقلاب 180° الناتج عن محور Y المقلوب."""
    a = (a + 180) % 360 - 180
    if a > 90:  a -= 180
    if a < -90: a += 180
    return a


def main():
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_3d_68"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    mesh = mpfm.FaceMesh(static_image_mode=True, max_num_faces=1,
                         refine_landmarks=True, min_detection_confidence=0.5)

    img = cv2.imread('/home/claude/lab/t1.png')
    H, W = img.shape[:2]
    faces = sorted(app.get(img), key=lambda f: -(f.bbox[2] - f.bbox[0]))[:4]

    rows = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox
        bw, bh = x2 - x1, y2 - y1
        m = .6
        c = img[int(max(0, y1 - bh * m)):int(min(H, y2 + bh * m)),
                int(max(0, x1 - bw * m)):int(min(W, x2 + bw * m))].copy()
        ch, cw = c.shape[:2]

        res = mesh.process(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        L = np.array([[p.x * cw, p.y * ch] for p in lm], dtype=np.float64)

        fx = cw * 1.2                       # ~55° مجال رؤية أفقي نموذجي لكاميرا ويب
        K = np.array([[fx, 0, cw / 2], [0, fx, ch / 2], [0, 0, 1]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(OBJ, L[PNP_IDS], K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        e = euler(R)
        mp_pitch, mp_yaw, mp_roll = wrap(e[0]), wrap(e[1]), wrap(e[2])

        proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - L[PNP_IDS]) ** 2, axis=1))))
        ipd_px = np.linalg.norm(L[IRIS_L] - L[IRIS_R])
        rms_pct = rms / ipd_px * 100

        ifp = f.pose  # [pitch, yaw, roll]
        rows.append(dict(
            face=i, det=round(float(f.det_score), 3),
            if_pitch=round(float(ifp[0]), 1), mp_pitch=round(mp_pitch, 1),
            if_yaw=round(float(ifp[1]), 1),   mp_yaw=round(mp_yaw, 1),
            if_roll=round(float(ifp[2]), 1),  mp_roll=round(mp_roll, 1),
            reproj_rms_px=round(rms, 2), reproj_pct_ipd=round(rms_pct, 1),
            ipd_iris_px=round(float(ipd_px), 2),
            eye_outer_px=round(float(np.linalg.norm(L[EYE_OUT_L] - L[EYE_OUT_R])), 2),
            kps_eye_px=round(float(np.linalg.norm(f.kps[0] - f.kps[1])), 2),
            depth_mm=round(float(tvec[2][0]), 0),
        ))

    print(f"{'#':>2} {'det':>5} | {'pitch IF/MP':>14} | {'yaw IF/MP':>14} |"
          f" {'roll IF/MP':>14} | {'reproj':>8} | {'IPD':>6} {'kps':>6} {'Δ%':>5}")
    for r in rows:
        d = (r['kps_eye_px'] - r['ipd_iris_px']) / r['ipd_iris_px'] * 100
        print(f"{r['face']:>2} {r['det']:>5} | {r['if_pitch']:>6}/{r['mp_pitch']:>7} |"
              f" {r['if_yaw']:>6}/{r['mp_yaw']:>7} | {r['if_roll']:>6}/{r['mp_roll']:>7} |"
              f" {r['reproj_pct_ipd']:>6}% | {r['ipd_iris_px']:>6} {r['kps_eye_px']:>6}"
              f" {d:>4.1f}%")

    A = np.array([[r['if_pitch'], r['if_yaw'], r['if_roll']] for r in rows])
    B = np.array([[r['mp_pitch'], r['mp_yaw'], r['mp_roll']] for r in rows])
    print("\nمتوسط الفرق المطلق (درجة): pitch=%.1f yaw=%.1f roll=%.1f"
          % tuple(np.mean(np.abs(A - B * np.array([1, -1, -1])), axis=0)))
    json.dump(rows, open('/home/claude/lab/pose_result.json', 'w'),
              indent=2, default=float)


if __name__ == "__main__":
    main()
