"""
اختبارات المشروع. المبدأ: نختبر خصائص يمكن إثباتها فعلاً (ثوابت هندسية،
لا-تغيّر تحت التحويلات، اشتغال الحواجز)، لا "تبدو النتيجة حلوة".
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core import geometry as G
from assetkit import prep
from assetkit.schema import GlassesAsset
from tools.make_test_asset import draw

CATALOG = config.CATALOG


# ----------------------------------------------------------- النموذج القانوني
def test_canonical_shape():
    assert G.canonical_mm().shape == (468, 3)


@pytest.mark.parametrize("a,b,lo,hi,label", [
    (33, 263, 82.0, 98.0, "المسافة بين الزاويتين الخارجيتين للعين"),
    (234, 454, 140.0, 165.0, "عرض ما بين الأذنين"),
    (168, 152, 105.0, 140.0, "من جسر الأنف إلى الذقن"),
])
def test_canonical_is_anthropometrically_sane(a, b, lo, hi, label):
    """النموذج القانوني يجب أن يقع داخل المدى البشري المنشور، وإلا كل مقاس مبني عليه خطأ."""
    d = float(np.linalg.norm(G.canonical_mm()[a] - G.canonical_mm()[b]))
    assert lo <= d <= hi, f"{label} = {d:.1f}mm خارج [{lo}, {hi}]"


# ------------------------------------------------------------------ الوضعية
@pytest.mark.parametrize("angles", [(0, 0, 0), (12, -20, 8), (-25, 35, -15),
                                    (5, 55, 40), (-30, -5, -60)])
def test_euler_roundtrip(angles):
    """تفكيك الزوايا يجب أن يعيد بناء المصفوفة نفسها."""
    pitch, yaw, roll = np.radians(angles)
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    out = np.array(G.euler_from_R(R))
    R2 = _rebuild(np.radians(out))
    assert np.allclose(R, R2, atol=1e-6), f"{angles} -> {out}"


def _rebuild(a):
    p, y, r = a
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_pose_recovers_synthetic_rotation():
    """
    اختبار حقيقة أرضية: ندوّر النموذج القانوني بزاوية معلومة، نُسقطه، ثم
    نطلب من solve_pose استرجاع الزاوية. هذا يفصل خطأ الهندسة عن خطأ الكاشف.
    """
    W = H = 800
    K = G.intrinsics(W, H)
    for truth in [(0, 0, 0), (10, -18, 6), (-14, 28, -12)]:
        R = _rebuild(np.radians(truth))
        t = np.array([[0.0], [0.0], [600.0]])
        pts = (R @ G.canonical_mm().T).T + t.reshape(1, 3)
        proj = (K @ pts.T).T
        lm = proj[:, :2] / proj[:, 2:3]
        lm = np.vstack([lm, np.zeros((10, 2))])       # حشو لبلوغ 478
        pose = G.solve_pose(lm, W, H)
        got = np.array([pose.pitch, pose.yaw, pose.roll])
        assert np.allclose(got, truth, atol=0.5), f"{truth} -> {got}"
        assert pose.reproj_rms_px < 0.5
        assert abs(pose.depth_mm - 600.0) < 1.0


# -------------------------------------------------------------- تجهيز الأصل
@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("assets")
    src = d / "g.jpg"
    cv2.imwrite(str(src), draw("rect", 900, 360))
    a = prep.prepare(src, d, "t_rect", "اختبار", lens_mm=50.8, bridge_mm=11.5,
                     temple_mm=140, lens_h_mm=40, frame_w_mm=None,
                     source_note="test")
    return a, d


def test_prep_finds_two_lenses(demo):
    a, _ = demo
    assert a.measured["lens_holes_found"] == 2


def test_prep_anchors_are_symmetric(demo):
    """الأصل التجريبي مرسوم متماثلًا، فالنقاط المكتشفة يجب أن تكون متماثلة."""
    a, d = demo
    w = a.load_front(d).shape[1]
    mid = w / 2.0
    assert abs(a.anchors.bridge_center[0] - mid) < 0.02 * w
    l, r = a.anchors.lens_center_left, a.anchors.lens_center_right
    assert abs((mid - l[0]) - (r[0] - mid)) < 0.02 * w
    assert abs(l[1] - r[1]) < 0.02 * w


def test_prep_hinges_at_extremes(demo):
    a, d = demo
    alpha = a.load_front(d)[:, :, 3]
    cols = np.where(alpha.max(axis=0) > 8)[0]
    assert abs(a.anchors.hinge_left[0] - cols[0]) <= 3
    assert abs(a.anchors.hinge_right[0] - cols[-1]) <= 3


def test_validate_rejects_inconsistent_specs(demo):
    """ترقيم مطبوع لا يطابق هندسة الصورة يجب أن يُرفض، لا أن يمر بصمت."""
    a, d = demo
    # ترقيم يدّعي عدسة 20mm وجسر 5mm بينما العرض الكلي المعلَن 140mm
    bad = GlassesAsset(**{**a.__dict__, "lens_width_mm": 20.0, "bridge_mm": 5.0,
                          "frame_width_mm": 140.0})
    assert bad.validate(d), "الفحص لم يلتقط تعارض المقاسات"


# ------------------------------------------------------------------ الحواجز
def test_scale_estimate_is_resolution_invariant():
    """
    خاصية جوهرية: معامل حجم الوجه k يجب ألا يتغير بتغيّر أبعاد الصورة،
    لأنه نسبة بين عمقين — وكلاهما يتناسب مع البُعد البؤري.
    """
    W = H = 900
    R = _rebuild(np.radians((0, 0, 0)))
    t = np.array([0.0, 0.0, 700.0])
    K = G.intrinsics(W, H)
    pts = (R @ G.canonical_mm().T).T + t
    proj = (K @ pts.T).T
    lm = proj[:, :2] / proj[:, 2:3]
    pose_a = G.solve_pose(np.vstack([lm, np.zeros((10, 2))]), W, H)

    s = 2.0
    lm2 = lm * s
    pose_b = G.solve_pose(np.vstack([lm2, np.zeros((10, 2))]),
                          int(W * s), int(H * s))
    # العمق نفسه لأن البُعد البؤري تضاعف مع الصورة
    assert abs(pose_a.depth_mm - pose_b.depth_mm) / pose_a.depth_mm < 0.01


def test_min_face_thresholds_are_ordered():
    assert config.MIN_FACE_PX_RENDER < config.MIN_FACE_PX_MEASURE


def test_iris_scale_error_matches_derived_threshold():
    """
    عتبة MIN_FACE_PX_MEASURE مشتقة من شرط: خطأ قياس القزحية ≤2%.
    نتحقق أن الاشتقاق متسق مع نفسه.
    """
    iris_px_needed = 0.5 / 0.02
    ipd = iris_px_needed / 0.186
    face_px = ipd * 2.2
    assert abs(face_px - config.MIN_FACE_PX_MEASURE) < 40


# ------------------------------------------------- استخراج الإطار من صورة مرتدَاة
def test_extraction_window_is_wider_than_any_real_frame():
    """
    حارس التسرّب يقارن امتداد الظل بعرض النافذة. هذا يفترض أن النافذة أوسع
    من أي إطار حقيقي، وإلا صار الحارس يرفض النجاح.
    """
    from assetkit import extract_worn as ew
    assert ew.WIN_W_MM > 165, "نافذة الاستخراج أضيق من أعرض إطار واقعي"
    assert abs(ew.OUT_W / ew.WIN_W_MM - ew.PX_PER_MM) < 1e-9


def test_rectified_scale_is_exact():
    """المقياس في الصورة المستوية ثابت معلوم، لا يُستنتج من الصورة."""
    from assetkit import extract_worn as ew
    assert abs(60.0 * ew.PX_PER_MM - 60.0 * ew.OUT_W / ew.WIN_W_MM) < 1e-9
