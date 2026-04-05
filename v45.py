# PCA & Surface Classification [VERSION 4.5.1 ΓÇö week7 + 40/20/40 zone split]
# Key changes from v4.x series:
#  1. PCA Rule 1: pick eigenvector with largest |Y| (not eigenvalue) -> fixes square molars
#  2. PCA Rule 3: enforce horizontal axis direction by quadrant -> CONSISTENT M/D direction
#  3. PCA Rule 4: clamp |angle| > 45 deg to 0 (safety net for bad masks)
#  4. Classification: X-thirds (center 1/3 = Occlusal, sides = M/D)
#     Q1/Q4: left=Distal, center=Occlusal, right=Mesial  (per week7 FDI anatomy)
#     Q2/Q3: left=Mesial, center=Occlusal, right=Distal

import os, json, math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SEG_DIR    = r"C:\Users\jaopi\Desktop\SP\week2-Tooth Detection & Segmentation\500-segmentation+recognition"
CARIES_DIR = r"C:\Users\jaopi\Desktop\SP\week3-Caries-to-Tooth Mapping\dental_analysis_output"
OUT_DIR    = "PCA_Output"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_TILT_DEG      = 45.0   # clamp extreme PCA angles (bad tooth masks)
MIN_CLUSTER_SIZE  = 15     # noise removal: drop clusters smaller than this
LEFT_BOUND  = 0.40   # X-split: Left zone  0.00-0.40 (wider = fewer M/D->Occ spillover)
RIGHT_BOUND = 0.60   # X-split: Right zone 0.60-1.00 | Center = 0.40-0.60 (Occlusal)

SURFACE_COLORS = {"Occlusal": "#E74C3C", "Mesial": "#3498DB",
                  "Distal": "#27AE60", "Other": "#2ECC71", -1: "#95A5A6"}

def is_upper_jaw(tid): return int(str(tid)[0]) in [1, 2]
def get_quadrant(tid):  return int(str(tid)[0])

def load_seg(case_id):
    path = os.path.join(SEG_DIR, f"case {case_id}", f"case_{case_id}_results.json")
    if not os.path.exists(path): print(f"[ERROR] SEG: {path}"); return None
    with open(path) as f: return json.load(f)

def load_caries(case_id):
    path = os.path.join(CARIES_DIR, f"case {case_id}", f"case_{case_id}_caries_mapping.json")
    if not os.path.exists(path): print(f"[ERROR] CARIES: {path}"); return None
    with open(path) as f: return json.load(f)

def build_seg_map(seg_data):
    return {str(t["tooth_id"]): t.get("pixel_coordinates", []) for t in seg_data.get("teeth_data", [])}

def get_caries_list(d): return d.get("teeth_caries_data", [])
def compute_centroid(p): a=np.array(p,dtype=np.float64); return float(np.mean(a[:,0])),float(np.mean(a[:,1]))

def get_bbox(pts):
    p=np.array(pts,dtype=np.float64); mn,mx=np.min(p,0),np.max(p,0)
    return mn[0],mn[1],mx[0]-mn[0],mx[1]-mn[1]


# =============================================================================
# NOISE REMOVAL (from week7)
# =============================================================================
def remove_small_clusters(caries_pts, min_cluster=MIN_CLUSTER_SIZE):
    if len(caries_pts) < min_cluster:
        return caries_pts
    pts = np.array(caries_pts, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = 2
    w = x_max - x_min + 1 + 2*pad
    h = y_max - y_min + 1 + 2*pad
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x_min - pad, y_min - pad])
    mask[shifted[:,1], shifted[:,0]] = 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) == 0: return caries_pts
    return np.column_stack([xs + x_min - pad, ys + y_min - pad]).astype(np.float64)


# =============================================================================
# PCA  ΓÇö 4-Rule Orientation (ported from week7 multi_zone_classifier.py)
# =============================================================================
def perform_pca(points, tooth_id):
    """
    4-Rule PCA orientation (week7 reference):
      Rule 1 ΓÇô Verticality: pick eigenvector with larger |Y| as vertical axis
      Rule 2 ΓÇô Occlusal/Apical: flip vertical axis to point toward crown
      Rule 3 ΓÇô Mesial/Distal: enforce horizontal axis direction
               Q1/Q4: horizontal_axis[0] > 0 (Mesial is +X = right)
               Q2/Q3: horizontal_axis[0] < 0 (Mesial is -X = left)
      Rule 4 ΓÇô Angle clamp: if |angle| > MAX_TILT_DEG, use 0 (safety net)
    Returns: (mean, rotation_angle_rad, clamped_bool)
    """
    pts = np.array(points, dtype=np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # cv2.PCACompute for consistency with week7
    _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
    ev0 = eigvecs[0].astype(np.float64)  # largest variance
    ev1 = eigvecs[1].astype(np.float64)  # second

    # Rule 1: Verticality ΓÇö pick eigenvector with larger |Y| as vertical axis
    if abs(ev0[1]) >= abs(ev1[1]):
        vertical_axis   = ev0.copy()
        horizontal_axis = ev1.copy()
    else:
        vertical_axis   = ev1.copy()   # square molar: ev1 is more vertical
        horizontal_axis = ev0.copy()

    # Rule 2: Occlusal/Apical direction
    upper = is_upper_jaw(tooth_id)
    if upper:
        if vertical_axis[1] < 0: vertical_axis = -vertical_axis   # crown faces +Y (down)
    else:
        if vertical_axis[1] > 0: vertical_axis = -vertical_axis   # crown faces -Y (up)

    # Rule 3: Mesial/Distal direction
    # Q1/Q4 (patient right, image left): midline is to the RIGHT -> Mesial at +X
    # Q2/Q3 (patient left, image right): midline is to the LEFT  -> Mesial at -X
    quadrant = get_quadrant(tooth_id)
    if quadrant in [1, 4]:
        if horizontal_axis[0] < 0: horizontal_axis = -horizontal_axis  # ensure +X
    else:
        if horizontal_axis[0] > 0: horizontal_axis = -horizontal_axis  # ensure -X

    # Compute rotation angle
    angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
    rot = math.pi / 2 - angle_from_x
    while rot >  math.pi: rot -= 2 * math.pi
    while rot < -math.pi: rot += 2 * math.pi

    # Rule 4: Angle clamp
    clamped = False
    if abs(math.degrees(rot)) > MAX_TILT_DEG:
        rot = 0.0
        clamped = True

    return mean, rot, clamped


def rotate(pts, center, angle):
    p = np.array(pts, dtype=np.float64) - center
    c, s = np.cos(angle), np.sin(angle)
    return np.dot(p, np.array([[c,-s],[s,c]]).T) + center


# =============================================================================
# CLASSIFICATION  v4.5 ΓÇö X-thirds dominant zone (week7 approach)
# =============================================================================
def classify_surface_full(tid, tooth_pts, caries_pts):
    """
    FDI anatomy reference (after 4-rule PCA, horizontal axis enforced):
      Q1/Q4 (image left):  Left third = Distal | Center = Occlusal | Right third = Mesial
      Q2/Q3 (image right): Left third = Mesial | Center = Occlusal | Right third = Distal

    Dominant zone (most pixels) determines the prediction.
    """
    # Step 1: Noise removal
    caries_clean = remove_small_clusters(caries_pts)
    if len(caries_clean) == 0:
        return "Other", 0.0, {}

    # Step 2: PCA alignment (4-rule)
    center, angle, clamped = perform_pca(tooth_pts, tid)
    tooth_rot   = rotate(tooth_pts,    center, angle)
    caries_rot  = rotate(caries_clean, center, angle)

    x, y, w, h = get_bbox(tooth_rot)
    if w <= 0 or h <= 0:
        return "Other", float(math.degrees(angle)), {}

    rel_xs = np.clip((caries_rot[:,0] - x) / w, 0.0, 1.0)
    n_pts  = len(rel_xs)

    # Step 3: X-thirds zone assignment
    # After Rule 3 enforcement, horizontal axis is consistent per quadrant:
    #   Q1/Q4: +X is toward Mesial -> right third = Mesial, left third = Distal
    #   Q2/Q3: -X is toward Mesial -> left third = Mesial, right third = Distal
    quadrant = get_quadrant(tid)
    if quadrant in [1, 4]:
        # Q1/Q4: low x = Distal, center = Occlusal, high x = Mesial
        d_mask = rel_xs < LEFT_BOUND
        c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
        m_mask = rel_xs > RIGHT_BOUND
    else:
        # Q2/Q3: low x = Mesial, center = Occlusal, high x = Distal
        m_mask = rel_xs < LEFT_BOUND
        c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
        d_mask = rel_xs > RIGHT_BOUND

    m_count = int(np.sum(m_mask))  # Mesial
    c_count = int(np.sum(c_mask))  # Occlusal (center)
    d_count = int(np.sum(d_mask))  # Distal

    # Step 4: Dominant zone wins
    vote_map = {"Mesial": m_count, "Occlusal": c_count, "Distal": d_count}
    winner   = max(vote_map, key=vote_map.get)

    vote_fractions = {k: round(v/max(n_pts,1), 4) for k,v in vote_map.items()}
    vote_fractions["pca_clamped"] = clamped
    return winner, float(math.degrees(angle)), vote_fractions


def compute_caries_stats(caries_pts, tooth_pts):
    cp, tp = len(caries_pts), len(tooth_pts)
    return cp, (cp/tp*100) if tp else 0


def visualize_tooth_with_zones(tooth_pts, caries_pts, tooth_id, classification,
                                vote_fractions=None, save_path=None):
    tooth_pts  = np.array(tooth_pts,  dtype=np.float64)
    caries_pts = np.array(caries_pts, dtype=np.float64)
    x, y = np.min(tooth_pts, axis=0); w, h = np.ptp(tooth_pts, axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    q = get_quadrant(tooth_id)
    # Q1/Q4: left=Distal, center=Occlusal, right=Mesial
    # Q2/Q3: left=Mesial, center=Occlusal, right=Distal
    left_col  = SURFACE_COLORS["Distal"]  if q in [1,4] else SURFACE_COLORS["Mesial"]
    right_col = SURFACE_COLORS["Mesial"] if q in [1,4] else SURFACE_COLORS["Distal"]
    t1 = x + w*LEFT_BOUND; t2 = x + w*RIGHT_BOUND
    ax.fill([x,t1,t1,x],          [y,y,y+h,y+h], alpha=0.18, color=left_col)
    ax.fill([t1,t2,t2,t1],         [y,y,y+h,y+h], alpha=0.18, color=SURFACE_COLORS["Occlusal"])
    ax.fill([t2,x+w,x+w,t2],       [y,y,y+h,y+h], alpha=0.18, color=right_col)
    ax.scatter(tooth_pts[:,0], tooth_pts[:,1], c="gray", s=2, alpha=0.4)
    color = SURFACE_COLORS.get(classification, "#95A5A6")
    if len(caries_pts)>0: ax.scatter(caries_pts[:,0],caries_pts[:,1],c=color,s=10,zorder=5)
    cx,cy = compute_centroid(caries_pts)
    ax.plot(cx,cy,"*",markersize=14,color="gold",zorder=6)
    ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,ls="--",lw=1))
    title = f"Tooth {tooth_id} Q{q} ΓÇö {classification} (v4.5)"
    if vote_fractions:
        v=vote_fractions; title+=f"\nOcc={v.get('Occlusal',0):.2f} M={v.get('Mesial',0):.2f} D={v.get('Distal',0):.2f}"
    ax.set_title(title,fontsize=9); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path,dpi=150,bbox_inches="tight"); plt.close(fig)
    return fig


def process_case(case_id):
    print(f"\n=== CASE {case_id} ===")
    seg_data    = load_seg(case_id)
    caries_data = load_caries(case_id)
    if seg_data is None or caries_data is None: return
    seg_map     = build_seg_map(seg_data)
    caries_list = get_caries_list(caries_data)
    results = {"case_number": case_id, "teeth_data": []}
    for tooth in caries_list:
        tid        = str(tooth["tooth_id"])
        caries_pts = tooth.get("caries_coordinates", [])
        tooth_pts  = seg_map.get(tid, [])
        if len(caries_pts)==0 or len(tooth_pts)<10: continue
        surface,angle,vf = classify_surface_full(tid, tooth_pts, caries_pts)
        pixels,pct = compute_caries_stats(caries_pts, tooth_pts)
        results["teeth_data"].append({
            "tooth_id": tid, "has_caries": True,
            "confidence": tooth.get("confidence",0),
            "caries_position_detail": surface,
            "predicted_surface_fine": surface,
            "vote_fractions": vf,
            "rotation_angle": round(angle,2),
            "tooth_coordinates": tooth_pts,
            "caries_coordinates": caries_pts,
            "caries_pixels": pixels,
            "caries_percentage": round(pct,4)
        })
    case_dir = os.path.join(OUT_DIR, f"case_{case_id}")
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, f"case_{case_id}.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"[DONE] saved -> case_{case_id}.json")


def process_case_visual(case_id):
    case_dir=Path(OUT_DIR)/f"case_{case_id}"; jp=case_dir/f"case_{case_id}.json"
    if not jp.exists(): print(f"[SKIP] Case {case_id}"); return
    with open(jp) as f: data=json.load(f)
    for tooth in data["teeth_data"]:
        tid=tooth["tooth_id"]; tp=tooth["tooth_coordinates"]; cp=tooth["caries_coordinates"]
        c,a,_ = perform_pca(tp,tid); tr=rotate(tp,c,a); cr=rotate(np.array(cp,dtype=np.float64),c,a)
        td=case_dir/f"tooth_{tid}"; td.mkdir(parents=True,exist_ok=True)
        visualize_tooth_with_zones(tr,cr,tid,tooth["caries_position_detail"],
            vote_fractions=tooth.get("vote_fractions",{}),
            save_path=td/f"tooth_{tid}_visual.png")
    print(f"[DONE] visuals -> Case {case_id}")


for cid in range(1, 501):
    process_case(cid)
    process_case_visual(cid)
