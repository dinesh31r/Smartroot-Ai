import cv2
import numpy as np

MAX_ROOT_ANALYSIS_SIZE = 900
FAST_ROOT_ANALYSIS_SIZE = 640


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _binarize_root(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_a = thresh
    mask_b = cv2.bitwise_not(thresh)

    h, w = gray.shape
    total = h * w
    area_a = cv2.countNonZero(mask_a)
    area_b = cv2.countNonZero(mask_b)

    def valid_area(area):
        ratio = _safe_div(area, total)
        return 0.01 < ratio < 0.85

    if valid_area(area_a) and not valid_area(area_b):
        mask = mask_a
    elif valid_area(area_b) and not valid_area(area_a):
        mask = mask_b
    else:
        target = 0.2 * total
        mask = mask_a if abs(area_a - target) < abs(area_b - target) else mask_b

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def _skeletonize(binary_mask):
    skel = np.zeros(binary_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_mask.copy()

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _neighbor_counts(skel):
    skel_bin = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_bin, -1, kernel)
    branch_points = np.sum((skel_bin == 1) & (neighbors >= 3))
    end_points = np.sum((skel_bin == 1) & (neighbors == 1))
    return int(branch_points), int(end_points)


def _growth_direction_from_aspect(aspect_ratio):
    if aspect_ratio > 1.35:
        return "Vertical"
    if aspect_ratio < 0.75:
        return "Lateral"
    return "Mixed"


def _branch_density_label(branch_density):
    if branch_density < 0.002:
        return "Low"
    if branch_density < 0.006:
        return "Medium"
    return "High"


def _biomass_label(area_ratio, thickness_mean):
    score = area_ratio * 0.7 + min(thickness_mean / 12.0, 1.0) * 0.3
    if score < 0.18:
        return "Low"
    if score < 0.35:
        return "Medium"
    return "High"


def _age_label(thickness_mean, biomass_level):
    if thickness_mean < 2.4 and biomass_level == "Low":
        return "Young"
    if thickness_mean > 4.8 and biomass_level == "High":
        return "Old"
    return "Mature"


def _root_type_label(branch_density_label, aspect_ratio, length_density):
    if branch_density_label == "High" and length_density > 0.35:
        return "Fibrous"
    if branch_density_label == "Low" and aspect_ratio > 1.3:
        return "Tap"
    return "Adventitious"


def _skeleton_bbox(skel):
    coords = np.column_stack(np.where(skel > 0))
    if coords.size == 0:
        return 0, 0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_max - y_min + 1), int(x_max - x_min + 1)


def _skeleton_angles(skel):
    skel_bin = (skel > 0).astype(np.uint8)
    h, w = skel_bin.shape
    angles = []
    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, -1),
        (-1, 1)
    ]
    ys, xs = np.where(skel_bin > 0)
    for y, x in zip(ys, xs):
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel_bin[ny, nx]:
                angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                angles.append(angle)
    return angles


def _endpoint_angles(skel):
    skel_bin = (skel > 0).astype(np.uint8)
    h, w = skel_bin.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_bin, -1, kernel)
    endpoints = np.column_stack(np.where((skel_bin == 1) & (neighbors == 1)))
    directions = [
        (0, 1), (1, 0), (1, 1), (1, -1),
        (0, -1), (-1, 0), (-1, -1), (-1, 1)
    ]
    angles = []
    for y, x in endpoints:
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel_bin[ny, nx]:
                angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                angles.append((y, angle))
                break
    return angles


def _row_width_at(mask, y):
    h, w = mask.shape
    y = max(0, min(h - 1, y))
    row = mask[y, :]
    xs = np.where(row > 0)[0]
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min() + 1)


def _row_width_near(mask, y, window=3):
    for dy in range(0, window + 1):
        for yy in (y - dy, y + dy):
            w = _row_width_at(mask, yy)
            if w > 0:
                return w
    return 0


def extract_root_traits(image_path, fast=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read root image.")

    h0, w0 = image.shape[:2]
    max_dim = max(h0, w0)
    max_size = FAST_ROOT_ANALYSIS_SIZE if fast else MAX_ROOT_ANALYSIS_SIZE
    if max_dim > max_size:
        scale = max_size / max_dim
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _binarize_root(gray)

    mask_area = cv2.countNonZero(mask)
    h, w = mask.shape
    total_area = h * w

    if mask_area == 0:
        raise ValueError("Root mask could not be extracted from image.")

    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    aspect_ratio = _safe_div(bbox_h, bbox_w)

    skel = _skeletonize(mask)
    skeleton_length = cv2.countNonZero(skel)
    branch_points, end_points = _neighbor_counts(skel)
    branch_density = _safe_div(branch_points, max(skeleton_length, 1))

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    thickness_vals = dist[mask > 0] * 2.0
    thickness_mean = float(np.mean(thickness_vals))
    thickness_std = float(np.std(thickness_vals))

    left_area = cv2.countNonZero(mask[:, : w // 2])
    right_area = cv2.countNonZero(mask[:, w // 2:])
    symmetry_index = 1.0 - _safe_div(abs(left_area - right_area), max(mask_area, 1))
    symmetry_index = float(max(0.0, min(1.0, symmetry_index)))

    length_density = _safe_div(skeleton_length, max(mask_area, 1))
    branching_factor = _safe_div(branch_points, max(end_points, 1))
    root_density = _safe_div(mask_area, total_area)
    root_length_index = _safe_div(skeleton_length, max(h + w, 1))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    root_region = mask > 0
    brown_mask = (h_ch >= 5) & (h_ch <= 30) & (s_ch > 40) & (v_ch < 200)
    dark_mask = v_ch < 60

    brown_ratio = float(_safe_div(np.sum(brown_mask & root_region), max(mask_area, 1)))
    dark_ratio = float(_safe_div(np.sum(dark_mask & root_region), max(mask_area, 1)))

    contour_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
    contour_count = len(contours)
    fragmentation = min(1.0, max(0.0, _safe_div(contour_count - 1, 6)))

    area_ratio = _safe_div(mask_area, total_area)

    thickness_percentiles = {}
    if (not fast) and thickness_vals.size > 0:
        for p in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            thickness_percentiles[f"D{p}"] = float(np.percentile(thickness_vals, p))

    skel_coords = np.column_stack(np.where(skel > 0))
    if skel_coords.size > 0:
        skel_thickness = dist[skel > 0] * 2.0
    else:
        skel_thickness = np.array([])

    skel_percentiles = {}
    if (not fast) and skel_thickness.size > 0:
        for p in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            skel_percentiles[f"DS{p}"] = float(np.percentile(skel_thickness, p))

    skel_depth, skel_width = _skeleton_bbox(skel)

    if fast:
        ang_stats = {"angle_mean": 0.0, "angle_min": 0.0, "angle_max": 0.0}
    else:
        angles = _skeleton_angles(skel)
        if angles:
            ang_stats = {
                "angle_mean": float(np.mean(angles)),
                "angle_min": float(np.min(angles)),
                "angle_max": float(np.max(angles))
            }
        else:
            ang_stats = {"angle_mean": 0.0, "angle_min": 0.0, "angle_max": 0.0}

    endpoint_angles = [] if fast else _endpoint_angles(skel)
    y_min = int(y_min)
    y_max = int(y_max)
    top_thresh = y_min + int(0.2 * bbox_h)
    bottom_thresh = y_max - int(0.2 * bbox_h)

    adv_angles = [a for y, a in endpoint_angles if y <= top_thresh]
    basal_angles = [a for y, a in endpoint_angles if y >= bottom_thresh]

    adventitious_count = len(adv_angles)
    basal_count = len(basal_angles)
    adventitious_angle = float(np.mean(adv_angles)) if adv_angles else 0.0
    basal_angle = float(np.mean(basal_angles)) if basal_angles else 0.0

    hyp_region = mask[y_min:y_min + max(1, int(0.1 * bbox_h)), :]
    hyp_dist = dist[y_min:y_min + max(1, int(0.1 * bbox_h)), :]
    hyp_vals = hyp_dist[hyp_region > 0] * 2.0
    hypocotyl_diameter = float(np.median(hyp_vals)) if hyp_vals.size > 0 else 0.0

    center_x = x_min + bbox_w // 2
    x_left = max(0, center_x - max(1, bbox_w // 10))
    x_right = min(w, center_x + max(1, bbox_w // 10))
    tap_region = dist[:, x_left:x_right]
    tap_mask = mask[:, x_left:x_right] > 0
    tap_vals = tap_region[tap_mask] * 2.0
    taproot_diameter = float(np.max(tap_vals)) if tap_vals.size > 0 else 0.0

    cp_dia25 = _row_width_near(mask, y_min + int(0.25 * bbox_h))
    cp_dia50 = _row_width_near(mask, y_min + int(0.50 * bbox_h))
    cp_dia75 = _row_width_near(mask, y_min + int(0.75 * bbox_h))
    cp_dia90 = _row_width_near(mask, y_min + int(0.90 * bbox_h))

    nodal_len = float(skeleton_length) / max(branch_points, 1)
    if branch_points > 0:
        branch_mask = (skel > 0) & (cv2.filter2D((skel > 0).astype(np.uint8), -1,
                          np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)) >= 3)
        branch_diam = dist[branch_mask] * 2.0
        nodal_avg_dia = float(np.mean(branch_diam)) if branch_diam.size > 0 else 0.0
    else:
        nodal_avg_dia = 0.0

    lateral_branch_freq = _safe_div(branch_points, max(skeleton_length, 1))
    lateral_avg_len = float(skeleton_length) / max(end_points, 1)
    lateral_angle_mean = ang_stats["angle_mean"]
    lateral_angle_min = ang_stats["angle_min"]
    lateral_angle_max = ang_stats["angle_max"]

    branch_density_label = _branch_density_label(branch_density)
    biomass_level = _biomass_label(area_ratio, thickness_mean)
    age_label = _age_label(thickness_mean, biomass_level)
    root_type = _root_type_label(branch_density_label, aspect_ratio, length_density)
    growth_direction = _growth_direction_from_aspect(aspect_ratio)

    return {
        "height": h,
        "width": w,
        "mask_area": int(mask_area),
        "area_ratio": float(area_ratio),
        "root_area": int(mask_area),
        "avg_root_density": float(area_ratio),
        "skeleton_length": int(skeleton_length),
        "branch_points": int(branch_points),
        "end_points": int(end_points),
        "branch_density": float(branch_density),
        "branch_density_label": branch_density_label,
        "aspect_ratio": float(aspect_ratio),
        "growth_direction": growth_direction,
        "symmetry_index": symmetry_index,
        "length_density": float(length_density),
        "branching_factor": float(branching_factor),
        "root_density": float(root_density),
        "root_length_index": float(root_length_index),
        "root_system_depth": int(bbox_h),
        "root_system_width": int(bbox_w),
        "skeleton_depth": int(skel_depth),
        "skeleton_width": int(skel_width),
        "thickness_mean": thickness_mean,
        "thickness_std": thickness_std,
        "diameter_percentiles": thickness_percentiles,
        "skeleton_diameter_percentiles": skel_percentiles,
        "root_distribution_x": float(_safe_div(np.mean(coords[:, 1]) - x_min, max(bbox_w, 1))),
        "root_distribution_y": float(_safe_div(np.mean(coords[:, 0]) - y_min, max(bbox_h, 1))),
        "root_tip_count": int(end_points),
        "top_angle": float(np.mean(adv_angles)) if adv_angles else 0.0,
        "bottom_angle": float(np.mean(basal_angles)) if basal_angles else 0.0,
        "angle_mean": ang_stats["angle_mean"],
        "angle_min": ang_stats["angle_min"],
        "angle_max": ang_stats["angle_max"],
        "adventitious_count": int(adventitious_count),
        "basal_count": int(basal_count),
        "adventitious_angle": float(adventitious_angle),
        "basal_angle": float(basal_angle),
        "taproot_diameter": float(taproot_diameter),
        "hypocotyl_diameter": float(hypocotyl_diameter),
        "cp_dia25": int(cp_dia25),
        "cp_dia50": int(cp_dia50),
        "cp_dia75": int(cp_dia75),
        "cp_dia90": int(cp_dia90),
        "nodal_length": float(nodal_len),
        "nodal_avg_diameter": float(nodal_avg_dia),
        "lateral_branch_freq": float(lateral_branch_freq),
        "lateral_avg_length": float(lateral_avg_len),
        "lateral_angle_mean": float(lateral_angle_mean),
        "lateral_angle_min": float(lateral_angle_min),
        "lateral_angle_max": float(lateral_angle_max),
        "brown_ratio": brown_ratio,
        "dark_ratio": dark_ratio,
        "fragmentation": fragmentation,
        "root_type": root_type,
        "biomass_level": biomass_level,
        "age_estimate": age_label
    }
