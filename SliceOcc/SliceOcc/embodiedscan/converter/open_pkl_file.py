import os, sys, pickle, json
from collections import Counter, defaultdict

import numpy as np

PKL_PATH = "/home/sequor/Downloads/embodiedscan/embodiedscan_infos_train.pkl"  # 경로 맞게 수정

def load_pickle(path):
    # Py2로 저장된 pkl 대비
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")
    return obj

def short(v, maxlen=120):
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen] + "..."

def np_info(arr):
    try:
        return {"type": str(type(arr)), "dtype": str(arr.dtype), "shape": list(arr.shape)}
    except Exception:
        return {"type": str(type(arr))}

def summarize_value(v, depth=0):
    """소규모 요약 (재귀적으로 타입/크기만)"""
    if isinstance(v, dict):
        return {k: summarize_value(v[k], depth+1) if depth < 1 else str(type(v[k])) for k in list(v.keys())[:20]}
    elif isinstance(v, (list, tuple)):
        out = {"type": type(v).__name__, "len": len(v)}
        if len(v) > 0:
            sample0 = v[0]
            if isinstance(sample0, (dict, list, tuple, np.ndarray)):
                out["sample0"] = summarize_value(sample0, depth+1)
            else:
                out["sample0"] = {"type": type(sample0).__name__, "value": short(sample0)}
        return out
    elif isinstance(v, np.ndarray):
        return np_info(v)
    else:
        return {"type": type(v).__name__, "value": short(v)}

def safe_get(d, key, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

def check_file(path):
    return os.path.exists(path)

def main():
    data = load_pickle(PKL_PATH)

    print("=== TOP-LEVEL ===")
    print("type:", type(data))
    if isinstance(data, list):
        print("num_samples:", len(data))
        if len(data) == 0:
            return
        sample0 = data[0]
        print("\n=== KEYS IN FIRST SAMPLE ===")
        if isinstance(sample0, dict):
            print(sorted(list(sample0.keys())))
            print("\n=== STRUCTURE SUMMARY (first sample) ===")
            print(json.dumps(summarize_value(sample0), indent=2, ensure_ascii=False))
        else:
            print("First item type:", type(sample0))
    elif isinstance(data, dict):
        print("keys:", list(data.keys()))
        print("\n=== STRUCTURE SUMMARY (top-level) ===")
        print(json.dumps(summarize_value(data), indent=2, ensure_ascii=False))

    # 통계 수집 (리스트 형태 가정: samples = list[dict])
    if not isinstance(data, list):
        return
    samples = data

    # 흔히 쓰는 키들: sample_idx / scan_id / images / instances / axis_align_matrix / occ 경로 등
    key_counter = Counter()
    missing_keys = defaultdict(int)

    # 이미지 개수 분포, bbox 개수 분포 등
    n_images_hist = Counter()
    n_instances_hist = Counter()

    # 경로 유효성 체크(이미지, occupancy)
    exist_counts = Counter()

    for s in samples:
        if not isinstance(s, dict):
            continue

        for k in s.keys():
            key_counter[k] += 1

        # images
        images = safe_get(s, "images", [])
        if isinstance(images, list):
            n_images_hist[len(images)] += 1
            # 경로/포즈/내참(K) 체크
            for im in images[:5]:  # 전부 검사하면 느릴 수 있어 앞의 몇 개만
                if isinstance(im, dict):
                    img_path = safe_get(im, "img_path")
                    if img_path is not None:
                        exist_counts["img_path_total"] += 1
                        if check_file(img_path):
                            exist_counts["img_path_exist"] += 1
                    if isinstance(safe_get(im, "intrinsics"), np.ndarray):
                        exist_counts["has_intrinsics"] += 1
                    if isinstance(safe_get(im, "pose"), np.ndarray):
                        exist_counts["has_pose"] += 1

        else:
            missing_keys["images"] += 1

        # instances (3D bbox) 통계
        instances = safe_get(s, "instances", [])
        if isinstance(instances, list):
            n_instances_hist[len(instances)] += 1
        else:
            missing_keys["instances"] += 1

        # occupancy/visible mask 경로 추정 키가 있다면 체크 (데이터셋 구현에 맞게 수정)
        # 보통 parse_ann_info에서 sample_idx/scan_id로 경로 조립함. 직접 경로 키가 있다면 여기서 확인.
        # 예시) s.get("occupancy_path"), s.get("visible_occupancy_path")
        occ_path = safe_get(s, "occupancy_path")
        if occ_path is not None:
            exist_counts["occ_path_total"] += 1
            if check_file(occ_path):
                exist_counts["occ_path_exist"] += 1

    print("\n=== KEY PRESENCE (how many samples contain each key) ===")
    for k, c in key_counter.most_common():
        print(f"{k:30s} : {c}/{len(samples)}")

    if missing_keys:
        print("\n=== MISSING KEYS COUNT (samples missing these keys) ===")
        for k, c in missing_keys.items():
            print(f"{k:30s} : {c}")

    print("\n=== #IMAGES HISTOGRAM ===")
    for n, c in sorted(n_images_hist.items()):
        print(f"{n:3d} images : {c}")

    print("\n=== #INSTANCES HISTOGRAM ===")
    for n, c in sorted(n_instances_hist.items()):
        print(f"{n:3d} boxes : {c}")

    print("\n=== PATH / CAMERA FIELDS QUICK CHECK (sampled) ===")
    for k, v in exist_counts.items():
        print(f"{k:25s} : {v}")

    # 샘플 몇 개를 더 자세히
    print("\n=== SAMPLE PREVIEW (first 2) ===")
    for i, s in enumerate(samples[:2]):
        print(f"\n--- sample[{i}] ---")
        print("keys:", sorted(list(s.keys())))
        # 대표 메타
        for mk in ["sample_idx", "scan_id"]:
            if mk in s:
                print(f"{mk}: {s[mk]}")
        # 이미지 한두 장만 디테일
        images = safe_get(s, "images", [])
        if isinstance(images, list) and images:
            im0 = images[0]
            if isinstance(im0, dict):
                print("image[0].img_path:", im0.get("img_path"))
                K = im0.get("intrinsics")
                Pose = im0.get("pose")
                if isinstance(K, np.ndarray):
                    print("image[0].intrinsics:", {"shape": K.shape, "dtype": K.dtype})
                if isinstance(Pose, np.ndarray):
                    print("image[0].pose:", {"shape": Pose.shape, "dtype": Pose.dtype})

        # occupancy/visible mask 직접 경로가 들어있다면 확인
        if "occupancy_path" in s:
            print("occupancy_path:", s["occupancy_path"], "exists?", check_file(s["occupancy_path"]))

if __name__ == "__main__":
    main()
