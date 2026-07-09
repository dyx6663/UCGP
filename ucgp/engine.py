import os

# --- 限制 CPU 线程数 ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

# --- CUDA allocator ---
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import math
import gc
import glob
import json
import shutil
import random
import threading
import traceback
from functools import lru_cache
from collections import defaultdict, OrderedDict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import warnings
try:
    from tqdm import TqdmWarning
    warnings.filterwarnings("ignore", category=TqdmWarning)
except Exception:
    pass


def _maybe_copy_ckpt_to_local(ckpt_path: str, local_dir: str = None):
    """Optionally copy remote checkpoints to a local cache to reduce IO jitter."""
    if local_dir is None:
        local_dir = os.environ.get("UCGP_CKPT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "ucgp", "ckpts"))
    try:
        if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
            return ckpt_path
        remote_prefixes = [p for p in os.environ.get("UCGP_REMOTE_CKPT_PREFIXES", "").split(os.pathsep) if p]
        should_cache = any(os.path.abspath(ckpt_path).startswith(os.path.abspath(p)) for p in remote_prefixes)
        if should_cache:
            os.makedirs(local_dir, exist_ok=True)
            dst = os.path.join(local_dir, os.path.basename(ckpt_path))
            try:
                if os.path.isfile(dst) and (os.path.getsize(dst) == os.path.getsize(ckpt_path)):
                    return dst
            except Exception:
                pass
            print(f"[CKPT] Copying {ckpt_path} -> {dst}", flush=True)
            shutil.copy2(ckpt_path, dst)
            return dst
    except Exception as e:
        print(f"[CKPT] Copy failed: {e}", flush=True)
    return ckpt_path


LOG_LEVEL = 1  # 0=最详细；数值越大越安静
LOSS_EPS = 1e-6


def log(msg: str, level: int = 1):
    force_keep = (
        "[BEST_LOSS_UP]", "[FINAL]", "[ERR]", "[DET]", "[DATA]", "[CKPT]",
        "[STAGE1]", "[INIT]", "[PREPROC]", "[EV]", "[GEN_START]", "[GEN_END]"
    )
    if any(k in msg for k in force_keep):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{t}] {msg}", flush=True)
        return
    if LOG_LEVEL < level:
        return
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{t}] {msg}", flush=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAVE_BEST_SNAPSHOT_EVERY_GENS = 10
MAX_SAMPLES = 300
REEVAL_FREQ = 5
SAVE_IMAGES_ON_BEST_FULL = True
BEST_LATEST_DIRNAME = "best_full_latest"

MASTER_SIZE = 512
FAST_RENDER_BLUR_KSIZE = 0
FAST_RENDER_LINE_SAMPLES = 48
EXPANSION_RATIO = 0.40
ROI_CORE_DIV = 4.0

CVAR_Q = 0.50
EV_SUBSPACE_RANK = 16
LAMBDA_TOPO = 0.12
LAMBDA_PHYS = 0.03

PATCH_COLOR_RGB = (0, 0, 0)
DARKEN_STRENGTH = 0.85

LLAMA_FACTORY_ROOT = os.environ.get("UCGP_LLAMA_FACTORY_ROOT", "")
if LLAMA_FACTORY_ROOT and os.path.isdir(LLAMA_FACTORY_ROOT) and LLAMA_FACTORY_ROOT not in sys.path:
    sys.path.append(LLAMA_FACTORY_ROOT)

try:
    from mmdet.apis import init_detector, inference_detector
except Exception as e:
    init_detector = None
    inference_detector = None
    log(f"[WARN] mmdet import failed: {repr(e)}", level=0)

try:
    from scripts.infer_clips import AutoCLIPModel, AutoCLIPProcessor
except Exception as e:
    try:
        from .backends.autoclip import AutoCLIPModel, AutoCLIPProcessor
        log("[INIT] Using bundled Hugging Face CLIP backend.", level=0)
    except Exception as e2:
        AutoCLIPModel = None
        AutoCLIPProcessor = None
        log(f"[WARN] CLIP backend import failed: scripts.infer_clips={repr(e)} bundled={repr(e2)}", level=0)


DETECTOR_CONFIG = os.environ.get("UCGP_DETECTOR_CONFIG", "")
DETECTOR_CKPT = os.environ.get("UCGP_DETECTOR_CKPT", "")
SCORE_THR = float(os.environ.get("UCGP_SCORE_THR", "0.5"))

IMG_EXTS = ["jpg", "jpeg", "png", "webp", "bmp"]
RECURSIVE_SCAN = True

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}

DE_POP, DE_GENS = 50, 100

JDE_TAU1 = 0.1
JDE_TAU2 = 0.1
JDE_F_LO, JDE_F_HI = 0.4, 0.95
JDE_CR_LO, JDE_CR_HI = 0.2, 0.95

DISC_MUT_P0 = 0.12
DISC_MUT_P1 = 0.05

STAGNATION_LIMIT = 10
RESTART_RATIO = 0.40
RESTART_NOISE = 0.20

ELITE_SIZE = 8

LOCAL_SEARCH_RATE = 0.25
LOCAL_SEARCH_MAX_ROUNDS = 1
LOCAL_SEARCH_STEP = 0.10

META_GUIDED_RATIO = 0.30
META_UPDATE_FREQ = 2
META_DECAY = 0.95
META_EXPLORATION = 0.15

META_STABILITY_IN_MUTATE = 0.55
ELITE_FULLEVAL_CREDIT = True

RESAMPLE_PROB_BASE = 0.22
RESAMPLE_STAGNATION = 4
RESTART_RESET_STRATEGY_PROB = 0.85

CREDIT_BLEND_RATIO = 0.75

K_POS = 3.0
K_NEG = 1.25
FALLBACK_SIGMA = 0.010
CALIB_MAX_KEEP = 256
CALIB_MIN_SAMPLES = 12
CALIB_CLAMP_MIN = 0.003
CALIB_CLAMP_MAX = 0.080

EVAL_MODE = "rolling"
ASR_EPS = 1e-6

G = 5
GATE_THR = 0.55
THICKNESS_TO_CELL_RATIO = 0.20
MAX_CURV_RATIO = 0.40
ALPHA_GAMMA = 0.8

STANDARD_SIZE = 100.0
_BASE_NODES_CACHE = {}
_EDGE_LIST_CACHE = {}

CLIP_BATCH_SIZE = 32
ELITE_FULL_EVAL_FREQ = 1
TOPK_FORCE_FULL_CONFIRM = 8
ELITE_INJECT_N = 1
PREVIEW_MODE, RANDOM_SEED = False, 42

_clip_model = None
_clip_processor = None
_clip_class_embs = None
_clip_logit_scale = None

_ev_mu, _ev_U = None, None
_clean_ev_feats = None
_topo_sigma = None

_PREPROC_GEOM = None
CLIP_INPUT_SIZE = 224

gpu_evaluator = None


def reset_per_run_globals():
    global _clip_model, _clip_processor, _clip_class_embs, _clip_logit_scale
    global _ev_mu, _ev_U, _clean_ev_feats, _topo_sigma
    global _PREPROC_GEOM, CLIP_INPUT_SIZE
    global gpu_evaluator

    try:
        if gpu_evaluator is not None:
            try:
                gpu_evaluator.clear_mask_cache()
            except Exception:
                pass
    except Exception:
        pass

    gpu_evaluator = None
    _clip_model = None
    _clip_processor = None
    _clip_class_embs = None
    _clip_logit_scale = None

    _ev_mu, _ev_U = None, None
    _clean_ev_feats = None
    _topo_sigma = None
    _PREPROC_GEOM = None
    CLIP_INPUT_SIZE = 224

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _dummy_pil(sz: int = 512, color=(127, 127, 127)) -> Image.Image:
    arr = np.zeros((sz, sz, 3), dtype=np.uint8)
    arr[:, :] = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    return Image.fromarray(arr, mode="RGB")


def _resize_dims_keep_ratio(H: int, W: int, shorter: int):
    shorter = int(shorter)
    if H <= 0 or W <= 0:
        return shorter, shorter
    if W <= H:
        new_w = shorter
        new_h = int(math.floor(shorter * float(H) / float(W)))
    else:
        new_h = shorter
        new_w = int(math.floor(shorter * float(W) / float(H)))
    return max(1, int(new_h)), max(1, int(new_w))


def infer_preproc_geom(clip_processor):
    with torch.no_grad():
        enc = clip_processor(images=_dummy_pil(512), return_tensors="pt")
        pv = enc["pixel_values"]
        S = int(pv.shape[-1])

    geom = dict(
        output_size=int(S),
        crop_size=int(S),
        center_crop=True,
        resize_mode="shorter",
        resize_shorter=None,
        resize_hw=None,
    )

    try:
        if hasattr(clip_processor, "backend") and clip_processor.backend == "hf":
            hf = getattr(clip_processor, "hf", None)
            ip = getattr(hf, "image_processor", hf)

            cs = getattr(ip, "crop_size", None)
            if isinstance(cs, dict):
                h = cs.get("height", None) or cs.get("shortest_edge", None) or cs.get("width", None)
                if h is not None:
                    geom["crop_size"] = int(h)
                    geom["center_crop"] = True
            elif isinstance(cs, int):
                geom["crop_size"] = int(cs)
                geom["center_crop"] = True

            sz = getattr(ip, "size", None)
            if isinstance(sz, dict):
                if "shortest_edge" in sz:
                    geom["resize_mode"] = "shorter"
                    geom["resize_shorter"] = int(sz["shortest_edge"])
                elif ("height" in sz) and ("width" in sz):
                    geom["resize_mode"] = "exact"
                    geom["resize_hw"] = (int(sz["height"]), int(sz["width"]))
            elif isinstance(sz, int):
                geom["resize_mode"] = "shorter"
                geom["resize_shorter"] = int(sz)
    except Exception:
        pass

    try:
        if hasattr(clip_processor, "backend") and clip_processor.backend == "openclip":
            tr = getattr(clip_processor, "transform", None)
            ts = getattr(tr, "transforms", None)
            ts = ts if isinstance(ts, (list, tuple)) else ([tr] if tr is not None else [])
            for t in ts:
                name = t.__class__.__name__.lower()
                if name == "resize":
                    sz = getattr(t, "size", None)
                    if isinstance(sz, int):
                        geom["resize_mode"] = "shorter"
                        geom["resize_shorter"] = int(sz)
                        geom["resize_hw"] = None
                    elif isinstance(sz, (tuple, list)) and len(sz) == 2:
                        geom["resize_mode"] = "exact"
                        geom["resize_hw"] = (int(sz[0]), int(sz[1]))
                        geom["resize_shorter"] = None
                elif name == "centercrop":
                    cs = getattr(t, "size", None)
                    if isinstance(cs, int):
                        geom["crop_size"] = int(cs)
                        geom["center_crop"] = True
                    elif isinstance(cs, (tuple, list)) and len(cs) == 2:
                        geom["crop_size"] = int(cs[0])
                        geom["center_crop"] = True
    except Exception:
        pass

    geom["output_size"] = int(S)
    geom["crop_size"] = int(geom.get("crop_size", S))
    return geom


def map_bbox_to_preproc_space(bbox, H: int, W: int, geom: dict):
    S = int(geom["output_size"])
    x1, y1, x2, y2 = map(float, bbox[:4])

    mode = geom.get("resize_mode", "shorter")
    if mode == "exact" and geom.get("resize_hw", None) is not None:
        new_h, new_w = map(int, geom["resize_hw"])
        sx = float(new_w) / float(W)
        sy = float(new_h) / float(H)
    elif mode == "shorter" and geom.get("resize_shorter", None) is not None:
        new_h, new_w = _resize_dims_keep_ratio(H, W, geom["resize_shorter"])
        sx = float(new_w) / float(W)
        sy = float(new_h) / float(H)
    else:
        new_h, new_w = int(H), int(W)
        sx = sy = 1.0

    if geom.get("center_crop", True):
        off_x = 0.5 * (float(new_w) - float(S))
        off_y = 0.5 * (float(new_h) - float(S))
    else:
        off_x = off_y = 0.0

    mx1 = int(round(x1 * sx - off_x))
    my1 = int(round(y1 * sy - off_y))
    mx2 = int(round(x2 * sx - off_x))
    my2 = int(round(y2 * sy - off_y))

    mx1 = max(0, min(S, mx1))
    my1 = max(0, min(S, my1))
    mx2 = max(0, min(S, mx2))
    my2 = max(0, min(S, my2))
    return mx1, my1, mx2, my2, sx, sy


def ensure_rgb3(img: np.ndarray):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return img[:, :, :3]
    return img


def full_image_and_bbox(img_rgb: np.ndarray, bb_xyxy):
    """Return the full image and a clamped bbox in the full-image coordinate frame."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, bb_xyxy[:4])
    x1 = max(0.0, min(float(W - 1), x1))
    y1 = max(0.0, min(float(H - 1), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None, None
    return img_rgb, [x1, y1, x2, y2]


def list_images(img_dir: str, exts=None, recursive: bool = True):
    if exts is None:
        exts = IMG_EXTS
    if not os.path.isdir(img_dir):
        return []
    all_imgs = []
    if recursive:
        for e in exts:
            all_imgs += glob.glob(os.path.join(img_dir, "**", f"*.{e}"), recursive=True)
            all_imgs += glob.glob(os.path.join(img_dir, "**", f"*.{e.upper()}"), recursive=True)
    else:
        for e in exts:
            all_imgs += glob.glob(os.path.join(img_dir, f"*.{e}"))
            all_imgs += glob.glob(os.path.join(img_dir, f"*.{e.upper()}"))
    return sorted(list(set(all_imgs)))


def extract_person_bboxes(det_result, thr: float = SCORE_THR, person_id: int = 0):
    if det_result is None:
        return []
    if isinstance(det_result, tuple):
        det_result = det_result[0]
    if not isinstance(det_result, (list, tuple)):
        return []
    person_bbs = det_result[person_id] if person_id < len(det_result) else []
    out = []
    for bb in person_bbs:
        try:
            if float(bb[4]) >= float(thr):
                out.append(bb)
        except Exception:
            continue
    return out


_DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a blurry photo of a {}.",
    "a bright photo of a {}.",
]


def init_clip_batch(CLIP_MODEL_ID: str, CLIP_CKPT: str = None):
    global _clip_model, _clip_processor, _clip_class_embs, _clip_logit_scale
    global _PREPROC_GEOM, CLIP_INPUT_SIZE

    if AutoCLIPModel is None or AutoCLIPProcessor is None:
        raise RuntimeError("[INIT] AutoCLIPModel/AutoCLIPProcessor import failed. Check scripts.infer_clips")

    try:
        if hasattr(AutoCLIPProcessor, "from_pretrained"):
            _clip_processor = AutoCLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        else:
            _clip_processor = AutoCLIPProcessor(CLIP_MODEL_ID)
    except Exception as e:
        raise RuntimeError(f"[INIT] build processor failed: {repr(e)}")

    try:
        if hasattr(AutoCLIPModel, "from_pretrained"):
            _clip_model = AutoCLIPModel.from_pretrained(CLIP_MODEL_ID)
        else:
            _clip_model = AutoCLIPModel(CLIP_MODEL_ID)
    except Exception as e:
        raise RuntimeError(f"[INIT] build model failed: {repr(e)}")

    if CLIP_CKPT:
        ckpt_path = _maybe_copy_ckpt_to_local(CLIP_CKPT)
        if os.path.isfile(ckpt_path):
            try:
                before_sd = {}
                for k, v in _clip_model.state_dict().items():
                    if torch.is_tensor(v):
                        before_sd[k] = v.detach().cpu().clone()
                    if len(before_sd) >= 20:
                        break

                sd = torch.load(ckpt_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                if not isinstance(sd, dict):
                    raise RuntimeError(f"checkpoint is not a state_dict dict, got type={type(sd)}")

                print(f"[CKPT] path: {ckpt_path}", flush=True)
                print(f"[CKPT] top_level_keys: {list(sd.keys())[:20]}", flush=True)
                print(f"[CKPT] num_keys_in_ckpt: {len(sd.keys())}", flush=True)

                if hasattr(_clip_model, "load_finetuned_state_dict"):
                    msg = _clip_model.load_finetuned_state_dict(sd, strict=False)
                    print(f"[CKPT] load_finetuned_state_dict return: {msg}", flush=True)
                    if isinstance(msg, (tuple, list)) and len(msg) >= 2:
                        missing_keys = list(msg[0]) if msg[0] is not None else []
                        unexpected_keys = list(msg[1]) if msg[1] is not None else []
                    else:
                        missing_keys = []
                        unexpected_keys = []
                else:
                    if any(k.startswith("module.") for k in sd.keys()):
                        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                    msg = _clip_model.load_state_dict(sd, strict=False)
                    missing_keys = list(getattr(msg, "missing_keys", []))
                    unexpected_keys = list(getattr(msg, "unexpected_keys", []))

                n_missing = len(missing_keys)
                n_unexpected = len(unexpected_keys)
                print(f"[CKPT] missing_keys: {n_missing}", flush=True)
                print(f"[CKPT] unexpected_keys: {n_unexpected}", flush=True)
                if n_missing > 0:
                    print(f"[CKPT] first_missing_keys: {missing_keys[:20]}", flush=True)
                if n_unexpected > 0:
                    print(f"[CKPT] first_unexpected_keys: {unexpected_keys[:20]}", flush=True)

                changed_cnt = 0
                checked_cnt = 0
                after_sd = _clip_model.state_dict()
                for k, v_before in before_sd.items():
                    if k in after_sd and torch.is_tensor(after_sd[k]):
                        v_after = after_sd[k].detach().cpu()
                        checked_cnt += 1
                        if not torch.equal(v_before, v_after):
                            changed_cnt += 1
                print(f"[CKPT] sampled_param_changed: {changed_cnt}/{checked_cnt}", flush=True)
                if changed_cnt == 0:
                    print("[CKPT][WARN] sampled params did not change at all -> very likely NOT loading finetuned weights effectively.", flush=True)
                else:
                    print("[CKPT][OK] sampled params changed -> finetuned weights likely took effect.", flush=True)
                if n_missing > 50 or n_unexpected > 50:
                    print("[CKPT][WARN] too many missing/unexpected keys -> checkpoint/model mismatch is likely.", flush=True)
                log(f"[CKPT] loaded CLIP ckpt: {ckpt_path}", level=0)
            except Exception as e:
                log(f"[WARN] load CLIP_CKPT failed (ignored): {repr(e)}", level=0)

    _clip_model = _clip_model.to(DEVICE)
    _clip_model.eval()
    if DEVICE.startswith("cuda"):
        try:
            _clip_model = _clip_model.half()
        except Exception:
            pass

    _PREPROC_GEOM = infer_preproc_geom(_clip_processor)
    CLIP_INPUT_SIZE = int(_PREPROC_GEOM["output_size"])

    prompts = []
    for c in CLASSES:
        for t in _DEFAULT_TEMPLATES:
            prompts.append(t.format(c))

    try:
        enc = _clip_processor(text=prompts, return_tensors="pt", padding=True)
    except Exception as e:
        raise RuntimeError(f"[INIT] processor(text=...) failed: {repr(e)}")

    for k in list(enc.keys()):
        if torch.is_tensor(enc[k]):
            enc[k] = enc[k].to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        if hasattr(_clip_model, "get_text_features"):
            txt = _clip_model.get_text_features(**enc)
        elif hasattr(_clip_model, "encode_text"):
            if "input_ids" not in enc:
                raise RuntimeError("[INIT] encode_text backend but no input_ids in processor output")
            txt = _clip_model.encode_text(enc["input_ids"])
        else:
            raise RuntimeError("[INIT] CLIP model has no get_text_features/encode_text")

        txt = txt.float()
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        C = len(CLASSES)
        T = len(_DEFAULT_TEMPLATES)
        txt = txt.view(C, T, -1).mean(dim=1)
        txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        _clip_class_embs = txt.detach().to(DEVICE, dtype=torch.float32)

        if hasattr(_clip_model, "logit_scale") and torch.is_tensor(_clip_model.logit_scale):
            _clip_logit_scale = _clip_model.logit_scale.detach().float().exp().to(DEVICE)
        else:
            _clip_logit_scale = torch.tensor(1.0 / 0.07, device=DEVICE, dtype=torch.float32)

    log(
        f"[INIT] CLIP ready: classes={len(CLASSES)} feat_dim={int(_clip_class_embs.shape[1])} input_S={CLIP_INPUT_SIZE} "
        f"logit_scale={float(_clip_logit_scale.item() if torch.is_tensor(_clip_logit_scale) else _clip_logit_scale):.3f}",
        level=0
    )


@torch.inference_mode()
def clip_forward_full_image_single(rgb_img: np.ndarray, bbs=None):
    global _clip_model, _clip_processor, _clip_class_embs, _clip_logit_scale
    if rgb_img is None:
        return None, None
    if _clip_model is None or _clip_processor is None or _clip_class_embs is None or _clip_logit_scale is None:
        raise RuntimeError("CLIP not initialized. Call init_clip_batch() first.")

    rgb_img = ensure_rgb3(rgb_img)
    H, W = rgb_img.shape[:2]

    if bbs is None:
        bbs_list = []
    elif isinstance(bbs, np.ndarray):
        bbs_list = bbs.tolist() if bbs.size > 0 else []
    else:
        bbs_list = list(bbs)

    if not bbs_list:
        bbs_list = [[0, 0, W - 1, H - 1, 1.0]]

    if rgb_img is None or rgb_img.size == 0:
        return None, None

    enc = _clip_processor(images=Image.fromarray(rgb_img, mode="RGB"), return_tensors="pt")
    batch = enc["pixel_values"].to(DEVICE, non_blocking=True)
    batch = batch.to(next(_clip_model.parameters()).dtype)

    img_feat = _clip_model.get_image_features(pixel_values=batch)
    img_feat = img_feat.float()
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    feat = img_feat

    logits = (_clip_logit_scale.float()) * (feat @ _clip_class_embs.float().t())
    logits_np = logits.detach().cpu().numpy().astype(np.float32)[0]
    return feat.squeeze(0), logits_np


@torch.inference_mode()
def clip_forward_full_image_per_bbox_logits(rgb_img: np.ndarray, bbs):
    global _clip_model, _clip_processor, _clip_class_embs, _clip_logit_scale
    if rgb_img is None:
        return [], None
    if _clip_model is None or _clip_processor is None or _clip_class_embs is None or _clip_logit_scale is None:
        raise RuntimeError("CLIP not initialized. Call init_clip_batch() first.")

    rgb_img = ensure_rgb3(rgb_img)
    if rgb_img is None:
        return [], None

    if bbs is None:
        bbs_list = []
    elif isinstance(bbs, np.ndarray):
        bbs_list = bbs.tolist() if bbs.size > 0 else []
    else:
        bbs_list = list(bbs)

    if not bbs_list:
        return [], None

    valid_bbs = []
    for bb in bbs_list:
        try:
            _, bb_full = full_image_and_bbox(rgb_img, bb)
        except Exception:
            bb_full = None
        if bb_full is None:
            continue
        valid_bbs.append(bb)

    if not valid_bbs:
        return [], None

    model_dtype = next(_clip_model.parameters()).dtype
    enc = _clip_processor(images=Image.fromarray(rgb_img, mode="RGB"), return_tensors="pt")
    batch = enc["pixel_values"].to(DEVICE, non_blocking=True).to(model_dtype)
    img_feat = _clip_model.get_image_features(pixel_values=batch)
    img_feat = img_feat.float()
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    logits = (_clip_logit_scale.float()) * (img_feat @ _clip_class_embs.float().t())
    logits_one = logits.detach().cpu().numpy().astype(np.float32)
    logits_np = np.repeat(logits_one, repeats=len(valid_bbs), axis=0)
    return valid_bbs, logits_np


def build_eval_set_sequential_top1_person(img_dir: str, det_model, max_samples: int, recursive: bool = True, person_id: int = 0, thr: float = SCORE_THR):
    if inference_detector is None:
        raise RuntimeError("[DATA] mmdet inference_detector is None (mmdet import failed?)")
    if det_model is None:
        raise RuntimeError("[DATA] det_model is None")

    all_imgs = list_images(img_dir, exts=IMG_EXTS, recursive=recursive)
    log(f"[DATA] Scanning {len(all_imgs)} images...", level=0)

    person_idx = int(LABEL2IDX.get("person", 0))
    kept_paths, bboxes_list, true_labels = [], [], []

    for p in all_imgs:
        if len(kept_paths) >= int(max_samples):
            break
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]

        try:
            det_result = inference_detector(det_model, bgr)
        except Exception as e:
            log(f"[DET] inference failed: {p} err={repr(e)}", level=1)
            continue

        person_bbs = extract_person_bboxes(det_result, thr=float(thr), person_id=int(person_id))
        if not person_bbs:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid_bbs, logits_np = clip_forward_full_image_per_bbox_logits(rgb, person_bbs)
        if logits_np is None or len(valid_bbs) == 0:
            continue

        top1 = logits_np.argmax(axis=1).astype(np.int32)
        for bb, t1 in zip(valid_bbs, top1):
            if int(t1) == int(person_idx):
                kept_paths.append(p)
                bboxes_list.append([bb])
                true_labels.append("person")
                if len(kept_paths) >= int(max_samples):
                    break

    log(f"[DATA] Screening done: kept bbox-samples = {len(kept_paths)} (max_samples={max_samples})", level=0)
    return kept_paths, bboxes_list, true_labels


@torch.no_grad()
def make_patch_color_tensor_norm(clip_processor, patch_color_rgb, device, dtype, out_size: int):
    S = int(out_size)
    sz = max(256, S)
    arr = np.zeros((sz, sz, 3), dtype=np.uint8)
    arr[:, :] = np.array(patch_color_rgb, dtype=np.uint8).reshape(1, 1, 3)
    pil = Image.fromarray(arr, mode="RGB")
    enc = clip_processor(images=pil, return_tensors="pt")
    pv = enc["pixel_values"]
    v = pv.mean(dim=(2, 3), keepdim=True)
    return v.to(device=device, dtype=dtype)

def decode_theta(theta: np.ndarray, G: int):
    gate_end = int(G * G)
    return theta[0:gate_end], theta[gate_end:]


def make_base_nodes(G: int):
    xs = np.linspace(0, STANDARD_SIZE - 1, G + 1)
    ys = np.linspace(0, STANDARD_SIZE - 1, G + 1)
    nodes = np.zeros((G + 1, G + 1, 2), dtype=np.float32)
    for r in range(G + 1):
        for c in range(G + 1):
            nodes[r, c] = [xs[c], ys[r]]
    return nodes


def build_edge_list(G: int):
    edges = []
    for r in range(G + 1):
        for c in range(G):
            edges.append(((r, c), (r, c + 1)))
    for r in range(G):
        for c in range(G + 1):
            edges.append(((r, c), (r + 1, c)))
    return edges


def get_cached_geometry(G: int):
    if G not in _BASE_NODES_CACHE:
        _BASE_NODES_CACHE[G] = make_base_nodes(G)
        _EDGE_LIST_CACHE[G] = build_edge_list(G)
    return _BASE_NODES_CACHE[G], _EDGE_LIST_CACHE[G]


def build_edge_set_from_gates(cell_gates_flat: np.ndarray, G: int):
    cell_gates = cell_gates_flat.reshape((G, G))
    edge_set = set()
    for cr in range(G):
        for cc in range(G):
            if cell_gates[cr, cc] < GATE_THR:
                continue
            corners = [(cr, cc), (cr, cc + 1), (cr + 1, cc + 1), (cr + 1, cc)]
            cell_edges = [
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[2], corners[3]),
                (corners[3], corners[0]),
            ]
            for (n1, n2) in cell_edges:
                key = (min(n1, n2), max(n1, n2))
                edge_set.add(key)
    return edge_set


def apply_offsets_to_edges(base_nodes, edges, offsets, G: int, max_curv_ratio: float = MAX_CURV_RATIO):
    cell_size = (STANDARD_SIZE - 1) / float(G)
    max_offset = float(max_curv_ratio) * cell_size
    controls = {}
    for idx, ((r1, c1), (r2, c2)) in enumerate(edges):
        p1 = base_nodes[r1, c1]
        p2 = base_nodes[r2, c2]
        mid = (p1 + p2) / 2.0
        d = p2 - p1
        perp = np.array([-d[1], d[0]], dtype=np.float32)
        perp_len = np.linalg.norm(perp)
        if perp_len > 1e-6:
            perp /= perp_len
        off_val = float(offsets[idx]) * max_offset
        ctrl = mid + perp * off_val
        controls[((r1, c1), (r2, c2))] = ctrl
    return controls


def quadratic_bezier(p0, p1, p2, num: int = 24):
    t = np.linspace(0, 1, num, dtype=np.float32).reshape(-1, 1)
    omt = (1 - t)
    pts = (omt * omt) * p0 + 2 * (omt * t) * p1 + (t * t) * p2
    return pts.astype(np.float32)


def get_edge_curve(n1, n2, controls, base_nodes, num: int = 24):
    key = (n1, n2) if (n1, n2) in controls else (n2, n1)
    ctrl = controls[key] if key in controls else (base_nodes[n1[0], n1[1]] + base_nodes[n2[0], n2[1]]) / 2
    p0 = base_nodes[n1[0], n1[1]]
    p2 = base_nodes[n2[0], n2[1]]
    return quadratic_bezier(p0, ctrl, p2, num=num)


MASTER_MASK_CACHE_MAXSIZE = 256
THETA_QUANT = 1e-4


def _theta_to_key(theta: np.ndarray) -> bytes:
    q = np.round(theta / THETA_QUANT).astype(np.int32, copy=False)
    return q.tobytes()


def _key_to_theta(theta_key: bytes, D: int) -> np.ndarray:
    q = np.frombuffer(theta_key, dtype=np.int32, count=int(D))
    return q.astype(np.float32) * THETA_QUANT


@lru_cache(maxsize=MASTER_MASK_CACHE_MAXSIZE)
def _generate_master_mask_cached(theta_key: bytes, G: int, size: int, D: int):
    theta = _key_to_theta(theta_key, int(D))
    return _generate_master_mask_impl(theta, G=int(G), size=int(size))


_MASK_ID_TO_THETAKEY = {}
_MASK_ID_TO_THETAKEY_MAX = 2048
_MASK_RESIZE_LOCK = threading.Lock()


def generate_master_mask(theta, G: int = 5, size: int = MASTER_SIZE):
    theta = np.asarray(theta, dtype=np.float32)
    key = _theta_to_key(theta)
    mask = _generate_master_mask_cached(key, int(G), int(size), int(theta.size))
    try:
        _MASK_ID_TO_THETAKEY[id(mask)] = key
        if len(_MASK_ID_TO_THETAKEY) > _MASK_ID_TO_THETAKEY_MAX:
            remove_n = len(_MASK_ID_TO_THETAKEY) - _MASK_ID_TO_THETAKEY_MAX
            for _ in range(remove_n):
                _MASK_ID_TO_THETAKEY.pop(next(iter(_MASK_ID_TO_THETAKEY)), None)
    except Exception:
        pass
    return mask


def _generate_master_mask_impl(theta: np.ndarray, G: int = 5, size: int = MASTER_SIZE):
    cell_gates_flat, edge_offsets_flat = decode_theta(theta, G)
    total_ratio = 1.0 + 2.0 * float(EXPANSION_RATIO)
    core_size_virtual = float(size) / total_ratio
    pad_virtual = core_size_virtual * float(EXPANSION_RATIO)
    mask_hi = np.zeros((int(size), int(size)), dtype=np.uint8)
    scale = core_size_virtual / float(STANDARD_SIZE)
    cell_size_on_master = core_size_virtual / float(G)
    thickness_px = int(round(cell_size_on_master * float(THICKNESS_TO_CELL_RATIO)))
    thickness_px = max(1, thickness_px)

    base_nodes, edges = get_cached_geometry(G)
    controls = apply_offsets_to_edges(base_nodes, edges, edge_offsets_flat, G)
    edge_set = build_edge_set_from_gates(cell_gates_flat, G)
    num_pts = int(FAST_RENDER_LINE_SAMPLES) * 2

    pts_to_draw = []
    for (n1, n2) in edge_set:
        curve_std = get_edge_curve(n1, n2, controls, base_nodes, num=num_pts)
        curve_hi = curve_std * scale
        curve_hi[:, 0] += pad_virtual
        curve_hi[:, 1] += pad_virtual
        pts = np.round(curve_hi).astype(np.int32).reshape(-1, 1, 2)
        pts_to_draw.append(pts)

    if pts_to_draw:
        cv2.polylines(mask_hi, pts_to_draw, isClosed=False, color=255, thickness=thickness_px, lineType=cv2.LINE_AA)

    if FAST_RENDER_BLUR_KSIZE > 0:
        k = int(FAST_RENDER_BLUR_KSIZE)
        if k % 2 == 0:
            k += 1
        mask_hi = cv2.GaussianBlur(mask_hi, (k, k), 0)

    alpha_master = mask_hi.astype(np.float32) / 255.0
    if float(ALPHA_GAMMA) != 1.0:
        alpha_master = np.power(alpha_master, float(ALPHA_GAMMA))
    return alpha_master


RESIZE_CACHE_MAXSIZE = 512
_MASK_RESIZE_CACHE = OrderedDict()


def _get_master_key(master_mask: np.ndarray):
    try:
        return _MASK_ID_TO_THETAKEY.get(id(master_mask), id(master_mask))
    except Exception:
        return id(master_mask)


def _mask_resize_cache_get(master_mask: np.ndarray, target_w: int, target_h: int, interp: int):
    mkey = _get_master_key(master_mask)
    key = (mkey, int(target_w), int(target_h), int(interp))
    with _MASK_RESIZE_LOCK:
        v = _MASK_RESIZE_CACHE.get(key, None)
        if v is not None:
            _MASK_RESIZE_CACHE.move_to_end(key)
    return v


def _mask_resize_cache_put(master_mask: np.ndarray, target_w: int, target_h: int, interp: int, resized: np.ndarray):
    mkey = _get_master_key(master_mask)
    key = (mkey, int(target_w), int(target_h), int(interp))
    with _MASK_RESIZE_LOCK:
        _MASK_RESIZE_CACHE[key] = resized
        _MASK_RESIZE_CACHE.move_to_end(key)
        while len(_MASK_RESIZE_CACHE) > RESIZE_CACHE_MAXSIZE:
            _MASK_RESIZE_CACHE.popitem(last=False)


def _resize_master_cached(master_mask: np.ndarray, target_w: int, target_h: int, interp: int):
    cached = _mask_resize_cache_get(master_mask, target_w, target_h, interp)
    if cached is not None:
        return cached
    resized = cv2.resize(master_mask, (int(target_w), int(target_h)), interpolation=int(interp))
    _mask_resize_cache_put(master_mask, target_w, target_h, interp, resized)
    return resized


def apply_master_mask_fast(img_rgb: np.ndarray, bbox, master_mask: np.ndarray):
    if master_mask is None:
        return img_rgb

    x1, y1, x2, y2 = map(float, bbox[:4])
    img_H, img_W = img_rgb.shape[:2]
    bh = max(1.0, y2 - y1)
    raw_side = bh / float(ROI_CORE_DIV)
    target_core = max(4.0, raw_side)
    target_canvas = target_core * (1.0 + 2.0 * float(EXPANSION_RATIO))
    target_w = int(round(target_canvas))
    target_h = int(round(target_canvas))
    if target_w < 1 or target_h < 1:
        return img_rgb

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    paste_x1 = int(round(cx - target_w / 2.0))
    paste_y1 = int(round(cy - target_h / 2.0))
    paste_x2 = paste_x1 + target_w
    paste_y2 = paste_y1 + target_h

    ix1, iy1 = max(0, paste_x1), max(0, paste_y1)
    ix2, iy2 = min(img_W, paste_x2), min(img_H, paste_y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return img_rgb

    interp = cv2.INTER_LINEAR
    mask_f = _resize_master_cached(master_mask, target_w, target_h, interp)
    mx1, my1 = ix1 - paste_x1, iy1 - paste_y1
    mx2, my2 = mx1 + (ix2 - ix1), my1 + (iy2 - iy1)
    mask_crop = mask_f[my1:my2, mx1:mx2]
    roi_h, roi_w = (iy2 - iy1), (ix2 - ix1)
    if mask_crop.shape[0] != roi_h or mask_crop.shape[1] != roi_w:
        mask_crop = cv2.resize(mask_crop, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    roi = img_rgb[iy1:iy2, ix1:ix2].astype(np.float32)
    s = float(np.clip(DARKEN_STRENGTH, 0.0, 1.0))
    a = np.clip(s * mask_crop, 0.0, 1.0).astype(np.float32)[..., None]
    color = np.array(PATCH_COLOR_RGB, dtype=np.float32).reshape(1, 1, 3)
    blended = roi * (1.0 - a) + color * a
    img_rgb[iy1:iy2, ix1:ix2] = blended.astype(np.uint8)
    return img_rgb


def render_mesh_on_bbox(img_rgb: np.ndarray, bbox, theta=None, G: int = 5, master_mask=None):
    if master_mask is not None:
        return apply_master_mask_fast(img_rgb, bbox, master_mask)
    return img_rgb


class FastGPUFullImageEvaluator:
    def __init__(self, imgs_rgb, bboxes_list, clip_model, clip_processor, device):
        if _PREPROC_GEOM is None:
            raise RuntimeError("_PREPROC_GEOM is None. Call init_clip_batch() first.")
        self.device = device
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.geom = dict(_PREPROC_GEOM)
        self.S = int(self.geom["output_size"])

        pil_images = []
        self.meta_data = []
        for i, img in enumerate(imgs_rgb):
            img = ensure_rgb3(img)
            if img is None:
                raise ValueError("imgs_rgb contains None")
            H, W = img.shape[:2]
            bbs = bboxes_list[i] if bboxes_list is not None else []
            if bbs is None or len(bbs) == 0:
                bbs = [[0, 0, W - 1, H - 1, 1.0]]

            bb = bbs[0]
            x1, y1, x2, y2 = map(float, bb[:4])
            full_rgb, rel_bb = full_image_and_bbox(img, [x1, y1, x2, y2])
            if full_rgb is None or full_rgb.size == 0 or rel_bb is None:
                full_rgb = img
                rel_bb = [0, 0, W - 1, H - 1]

            pil_images.append(Image.fromarray(full_rgb, mode="RGB"))
            mx1, my1, mx2, my2, _, _ = map_bbox_to_preproc_space(rel_bb, H, W, self.geom)
            bw_m = max(1.0, float(mx2 - mx1))
            bh_m = max(1.0, float(my2 - my1))
            self.meta_data.append([(mx1, my1, mx2, my2, bw_m, bh_m)])

        log(f"[GPU-EVAL] Caching {len(pil_images)} processed full images...", level=0)
        all_tensors = []
        BS = 32
        for st in range(0, len(pil_images), BS):
            chunk = pil_images[st:st + BS]
            enc = self.clip_processor(images=chunk, return_tensors="pt")
            all_tensors.append(enc["pixel_values"])
        pv = torch.cat(all_tensors, dim=0) if all_tensors else torch.empty((0, 3, self.S, self.S))
        self.clean_imgs_gpu = pv.to(self.device, dtype=torch.float16, non_blocking=True)
        self._mask_tensor_cache = OrderedDict()
        self._patch_color_cache = {}

    def clear_mask_cache(self):
        try:
            self._mask_tensor_cache.clear()
        except Exception:
            pass

    def _get_mask_tensor_cached(self, theta, G: int, master_size: int):
        MASK_TENSOR_CACHE_MAX = 128
        theta = np.asarray(theta, dtype=np.float32)
        theta_key = _theta_to_key(theta)
        key = (theta_key, int(G), int(master_size))
        v = self._mask_tensor_cache.get(key, None)
        if v is not None:
            self._mask_tensor_cache.move_to_end(key)
            return v
        master_mask_np = generate_master_mask(theta, G=int(G), size=int(master_size))
        mask_tensor = torch.from_numpy(master_mask_np).to(self.device).float().unsqueeze(0).unsqueeze(0)
        self._mask_tensor_cache[key] = mask_tensor
        self._mask_tensor_cache.move_to_end(key)
        while len(self._mask_tensor_cache) > MASK_TENSOR_CACHE_MAX:
            self._mask_tensor_cache.popitem(last=False)
        return mask_tensor

    def _get_patch_color_tensor_norm(self, patch_color):
        key = tuple(int(x) for x in patch_color)
        v = self._patch_color_cache.get(key, None)
        if v is not None:
            return v
        t = make_patch_color_tensor_norm(
            self.clip_processor,
            patch_color_rgb=patch_color,
            device=self.device,
            dtype=torch.float16,
            out_size=self.S,
        )
        self._patch_color_cache[key] = t
        return t

    def _apply_patch_batch(self, indices, theta, G: int, master_size: int, patch_color, darken_strength: float):
        S = int(self.S)
        mask_tensor = self._get_mask_tensor_cached(theta, G=int(G), master_size=int(master_size))
        indices_tensor = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        batch_imgs = self.clean_imgs_gpu.index_select(0, indices_tensor).clone()
        color_tensor = self._get_patch_color_tensor_norm(patch_color)
        budget_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        _resize_cache = {}

        for i_local, idx_global in enumerate(indices):
            bboxes = self.meta_data[int(idx_global)]
            if not bboxes:
                continue
            for (mx1, my1, mx2, my2, bw_m, bh_m) in bboxes:
                raw_side = float(bh_m) / float(ROI_CORE_DIV)
                core_w = max(4.0, raw_side)
                core_h = max(4.0, raw_side)
                target_w = int(round(core_w * (1.0 + 2.0 * float(EXPANSION_RATIO))))
                target_h = int(round(core_h * (1.0 + 2.0 * float(EXPANSION_RATIO))))
                if target_w < 2 or target_h < 2:
                    continue

                cx = 0.5 * (float(mx1) + float(mx2))
                cy = 0.5 * (float(my1) + float(my2))
                paste_x1 = int(round(cx - target_w / 2.0))
                paste_y1 = int(round(cy - target_h / 2.0))
                paste_x2 = paste_x1 + target_w
                paste_y2 = paste_y1 + target_h

                ix1, iy1 = max(0, paste_x1), max(0, paste_y1)
                ix2, iy2 = min(S, paste_x2), min(S, paste_y2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                key = (target_h, target_w)
                small_mask_full = _resize_cache.get(key, None)
                if small_mask_full is None:
                    small_mask_full = F.interpolate(mask_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
                    _resize_cache[key] = small_mask_full

                mx_off1 = ix1 - paste_x1
                my_off1 = iy1 - paste_y1
                mx_off2 = mx_off1 + (ix2 - ix1)
                my_off2 = my_off1 + (iy2 - iy1)
                m = small_mask_full[:, :, my_off1:my_off2, mx_off1:mx_off2]
                if m.numel() == 0:
                    continue
                s = float(np.clip(darken_strength, 0.0, 1.0))
                m = (m * s).clamp_(0.0, 1.0)
                roi = batch_imgs[i_local:i_local + 1, :, iy1:iy2, ix1:ix2]
                blended = roi * (1.0 - m) + color_tensor * m
                batch_imgs[i_local:i_local + 1, :, iy1:iy2, ix1:ix2] = blended
                budget_loss_sum = budget_loss_sum + _compute_budget_loss_from_roi(
                    roi_adv=blended,
                    alpha=m,
                    bbox_area=float(max(1.0, bw_m * bh_m)),
                )
        return batch_imgs, budget_loss_sum

    def forward_batch_full_image(self, indices, theta, G: int, master_size: int, patch_color, darken_strength: float):
        batch_imgs_adv, budget_loss_sum = self._apply_patch_batch(
            indices=indices,
            theta=theta,
            G=int(G),
            master_size=int(master_size),
            patch_color=patch_color,
            darken_strength=float(darken_strength),
        )
        with torch.inference_mode():
            model_dtype = next(self.clip_model.parameters()).dtype
            model_input = batch_imgs_adv.to(model_dtype)
            img_feat = self.clip_model.get_image_features(pixel_values=model_input)
            img_feat = img_feat.float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return img_feat, budget_loss_sum


def _compute_sigma_median(dist2: torch.Tensor) -> float:
    if dist2.numel() == 0:
        return 1.0
    mask = ~torch.eye(dist2.shape[0], device=dist2.device, dtype=torch.bool)
    v = dist2[mask]
    v = v[v.isfinite()]
    if v.numel() == 0:
        return 1.0
    med = torch.median(v)
    return float(torch.clamp(med, min=1e-6).item())


def _affinity_probs(feat: torch.Tensor, sigma: float, eps: float = 1e-8) -> torch.Tensor:
    n = int(feat.shape[0])
    if n <= 1:
        return torch.zeros((n, n), device=feat.device, dtype=torch.float32)
    dist = torch.cdist(feat, feat, p=2.0)
    dist2 = dist * dist
    diag = torch.eye(n, device=feat.device, dtype=torch.bool)
    dist2 = dist2.masked_fill(diag, float("inf"))
    sig = max(float(sigma), 1e-6)
    logits = -dist2 / sig
    logits = logits - torch.max(logits, dim=1, keepdim=True).values
    w = torch.exp(logits).masked_fill(diag, 0.0)
    p = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    return p


def topo_graph_neg_kl(clean_feat: torch.Tensor, adv_feat: torch.Tensor, sigma: float, eps: float = 1e-8) -> float:
    if clean_feat is None or adv_feat is None:
        return 0.0
    n = int(clean_feat.shape[0])
    if n <= 1:
        return 0.0
    P = _affinity_probs(clean_feat, sigma=sigma, eps=eps)
    Q = _affinity_probs(adv_feat, sigma=sigma, eps=eps)
    kl = (P * (torch.log(P + eps) - torch.log(Q + eps))).sum(dim=1).mean()
    return float((-kl).item())


def _grad_mag_2d(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + eps)


def _compute_budget_loss_from_roi(roi_adv: torch.Tensor, alpha: torch.Tensor, bbox_area: float, eps: float = 1e-6) -> torch.Tensor:
    if roi_adv.numel() == 0 or alpha.numel() == 0:
        return torch.zeros((), device=roi_adv.device, dtype=torch.float32)

    gray = roi_adv.mean(dim=1, keepdim=True).float()
    alpha = alpha.float().clamp(0.0, 1.0)

    h, w = int(alpha.shape[-2]), int(alpha.shape[-1])
    ring_pad = max(1, int(round(min(h, w) * 0.05)))
    ksize = 2 * ring_pad + 1
    alpha_dil = F.max_pool2d(alpha, kernel_size=ksize, stride=1, padding=ring_pad)
    beta = (alpha_dil - alpha).clamp(0.0, 1.0)

    alpha_sum = alpha.sum().clamp_min(eps)
    beta_sum = beta.sum().clamp_min(eps)

    mu_alpha = (gray * alpha).sum() / alpha_sum
    mu_beta = (gray * beta).sum() / beta_sum
    var_alpha = (((gray - mu_alpha) ** 2) * alpha).sum() / alpha_sum
    var_beta = (((gray - mu_beta) ** 2) * beta).sum() / beta_sum
    sigma_alpha = torch.sqrt(var_alpha.clamp_min(0.0) + eps)
    sigma_beta = torch.sqrt(var_beta.clamp_min(0.0) + eps)
    l_therm = (mu_alpha - mu_beta).pow(2) + (sigma_alpha - sigma_beta).pow(2)

    gamma = _grad_mag_2d(alpha)
    g_adv = _grad_mag_2d(gray)
    boundary_avg = (gamma * g_adv).sum() / gamma.sum().clamp_min(eps)
    bg_grad_avg = (beta * g_adv).sum() / beta_sum
    l_edge = torch.relu(boundary_avg - bg_grad_avg)

    alpha_bar = alpha.sum() / max(float(bbox_area), 1.0)
    l_area = alpha_bar.pow(2)
    return (l_therm + l_edge + l_area).float()


def _cvar_topk_numpy(vals, q: float) -> float:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return float(np.inf)
    q = float(max(1e-6, min(1.0, q)))
    k = int(max(1, math.ceil(arr.size * q)))
    idx = np.argpartition(arr, arr.size - k)[arr.size - k:]
    return float(arr[idx].mean())


def ev_base_loss_vec(img_feat: torch.Tensor) -> torch.Tensor:
    global _ev_mu, _ev_U
    if (_ev_mu is None) or (_ev_U is None):
        return torch.zeros((img_feat.shape[0],), device=img_feat.device, dtype=img_feat.dtype)
    mu = _ev_mu.to(img_feat.device).view(1, -1)
    U = _ev_U.to(img_feat.device)
    r = (img_feat - mu)
    proj = r @ U
    rec = proj @ U.t()
    perp = r - rec
    e = (perp * perp).sum(dim=1)
    return -e


# =======================
# LOSS-only 排序/比较辅助
# =======================
def pop_stats(asr_cache, fit):
    if fit is None or len(fit) == 0:
        return {}
    l = np.asarray(fit, dtype=np.float32)
    a = np.asarray(asr_cache, dtype=np.float32) if asr_cache is not None and len(asr_cache) == len(l) else None
    best_i = int(np.argmin(l))
    return {
        "best_loss": float(l[best_i]),
        "asr_of_best_loss": float(a[best_i]) if a is not None else None,
        "mean_loss": float(np.mean(l)),
        "med_loss": float(np.median(l)),
        "worst_loss": float(np.max(l)),
        "best_i": best_i,
    }


def rank_by_loss(loss_arr):
    l = np.asarray(loss_arr, dtype=np.float32)
    return np.argsort(l, kind="stable").astype(np.int32)


def best_idx_by_loss(loss_arr) -> int:
    idx = rank_by_loss(loss_arr)
    return int(idx[0]) if idx.size > 0 else 0


def better_loss_only(t_loss: float, c_loss: float, eps_loss: float = LOSS_EPS) -> bool:
    return float(t_loss) < float(c_loss) - float(eps_loss)


# 兼容旧调用名：全部改成 LOSS-first
def rank_by_lexicographic(asr_arr, loss_arr):
    _ = asr_arr
    return rank_by_loss(loss_arr)


def best_idx_by_lexicographic(asr_arr, loss_arr) -> int:
    _ = asr_arr
    return best_idx_by_loss(loss_arr)


def better_lexicographic(t_asr: float, t_loss: float, c_asr: float, c_loss: float, eps_asr: float = ASR_EPS, eps_loss: float = LOSS_EPS) -> bool:
    _ = (t_asr, c_asr, eps_asr)
    return better_loss_only(t_loss, c_loss, eps_loss=eps_loss)


def theta_edge_stats(theta: np.ndarray, G: int):
    gate_end = int(G * G)
    cell_gates = theta[0:gate_end]
    edge_set = build_edge_set_from_gates(cell_gates, G)
    edge_cnt = len(edge_set)
    N_EDGES_LOCAL = 2 * G * (G + 1)
    edge_ratio = edge_cnt / float(N_EDGES_LOCAL)
    return int(edge_cnt), float(edge_ratio)


def schedule_n_eval(progress: float, max_samples: int):
    max_samples = int(max(1, max_samples))
    n1 = int(round(0.32 * max_samples))
    n2 = int(round(0.64 * max_samples))
    n1 = int(min(max_samples, max(1, n1)))
    n2 = int(min(max_samples, max(1, n2)))
    if progress < 0.30:
        return n1
    if progress < 0.70:
        return n2
    return max_samples


def _make_rng_for_eval(seed: int, gen: int, i: int):
    s = (int(seed) * 1000003 + int(gen) * 1009 + int(i) * 9176) & 0xFFFFFFFF
    return np.random.default_rng(s)


def make_gen_superset_indices(total: int, n_superset: int, seed: int, gen: int, mode: str = "rolling"):
    total = int(total)
    n_superset = int(min(max(1, n_superset), total))
    rng = _make_rng_for_eval(seed, gen, 0)
    if n_superset >= total:
        return np.arange(total, dtype=np.int32)
    if mode == "random":
        return rng.choice(total, size=n_superset, replace=False).astype(np.int32)
    start = int(rng.integers(0, total))
    return ((start + np.arange(n_superset)) % total).astype(np.int32)


def flatten_to_bbox_samples(imgs_rgb, bboxes_list, true_labels, kept_paths):
    imgs2, bbs2, lbl2, paths2 = [], [], [], []
    for img, bbs, lbl, p in zip(imgs_rgb, bboxes_list, true_labels, kept_paths):
        if bbs is None:
            continue
        if isinstance(bbs, np.ndarray):
            bbs = bbs.tolist()
        for bb in list(bbs):
            imgs2.append(img)
            bbs2.append([bb])
            lbl2.append(lbl)
            paths2.append(p)
    return imgs2, bbs2, lbl2, paths2


def set_eval_globals(imgs_rgb_in, bboxes_list_in, true_labels_in, G_val=None):
    globals()["imgs_rgb"] = imgs_rgb_in
    globals()["bboxes_list"] = bboxes_list_in
    globals()["true_labels"] = true_labels_in
    if G_val is not None:
        globals()["G"] = int(G_val)


def eval_success_count_indices(theta, indices):
    global gpu_evaluator, _clip_class_embs, _clip_logit_scale
    global _clean_ev_feats, _topo_sigma

    if gpu_evaluator is None:
        raise RuntimeError("gpu_evaluator is None")
    if _clip_class_embs is None or _clip_logit_scale is None:
        raise RuntimeError("CLIP embeddings not ready")
    if indices is None:
        indices = np.arange(len(globals().get("imgs_rgb", [])), dtype=np.int32)

    indices = np.asarray(indices, dtype=np.int32)
    n_eval = int(len(indices))

    base_losses = []
    success_cnt = 0
    n_valid = 0
    adv_feats_all = []
    idx_all = []
    budget_sum_total = torch.zeros((), device=DEVICE, dtype=torch.float32)

    GPU_BATCH = 64
    for b0 in range(0, n_eval, GPU_BATCH):
        b1 = min(n_eval, b0 + GPU_BATCH)
        idx_chunk = indices[b0:b1]
        img_feat, budget_sum_batch = gpu_evaluator.forward_batch_full_image(
            idx_chunk, theta, G,
            master_size=MASTER_SIZE,
            patch_color=PATCH_COLOR_RGB,
            darken_strength=DARKEN_STRENGTH
        )

        logits = (_clip_logit_scale.float()) * (img_feat @ _clip_class_embs.float().t())
        loss_vec = ev_base_loss_vec(img_feat)
        budget_sum_total = budget_sum_total + budget_sum_batch

        y_idx_cpu = np.array([LABEL2IDX.get(true_labels[k], -1) for k in idx_chunk], dtype=np.int64)
        valid_pos = np.where(y_idx_cpu >= 0)[0]
        if valid_pos.size == 0:
            continue

        y_idx_gpu = torch.from_numpy(y_idx_cpu).to(DEVICE)
        valid_mask_gpu = (y_idx_gpu >= 0)
        pred = logits.argmax(dim=-1)
        succ_mask = (pred != y_idx_gpu) & valid_mask_gpu
        success_cnt += int(succ_mask.sum().item())
        n_valid += int(valid_mask_gpu.sum().item())

        loss_vec_cpu = loss_vec.detach().float().cpu().numpy()
        base_losses.extend(loss_vec_cpu[valid_pos].tolist())

        sel = torch.as_tensor(valid_pos, device=DEVICE, dtype=torch.long)
        adv_feats_all.append(img_feat.index_select(0, sel))
        idx_tensor = torch.as_tensor(idx_chunk, device=DEVICE, dtype=torch.long)
        idx_all.append(idx_tensor.index_select(0, sel))

    if n_valid == 0:
        return float(np.inf), 0, 0, 0.0

    L_cvar = _cvar_topk_numpy(base_losses, CVAR_Q)
    L_budget = float(budget_sum_total.item()) / max(1.0, float(n_valid))

    L_topo = 0.0
    if (_clean_ev_feats is not None) and (_topo_sigma is not None) and adv_feats_all:
        adv_feat = torch.cat(adv_feats_all, dim=0)
        idx_cat = torch.cat(idx_all, dim=0)
        clean_feat = _clean_ev_feats.index_select(0, idx_cat)
        L_topo = topo_graph_neg_kl(clean_feat, adv_feat, sigma=float(_topo_sigma))

    fit = float(L_cvar + float(LAMBDA_TOPO) * float(L_topo) + float(LAMBDA_PHYS) * float(L_budget))
    asr = float(success_cnt / float(n_valid))
    return fit, int(success_cnt), int(n_valid), asr


def eval_success_flags_indices(theta, indices):
    global gpu_evaluator, _clip_class_embs, _clip_logit_scale
    global _clean_ev_feats, _topo_sigma

    if gpu_evaluator is None:
        raise RuntimeError("gpu_evaluator is None")
    if indices is None:
        indices = np.arange(len(globals().get("imgs_rgb", [])), dtype=np.int32)

    indices = np.asarray(indices, dtype=np.int32)
    n_eval = int(len(indices))
    base_losses = []
    flags = [False] * n_eval
    n_valid = 0
    success_cnt = 0
    adv_feats_all = []
    idx_all = []
    budget_sum_total = torch.zeros((), device=DEVICE, dtype=torch.float32)

    GPU_BATCH = 64
    for b0 in range(0, n_eval, GPU_BATCH):
        b1 = min(n_eval, b0 + GPU_BATCH)
        idx_chunk = indices[b0:b1]
        img_feat, budget_sum_batch = gpu_evaluator.forward_batch_full_image(
            idx_chunk, theta, G,
            master_size=MASTER_SIZE,
            patch_color=PATCH_COLOR_RGB,
            darken_strength=DARKEN_STRENGTH
        )
        logits = (_clip_logit_scale.float()) * (img_feat @ _clip_class_embs.float().t())
        loss_vec = ev_base_loss_vec(img_feat)
        budget_sum_total = budget_sum_total + budget_sum_batch

        y_idx_cpu = np.array([LABEL2IDX.get(true_labels[k], -1) for k in idx_chunk], dtype=np.int64)
        valid_pos = np.where(y_idx_cpu >= 0)[0]
        if valid_pos.size == 0:
            continue

        y_idx_gpu = torch.from_numpy(y_idx_cpu).to(DEVICE)
        valid_mask_gpu = (y_idx_gpu >= 0)
        pred = logits.argmax(dim=-1)
        succ_mask = (pred != y_idx_gpu) & valid_mask_gpu
        success_cnt += int(succ_mask.sum().item())
        n_valid += int(valid_mask_gpu.sum().item())

        succ_cpu = succ_mask.detach().cpu().numpy().astype(np.bool_)
        for j in range(len(succ_cpu)):
            flags[b0 + j] = bool(succ_cpu[j])

        loss_vec_cpu = loss_vec.detach().float().cpu().numpy()
        base_losses.extend(loss_vec_cpu[valid_pos].tolist())

        sel = torch.as_tensor(valid_pos, device=DEVICE, dtype=torch.long)
        adv_feats_all.append(img_feat.index_select(0, sel))
        idx_tensor = torch.as_tensor(idx_chunk, device=DEVICE, dtype=torch.long)
        idx_all.append(idx_tensor.index_select(0, sel))

    if n_valid == 0:
        return float(np.inf), flags, 0, 0.0

    L_cvar = _cvar_topk_numpy(base_losses, CVAR_Q)
    L_budget = float(budget_sum_total.item()) / max(1.0, float(n_valid))

    L_topo = 0.0
    if (_clean_ev_feats is not None) and (_topo_sigma is not None) and adv_feats_all:
        adv_feat = torch.cat(adv_feats_all, dim=0)
        idx_cat = torch.cat(idx_all, dim=0)
        clean_feat = _clean_ev_feats.index_select(0, idx_cat)
        L_topo = topo_graph_neg_kl(clean_feat, adv_feat, sigma=float(_topo_sigma))

    fit = float(L_cvar + float(LAMBDA_TOPO) * float(L_topo) + float(LAMBDA_PHYS) * float(L_budget))
    asr = float(success_cnt / float(n_valid))
    return fit, flags, int(n_valid), asr


def evaluate_theta_on_indices(theta, imgs_rgb_in, bboxes_list_in, true_labels_in, indices, G: int = 5, return_flags: bool = False):
    total = len(imgs_rgb_in)
    if total <= 0:
        if return_flags:
            return 1.0, 0.0, 0, 0.0, []
        return 1.0, 0.0, 0, 0.0
    set_eval_globals(imgs_rgb_in, bboxes_list_in, true_labels_in, G_val=G)
    indices = np.asarray(indices, dtype=np.int32)
    if return_flags:
        fit, flags, n_valid, asr = eval_success_flags_indices(theta, indices)
        ec, er = theta_edge_stats(theta, G)
        return float(fit), float(asr), int(ec), float(er), flags
    fit, succ, n_valid, asr = eval_success_count_indices(theta, indices)
    ec, er = theta_edge_stats(theta, G)
    return float(fit), float(asr), int(ec), float(er)


def evaluate_theta_full(theta, imgs_rgb_in, bboxes_list_in, true_labels_in, G: int = 5, return_flags: bool = False):
    total = len(imgs_rgb_in)
    if total <= 0:
        if return_flags:
            return 1.0, 0.0, 0, 0.0, []
        return 1.0, 0.0, 0, 0.0
    indices = np.arange(total, dtype=np.int32)
    return evaluate_theta_on_indices(theta, imgs_rgb_in, bboxes_list_in, true_labels_in, indices, G=G, return_flags=return_flags)


def _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen: int, tag: str = ""):
    _ = (asr_cache, fit, best_hist_fit, gen, tag)
    return


class OnlineCalibrator:
    def __init__(self):
        self.residual_scaled = []

    def load_state(self, arr):
        try:
            if arr is None:
                return
            self.residual_scaled = [float(v) for v in list(arr)]
        except Exception:
            self.residual_scaled = []

    def state(self):
        return np.array(self.residual_scaled, dtype=np.float32)

    def update_from_flags(self, flags_full, asr_full: float, n_eval_cheap: int):
        try:
            n_eval_cheap = int(max(1, n_eval_cheap))
            if flags_full is None or len(flags_full) < n_eval_cheap:
                return
            asr_full = float(asr_full)
            asr_cheap = float(np.mean(flags_full[:n_eval_cheap]))
            residual = (asr_full - asr_cheap)
            residual_scaled = residual * math.sqrt(float(n_eval_cheap))
            self.residual_scaled.append(float(residual_scaled))
            if len(self.residual_scaled) > CALIB_MAX_KEEP:
                self.residual_scaled = self.residual_scaled[-CALIB_MAX_KEEP:]
        except Exception:
            pass

    def get_sigma(self, n_eval: int) -> float:
        n_eval = int(max(1, n_eval))
        if len(self.residual_scaled) < CALIB_MIN_SAMPLES:
            return float(FALLBACK_SIGMA)
        arr = np.array(self.residual_scaled, dtype=np.float32)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        sigma_scaled = float(1.4826 * mad)
        sigma = sigma_scaled / max(1e-6, math.sqrt(float(n_eval)))
        sigma = float(np.clip(sigma, CALIB_CLAMP_MIN, CALIB_CLAMP_MAX))
        return sigma


def eval_pairwise_metrics_indices(theta_trial, theta_cur, imgs_rgb_in, bboxes_list_in, true_labels_in, indices, G: int = 5, return_trial_flags: bool = False):
    set_eval_globals(imgs_rgb_in, bboxes_list_in, true_labels_in, G_val=G)
    indices = np.asarray(indices, dtype=np.int32)
    n_eval = int(len(indices))

    t_base_losses, c_base_losses = [], []
    t_succ, c_succ = 0, 0
    n_valid = 0
    adv_feats_t_all, adv_feats_c_all = [], []
    idx_all = []
    budget_sum_t_total = torch.zeros((), device=DEVICE, dtype=torch.float32)
    budget_sum_c_total = torch.zeros((), device=DEVICE, dtype=torch.float32)

    GPU_BATCH = 64
    for b0 in range(0, n_eval, GPU_BATCH):
        b1 = min(n_eval, b0 + GPU_BATCH)
        idx_chunk = indices[b0:b1]

        img_feat_t, budget_sum_t_batch = gpu_evaluator.forward_batch_full_image(
            idx_chunk, theta_trial, G,
            master_size=MASTER_SIZE, patch_color=PATCH_COLOR_RGB, darken_strength=DARKEN_STRENGTH
        )
        img_feat_c, budget_sum_c_batch = gpu_evaluator.forward_batch_full_image(
            idx_chunk, theta_cur, G,
            master_size=MASTER_SIZE, patch_color=PATCH_COLOR_RGB, darken_strength=DARKEN_STRENGTH
        )

        budget_sum_t_total = budget_sum_t_total + budget_sum_t_batch
        budget_sum_c_total = budget_sum_c_total + budget_sum_c_batch
        loss_t = ev_base_loss_vec(img_feat_t)
        loss_c = ev_base_loss_vec(img_feat_c)
        logits_t = (_clip_logit_scale.float()) * (img_feat_t @ _clip_class_embs.float().t())
        logits_c = (_clip_logit_scale.float()) * (img_feat_c @ _clip_class_embs.float().t())

        y_idx_cpu = np.array([LABEL2IDX.get(true_labels[k], -1) for k in idx_chunk], dtype=np.int64)
        valid_pos = np.where(y_idx_cpu >= 0)[0]
        if valid_pos.size == 0:
            continue

        y_idx_gpu = torch.from_numpy(y_idx_cpu).to(DEVICE)
        valid_mask_gpu = (y_idx_gpu >= 0)
        pred_t = logits_t.argmax(dim=-1)
        pred_c = logits_c.argmax(dim=-1)
        succ_t = (pred_t != y_idx_gpu) & valid_mask_gpu
        succ_c = (pred_c != y_idx_gpu) & valid_mask_gpu
        t_succ += int(succ_t.sum().item())
        c_succ += int(succ_c.sum().item())
        n_valid += int(valid_mask_gpu.sum().item())

        loss_t_cpu = loss_t.detach().float().cpu().numpy()
        loss_c_cpu = loss_c.detach().float().cpu().numpy()
        t_base_losses.extend(loss_t_cpu[valid_pos].tolist())
        c_base_losses.extend(loss_c_cpu[valid_pos].tolist())

        sel = torch.as_tensor(valid_pos, device=DEVICE, dtype=torch.long)
        adv_feats_t_all.append(img_feat_t.index_select(0, sel))
        adv_feats_c_all.append(img_feat_c.index_select(0, sel))
        idx_tensor = torch.as_tensor(idx_chunk, device=DEVICE, dtype=torch.long)
        idx_all.append(idx_tensor.index_select(0, sel))

    if n_valid == 0:
        return float(np.inf), float(np.inf), 0, 0, 0, None, None

    t_cvar = _cvar_topk_numpy(t_base_losses, CVAR_Q)
    c_cvar = _cvar_topk_numpy(c_base_losses, CVAR_Q)
    L_budget_t = float(budget_sum_t_total.item()) / max(1.0, float(n_valid))
    L_budget_c = float(budget_sum_c_total.item()) / max(1.0, float(n_valid))

    L_topo_t, L_topo_c = 0.0, 0.0
    if (_clean_ev_feats is not None) and (_topo_sigma is not None) and adv_feats_t_all:
        adv_t = torch.cat(adv_feats_t_all, dim=0)
        adv_c = torch.cat(adv_feats_c_all, dim=0)
        idx_cat = torch.cat(idx_all, dim=0)
        clean = _clean_ev_feats.index_select(0, idx_cat)
        L_topo_t = topo_graph_neg_kl(clean, adv_t, sigma=float(_topo_sigma))
        L_topo_c = topo_graph_neg_kl(clean, adv_c, sigma=float(_topo_sigma))

    t_loss = float(t_cvar + float(LAMBDA_TOPO) * float(L_topo_t) + float(LAMBDA_PHYS) * float(L_budget_t))
    c_loss = float(c_cvar + float(LAMBDA_TOPO) * float(L_topo_c) + float(LAMBDA_PHYS) * float(L_budget_c))
    return t_loss, c_loss, int(t_succ), int(c_succ), int(n_valid), None, None


def smart_confirm_selection_incremental_ab(
    trial, current,
    imgs_rgb_in, bboxes_list_in, true_labels_in, G: int,
    n_eval: int, max_samples: int,
    gen: int, i: int, seed: int,
    calibrator: OnlineCalibrator,
    eval_mode: str = EVAL_MODE,
    superset_idx=None,
    force_full_confirm: bool = False,
):
    _ = calibrator
    total = len(imgs_rgb_in)
    if total <= 0:
        tec, ter = theta_edge_stats(trial, G)
        cec, cer = theta_edge_stats(current, G)
        return False, 0, np.inf, 0.0, cec, cer, np.inf, 0.0, tec, ter

    n_eval = int(min(max(1, n_eval), total))
    max_samples = int(min(max(1, max_samples), total))
    if superset_idx is None:
        superset_idx = make_gen_superset_indices(total, max_samples, seed, gen, mode=eval_mode)
    else:
        superset_idx = np.asarray(superset_idx, dtype=np.int32)
        if len(superset_idx) > max_samples:
            superset_idx = superset_idx[:max_samples]

    cheap_idx = superset_idx[:n_eval]
    t_loss0, c_loss0, t_succ0, c_succ0, n0, _, _ = eval_pairwise_metrics_indices(
        trial, current, imgs_rgb_in, bboxes_list_in, true_labels_in, cheap_idx, G=G, return_trial_flags=False
    )
    ta_cheap = t_succ0 / max(1, n0)
    ca_cheap = c_succ0 / max(1, n0)

    loss_gap = float(c_loss0 - t_loss0)  # >0 => trial better
    if force_full_confirm:
        n_confirm = max_samples
    else:
        base = max(1.0, abs(c_loss0))
        rel_gap = abs(loss_gap) / base
        if rel_gap < 0.01:
            n_confirm = min(max_samples, min(total, max(n_eval * 2, n_eval + 1)))
        elif rel_gap < 0.03:
            n_confirm = min(max_samples, min(total, max(int(round(n_eval * 1.5)), n_eval + 1)))
        else:
            n_confirm = n_eval

    if n_confirm <= n_eval:
        t_loss = float(t_loss0)
        c_loss = float(c_loss0)
        ta_confirm = float(ta_cheap)
        ca_confirm = float(ca_cheap)
    else:
        confirm_idx = superset_idx[:n_confirm]
        t_loss, c_loss, t_succ, c_succ, n_valid, _, _ = eval_pairwise_metrics_indices(
            trial, current, imgs_rgb_in, bboxes_list_in, true_labels_in, confirm_idx, G=G, return_trial_flags=False
        )
        if n_valid <= 0:
            t_loss = float(t_loss0)
            c_loss = float(c_loss0)
            ta_confirm = float(ta_cheap)
            ca_confirm = float(ca_cheap)
        else:
            ta_confirm = float(t_succ / float(max(1, n_valid)))
            ca_confirm = float(c_succ / float(max(1, n_valid)))

    accept = better_loss_only(t_loss, c_loss, eps_loss=LOSS_EPS)
    tec, ter = theta_edge_stats(trial, G)
    cec, cer = theta_edge_stats(current, G)
    return accept, int(n_confirm), float(c_loss), float(ca_confirm), int(cec), float(cer), float(t_loss), float(ta_confirm), int(tec), float(ter)


class StrategyPool:
    def __init__(self, size: int = 20):
        self.size = int(size)
        self.strategies = []

    def add_success(self, bl, br, dn, cs, score):
        self.strategies.append((int(bl), int(br), int(dn), int(cs), float(score)))
        self.strategies.sort(key=lambda x: x[4], reverse=True)
        self.strategies = self.strategies[:self.size]

    def sample_good_strategy(self):
        if (not self.strategies) or (random.random() > 0.8):
            bl = random.choice([0, 1, 2, 3])
            br = random.choice([0, 1, 2, 3])
            if bl == br:
                br = (br + 1) % 4
            dn = random.choice([1, 2, 3])
            cs = random.choice([0, 1, 2])
            return bl, br, dn, cs
        idx = random.randint(0, len(self.strategies) - 1)
        return self.strategies[idx][:4]


class DualEliteArchive:
    def __init__(self, size: int = ELITE_SIZE):
        self.size = int(size)
        self.archive = []
        self.current_best_theta = None
        self.current_best_fit = np.inf
        self.current_best_asr = -1.0
        self.current_best_er = 1.0

    def update(self, theta, fit, asr, er):
        if fit < self.current_best_fit - LOSS_EPS:
            self.current_best_theta = theta.copy()
            self.current_best_fit = float(fit)
            self.current_best_asr = float(asr)
            self.current_best_er = float(er)
        self.archive.append((theta.copy(), float(fit), float(asr), float(er)))
        self.archive.sort(key=lambda x: x[1])
        self.archive = self.archive[:self.size]

    def sample(self):
        return random.choice(self.archive)[0].copy() if self.archive else None

    def get_current_best(self):
        return self.current_best_theta.copy() if self.current_best_theta is not None else None


class MetaLearner:
    def __init__(self, decay: float = META_DECAY, exploration: float = META_EXPLORATION):
        self.decay = float(decay)
        self.exploration = float(exploration)
        self.successes = defaultdict(float)
        self.trials = defaultdict(float)
        self.best_strategy = None
        self.best_score = -np.inf

    def _strategy_key(self, bl, br, dn, cs):
        return (int(bl), int(br), int(dn), int(cs))

    def select_strategy(self):
        if len(self.trials) == 0:
            bl = random.choice([0, 1, 2, 3])
            br = random.choice([0, 1, 2, 3])
            if bl == br:
                br = (br + 1) % 4
            dn = random.choice([1, 2, 3])
            cs = random.choice([0, 1, 2])
            return bl, br, dn, cs

        total_trials = sum(self.trials.values()) + 1e-6
        ucb_scores = {}
        for key, n in self.trials.items():
            if n <= 0:
                continue
            avg = self.successes[key] / max(1e-6, n)
            bonus = self.exploration * math.sqrt(math.log(total_trials + 1.0) / max(1e-6, n))
            ucb_scores[key] = avg + bonus

        untried = []
        for bl in range(4):
            for br in range(4):
                if bl == br:
                    continue
                for dn in [1, 2, 3]:
                    for cs in [0, 1, 2]:
                        k = (bl, br, dn, cs)
                        if k not in self.trials:
                            untried.append(k)
        if untried:
            return random.choice(untried)
        return max(ucb_scores.keys(), key=lambda k: ucb_scores[k])

    def update(self, bl, br, dn, cs, improvement):
        key = self._strategy_key(bl, br, dn, cs)
        self.successes[key] *= self.decay
        self.trials[key] *= self.decay
        self.trials[key] += 1.0
        imp = float(improvement)
        imp = max(-1.0, min(1.0, imp))
        self.successes[key] += imp
        score = self.successes[key] / max(1e-6, self.trials[key])
        if score > self.best_score:
            self.best_score = score
            self.best_strategy = key


def get_adaptive_F_CR_range(progress: float, stagnation: int):
    if progress < 0.3:
        F_range = (0.3, 1.0)
        CR_range = (0.1, 1.0)
    elif progress < 0.7:
        F_range = (0.4, 0.95)
        CR_range = (0.2, 0.95)
    else:
        F_range = (0.5, 0.8)
        CR_range = (0.5, 0.9)
    if stagnation > 5:
        F_range = (0.2, 1.0)
        CR_range = (0.0, 1.0)
    return F_range, CR_range


def sample_pbest(pop, asr_cache, fit, p: float = 0.05, progress: float = 0.0):
    _ = asr_cache
    n = len(pop)
    p_adaptive = p + (0.15 - p) * (1.0 - progress)
    k = max(2, int(round(n * p_adaptive)))
    idx_sorted = rank_by_loss(fit)
    idx = idx_sorted[:k]
    return pop[int(random.choice(idx))]


def pde_init_params(n: int, strategy_pool: StrategyPool = None, meta_learner: MetaLearner = None, is_meta_guided=None):
    F = np.random.uniform(JDE_F_LO, JDE_F_HI, size=n).astype(np.float32)
    CR = np.random.uniform(JDE_CR_LO, JDE_CR_HI, size=n).astype(np.float32)
    dn = np.zeros(n, dtype=np.int32)
    cs = np.zeros(n, dtype=np.int32)
    bl = np.zeros(n, dtype=np.int32)
    br = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if is_meta_guided is not None and is_meta_guided[i] and meta_learner is not None:
            bl[i], br[i], dn[i], cs[i] = meta_learner.select_strategy()
        elif strategy_pool is not None and random.random() < 0.5:
            bl[i], br[i], dn[i], cs[i] = strategy_pool.sample_good_strategy()
        else:
            dn[i] = random.choice([1, 2, 3])
            cs[i] = random.choice([0, 1, 2])
            bl[i] = random.choice([0, 1, 2, 3])
            br[i] = random.choice([0, 1, 2, 3])
            if bl[i] == br[i]:
                br[i] = (br[i] + 1) % 4
    return F, CR, dn, cs, bl, br


def pde_mutate_params(F, CR, dn, cs, bl, br, progress: float, F_range, CR_range, is_meta_guided=None, meta_stability: float = 0.70):
    n = len(F)
    F_lo, F_hi = F_range
    CR_lo, CR_hi = CR_range
    for i in range(n):
        if random.random() < JDE_TAU1:
            F[i] = np.random.uniform(F_lo, F_hi)
        if random.random() < JDE_TAU2:
            CR[i] = np.random.uniform(CR_lo, CR_hi)

    p_disc = DISC_MUT_P0 + (DISC_MUT_P1 - DISC_MUT_P0) * progress
    for i in range(n):
        if is_meta_guided is not None and is_meta_guided[i]:
            if random.random() < meta_stability:
                continue
        if random.random() < p_disc:
            dn[i] = random.choice([1, 2, 3])
            cs[i] = random.choice([0, 1, 2])
            bl[i] = random.choice([0, 1, 2, 3])
            br[i] = random.choice([0, 1, 2, 3])
            if bl[i] == br[i]:
                br[i] = (br[i] + 1) % 4


def get_vec_by_type(t: int, pop, asr_cache, fit, i: int, progress: float = 0.0):
    _ = asr_cache
    if int(t) == 0:
        return pop[random.randrange(len(pop))]
    if int(t) == 1:
        best_idx = best_idx_by_loss(fit)
        return pop[best_idx]
    if int(t) == 2:
        return sample_pbest(pop, asr_cache, fit, p=0.05, progress=progress)
    return pop[i]


def differential_sum(pop, i: int, dn_i: int):
    idxs = [x for x in range(len(pop)) if x != i]
    need = 2 * int(dn_i)
    if len(idxs) < need:
        chosen = [random.choice(idxs) for _ in range(need)]
    else:
        chosen = random.sample(idxs, need)
    diff = 0.0
    for k in range(int(dn_i)):
        a = pop[chosen[2 * k]]
        b = pop[chosen[2 * k + 1]]
        diff = diff + (a - b)
    return diff


def crossover_binomial_np(mut, cur, CR: float):
    D = len(cur)
    jrand = random.randint(0, D - 1)
    out = cur.copy()
    for j in range(D):
        if random.random() < CR or j == jrand:
            out[j] = mut[j]
    return out


def crossover_exponential_np(mut, cur, CR: float):
    D = len(cur)
    start = random.randint(0, D - 1)
    out = cur.copy()
    L = 0
    j = start
    while True:
        out[j] = mut[j]
        L += 1
        j = (j + 1) % D
        if L >= D:
            break
        if random.random() >= CR:
            break
    return out


def crossover_arithmetic_np(mut, cur):
    alpha = random.random()
    return (alpha * mut + (1.0 - alpha) * cur).astype(np.float32)


def pde_make_trial(pop, asr_cache, fit, i: int, Fv: float, CRv: float, dn_i: int, cs_i: int, bl_i: int, br_i: int, b_lo: np.ndarray, b_hi: np.ndarray, progress: float = 0.0):
    base_prim = get_vec_by_type(int(bl_i), pop, asr_cache, fit, i, progress)
    base_sec = get_vec_by_type(int(br_i), pop, asr_cache, fit, i, progress)
    base = base_prim + float(Fv) * (base_sec - base_prim)
    diff = differential_sum(pop, i, int(dn_i))
    mut = base + float(Fv) * diff
    cur = pop[i]
    if int(cs_i) == 0:
        trial = crossover_binomial_np(mut, cur, float(CRv))
    elif int(cs_i) == 1:
        trial = crossover_exponential_np(mut, cur, float(CRv))
    else:
        trial = crossover_arithmetic_np(mut, cur)
    trial = np.clip(trial, b_lo, b_hi)
    return trial


def local_search_iterative(theta, bounds, imgs_rgb_in, bboxes_list_in, true_labels_in, G: int, max_rounds: int = 1, step: float = 0.10):
    best_theta = theta.copy()
    best_fit, best_asr, best_edge_cnt, best_edge_ratio = evaluate_theta_full(best_theta, imgs_rgb_in, bboxes_list_in, true_labels_in, G=G)
    total_improved = False
    gate_end = int(G * G)
    D = len(theta)

    def _try_dims(dim_list, step_scale: float = 1.0):
        nonlocal best_theta, best_fit, best_asr, best_edge_cnt, best_edge_ratio, total_improved
        random.shuffle(dim_list)
        dim_list = dim_list[:12]
        for dim in dim_list:
            step_dim = float(step) * float(step_scale)
            if dim >= gate_end:
                step_dim *= 0.5
            for direction in (-1, 1):
                trial = best_theta.copy()
                delta = (bounds[dim][1] - bounds[dim][0]) * step_dim * direction
                trial[dim] = np.clip(trial[dim] + delta, bounds[dim][0], bounds[dim][1])
                trial_fit, trial_asr, trial_edge_cnt, trial_edge_ratio = evaluate_theta_full(trial, imgs_rgb_in, bboxes_list_in, true_labels_in, G=G)
                if better_loss_only(trial_fit, best_fit):
                    best_theta = trial.copy()
                    best_fit = float(trial_fit)
                    best_asr = float(trial_asr)
                    best_edge_cnt = int(trial_edge_cnt)
                    best_edge_ratio = float(trial_edge_ratio)
                    total_improved = True
                    return True
        return False

    for _ in range(int(max_rounds)):
        improved = _try_dims(list(range(0, gate_end)), step_scale=1.0)
        if improved:
            continue
        improved = _try_dims(list(range(gate_end, D)), step_scale=1.0)
        if not improved:
            break
    return best_theta, best_fit, best_asr, best_edge_cnt, best_edge_ratio, total_improved


def layered_initialization(bounds, pop_size: int, G: int):
    pop = []
    for _ in range(pop_size // 3):
        pop.append(random_theta(bounds, G, gate_density=random.uniform(0.25, 0.40)))
    for _ in range(pop_size // 3):
        pop.append(random_theta(bounds, G, gate_density=random.uniform(0.50, 0.70)))
    for _ in range(pop_size - 2 * (pop_size // 3)):
        pop.append(random_theta(bounds, G, gate_density=random.uniform(0.75, 0.90)))
    random.shuffle(pop)
    return pop


def diversity_restart_enhanced(pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache, bounds, global_best_theta, dual_archive: DualEliteArchive, G: int, ratio: float = RESTART_RATIO):
    _ = asr_cache
    if global_best_theta is None:
        return []
    n = len(pop)
    k = int(max(1, n * ratio))
    worst_idx = rank_by_loss(fit)[-k:]
    replaced_indices = []
    b_lo = np.array([b[0] for b in bounds], dtype=np.float32)
    b_hi = np.array([b[1] for b in bounds], dtype=np.float32)
    span = (b_hi - b_lo)
    gate_end = int(G * G)

    for idx, wi in enumerate(worst_idx):
        wi = int(wi)
        if idx < k // 3:
            child = global_best_theta.copy()
            noise = np.random.normal(0, 1, size=child.shape).astype(np.float32)
            child[:gate_end] += noise[:gate_end] * (span[:gate_end] * RESTART_NOISE * 0.5)
            child[gate_end:] += noise[gate_end:] * (span[gate_end:] * RESTART_NOISE)
            child = np.clip(child, b_lo, b_hi)
        elif idx < 2 * k // 3:
            elite_sample = dual_archive.sample()
            if elite_sample is not None:
                child = elite_sample.copy()
                noise = np.random.normal(0, 1, size=child.shape).astype(np.float32)
                child[:gate_end] += noise[:gate_end] * (span[:gate_end] * RESTART_NOISE * 0.7)
                child[gate_end:] += noise[gate_end:] * (span[gate_end:] * RESTART_NOISE * 1.2)
                child = np.clip(child, b_lo, b_hi)
            else:
                child = random_theta(bounds, G, gate_density=random.uniform(0.4, 0.85))
        else:
            child = random_theta(bounds, G, gate_density=random.uniform(0.4, 0.85))

        pop[wi] = child
        fit[wi] = np.inf
        asr_cache[wi] = -1.0
        edge_ratio_cache[wi] = 0.0
        edge_cnt_cache[wi] = 0
        replaced_indices.append(wi)
    return replaced_indices


def inject_elite_best_into_pop(pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache, best_theta, best_full_asr: float, best_full_fit: float, G: int, n_inject: int = 1):
    if best_theta is None or n_inject <= 0:
        return []
    n = len(pop)
    n_inject = int(min(max(1, n_inject), n))
    worst_idx = rank_by_loss(fit)[-n_inject:]
    ec, er = theta_edge_stats(best_theta, G)
    replaced = []
    for wi in worst_idx:
        wi = int(wi)
        pop[wi] = best_theta.copy()
        asr_cache[wi] = float(best_full_asr)
        fit[wi] = float(best_full_fit)
        edge_ratio_cache[wi] = float(er)
        edge_cnt_cache[wi] = int(ec)
        replaced.append(wi)
    return replaced


def make_bounds(G: int = 5):
    N_EDGES_LOCAL = 2 * G * (G + 1)
    cell_gates = [(0.0, 1.0) for _ in range(G * G)]
    edge_offsets = [(-1.0, 1.0) for _ in range(N_EDGES_LOCAL)]
    return cell_gates + edge_offsets


def random_theta(bounds, G: int, gate_density=None):
    N_CELLS = int(G * G)
    N_EDGES_LOCAL = 2 * G * (G + 1)
    theta = np.array([random.uniform(a, b) for (a, b) in bounds], dtype=np.float32)
    if gate_density is None:
        theta[0:N_CELLS] = np.random.uniform(0.2, 1.0, size=N_CELLS)
    else:
        n_open = int(round(N_CELLS * float(gate_density)))
        gates = np.zeros(N_CELLS, dtype=np.float32)
        open_indices = np.random.choice(N_CELLS, size=n_open, replace=False)
        gates[open_indices] = np.random.uniform(0.7, 1.0, size=n_open)
        closed_indices = np.setdiff1d(np.arange(N_CELLS), open_indices)
        gates[closed_indices] = np.random.uniform(0.0, 0.4, size=len(closed_indices))
        theta[0:N_CELLS] = gates
    theta[N_CELLS:N_CELLS + N_EDGES_LOCAL] = np.random.uniform(-1.0, 1.0, size=N_EDGES_LOCAL)
    return theta


def save_adv_images_for_theta(OUT_DIR: str, subdir_name: str, theta, imgs_rgb_in, bboxes_list_in, kept_paths_in, asr_full: float, G: int = 5):
    out_sub = os.path.join(OUT_DIR, subdir_name)
    os.makedirs(out_sub, exist_ok=True)
    master_mask_np = generate_master_mask(theta, G=G, size=MASTER_SIZE)
    master_mask_t = torch.from_numpy(master_mask_np).to(DEVICE).float().unsqueeze(0).unsqueeze(0)

    global _clip_model, _clip_processor, _clip_class_embs, _clip_logit_scale, _PREPROC_GEOM
    if _clip_model is None or _clip_processor is None or _clip_class_embs is None or _clip_logit_scale is None:
        log("[SAVE] CLIP not initialized. Skip saving labels.", level=0)
    if _PREPROC_GEOM is None:
        log("[SAVE] _PREPROC_GEOM is None. Skip saving labels.", level=0)

    geom = dict(_PREPROC_GEOM) if _PREPROC_GEOM is not None else None
    S = int(geom["output_size"]) if geom is not None else None
    model_dtype = next(_clip_model.parameters()).dtype if _clip_model is not None else torch.float16

    color_tensor = None
    if _clip_processor is not None and geom is not None:
        color_tensor = make_patch_color_tensor_norm(
            _clip_processor,
            patch_color_rgb=PATCH_COLOR_RGB,
            device=DEVICE,
            dtype=model_dtype,
            out_size=S,
        )

    common_root = None
    try:
        if kept_paths_in:
            common_root = os.path.commonpath(list(kept_paths_in))
            if common_root and os.path.isfile(common_root):
                common_root = os.path.dirname(common_root)
    except Exception:
        common_root = None

    grouped = OrderedDict()
    N = len(imgs_rgb_in)
    for i in range(N):
        p = kept_paths_in[i]
        if p not in grouped:
            grouped[p] = {"img": imgs_rgb_in[i], "items": []}
        bbs = bboxes_list_in[i] if bboxes_list_in is not None else []
        if bbs is None:
            bbs = []
        for bb in bbs:
            grouped[p]["items"].append({"bb": bb})

    resize_cache = {}
    BS = 32

    for p, pack in grouped.items():
        img0 = pack["img"]
        if img0 is None:
            continue
        img0 = ensure_rgb3(img0)
        if img0 is None:
            continue

        uniq_items = []
        seen = set()
        for it in pack["items"]:
            bb = it.get("bb", None)
            if bb is None:
                continue
            try:
                x1, y1, x2, y2 = map(float, bb[:4])
                key = (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_items.append(it)

        adv = img0.copy()
        for it in uniq_items:
            adv = render_mesh_on_bbox(adv, it["bb"], theta=theta, G=G, master_mask=master_mask_np)

        if (_clip_model is not None and _clip_processor is not None and _clip_class_embs is not None and _clip_logit_scale is not None and geom is not None and color_tensor is not None):
            pil_images = []
            metas = []
            kept_items = []
            for it in uniq_items:
                bb = it["bb"]
                full_rgb, rel_bb = full_image_and_bbox(img0, bb)
                if full_rgb is None or rel_bb is None or full_rgb.size == 0:
                    it["pred_cls"] = "unknown"
                    it["pred_prob"] = 0.0
                    continue
                ch, cw = full_rgb.shape[:2]
                mx1, my1, mx2, my2, _, _ = map_bbox_to_preproc_space(rel_bb, ch, cw, geom)
                bw_m = max(1.0, float(mx2 - mx1))
                bh_m = max(1.0, float(my2 - my1))
                pil_images.append(Image.fromarray(full_rgb, mode="RGB"))
                metas.append((mx1, my1, mx2, my2, bw_m, bh_m))
                kept_items.append(it)

            out_idx = []
            out_prob = []
            for st in range(0, len(pil_images), BS):
                chunk = pil_images[st: st + BS]
                meta_chunk = metas[st: st + BS]
                enc = _clip_processor(images=chunk, return_tensors="pt")
                pv = enc["pixel_values"].to(DEVICE, non_blocking=True).to(model_dtype)
                B = int(pv.shape[0])

                for j in range(B):
                    mx1, my1, mx2, my2, bw_m, bh_m = meta_chunk[j]
                    raw_side = float(bh_m) / float(ROI_CORE_DIV)
                    core_w = max(4.0, raw_side)
                    core_h = max(4.0, raw_side)
                    target_w = int(round(core_w * (1.0 + 2.0 * float(EXPANSION_RATIO))))
                    target_h = int(round(core_h * (1.0 + 2.0 * float(EXPANSION_RATIO))))
                    if target_w < 2 or target_h < 2:
                        continue

                    cx = 0.5 * (float(mx1) + float(mx2))
                    cy = 0.5 * (float(my1) + float(my2))
                    paste_x1 = int(round(cx - target_w / 2.0))
                    paste_y1 = int(round(cy - target_h / 2.0))
                    paste_x2 = paste_x1 + target_w
                    paste_y2 = paste_y1 + target_h
                    ix1, iy1 = max(0, paste_x1), max(0, paste_y1)
                    ix2, iy2 = min(S, paste_x2), min(S, paste_y2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    key = (target_h, target_w, model_dtype)
                    small_mask_full = resize_cache.get(key, None)
                    if small_mask_full is None:
                        small_mask_full = F.interpolate(master_mask_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
                        small_mask_full = small_mask_full.to(dtype=model_dtype)
                        resize_cache[key] = small_mask_full

                    mx_off1 = ix1 - paste_x1
                    my_off1 = iy1 - paste_y1
                    mx_off2 = mx_off1 + (ix2 - ix1)
                    my_off2 = my_off1 + (iy2 - iy1)
                    m = small_mask_full[:, :, my_off1:my_off2, mx_off1:mx_off2]
                    if m.numel() == 0:
                        continue
                    s = float(np.clip(DARKEN_STRENGTH, 0.0, 1.0))
                    m = (m * s).clamp_(0.0, 1.0)
                    roi = pv[j: j + 1, :, iy1:iy2, ix1:ix2]
                    pv[j: j + 1, :, iy1:iy2, ix1:ix2] = roi * (1.0 - m) + color_tensor * m

                with torch.inference_mode():
                    img_feat = _clip_model.get_image_features(pixel_values=pv).float()
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    logits = (_clip_logit_scale.float()) * (img_feat @ _clip_class_embs.float().t())
                    probs = torch.softmax(logits, dim=-1)
                    pred_idx = probs.argmax(dim=-1)
                    pred_prob = probs.max(dim=-1).values

                out_idx.extend(pred_idx.detach().cpu().tolist())
                out_prob.extend(pred_prob.detach().cpu().tolist())

            for it, pi, pp in zip(kept_items, out_idx, out_prob):
                it["pred_idx"] = int(pi)
                it["pred_cls"] = str(CLASSES[int(pi)]) if int(pi) < len(CLASSES) else "unknown"
                it["pred_prob"] = float(pp)

        for it in uniq_items:
            if "pred_cls" not in it:
                it["pred_cls"] = "unknown"
                it["pred_prob"] = 0.0

        best_cls = "unknown"
        best_prob = -1.0
        for it in uniq_items:
            pp = float(it.get("pred_prob", 0.0))
            if pp > best_prob:
                best_prob = pp
                best_cls = str(it.get("pred_cls", "unknown"))
        if best_prob < 0:
            best_prob = 0.0

        try:
            if common_root:
                rel = os.path.relpath(p, common_root)
            else:
                rel = os.path.basename(p)
            rel_stem = os.path.splitext(rel)[0].replace(os.sep, "__")
        except Exception:
            rel_stem = os.path.splitext(os.path.basename(p))[0]

        best_cls_clean = best_cls.replace("/", "_").replace("\\", "_").replace(" ", "_")
        orig_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        orig_filename = f"{rel_stem}__orig.jpg"
        cv2.imwrite(os.path.join(out_sub, orig_filename), orig_bgr)
        adv_bgr = cv2.cvtColor(adv, cv2.COLOR_RGB2BGR)
        adv_filename = f"{rel_stem}__{best_cls_clean}__{best_prob:.3f}__adv.jpg"
        cv2.imwrite(os.path.join(out_sub, adv_filename), adv_bgr)


def sync_latest_dir(OUT_DIR: str, snapshot_dirname: str):
    latest = os.path.join(OUT_DIR, BEST_LATEST_DIRNAME)
    src = os.path.join(OUT_DIR, snapshot_dirname)
    if os.path.isdir(latest):
        shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(src, latest)


def save_eval_cache(eval_cache_path: str, kept_paths, bboxes_list_in, true_labels_in):
    os.makedirs(os.path.dirname(eval_cache_path), exist_ok=True)
    np.savez(
        eval_cache_path,
        kept_paths=np.array(kept_paths, dtype=object),
        bboxes_list=np.array(bboxes_list_in, dtype=object),
        true_labels=np.array(true_labels_in, dtype=object),
    )


def load_eval_cache(eval_cache_path: str):
    if not os.path.exists(eval_cache_path):
        return None
    try:
        d = np.load(eval_cache_path, allow_pickle=True)
        return list(d["kept_paths"]), list(d["bboxes_list"]), list(d["true_labels"])
    except Exception:
        return None


def save_checkpoint(
    path, gen, bounds_G,
    pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache,
    Fv, CRv, dnv, csv, blv, brv,
    best_global_theta, best_global_fit, best_global_asr, best_global_edge_ratio, best_global_edge_cnt, best_global_gen,
    stagnation,
    dual_archive: DualEliteArchive,
    strategy_pool: StrategyPool,
    meta_learner: MetaLearner = None,
    is_meta_guided=None,
    best_hist_fit: float = float(np.inf),
    best_hist_asr: float = float(-1.0),
    best_hist_theta=None,
    best_full_theta=None,
    best_full_asr: float = float(-1.0),
    best_full_fit: float = float(np.inf),
    calibrator: OnlineCalibrator = None,
):
    elite_data = np.array([(e[0], e[1], e[2], e[3]) for e in dual_archive.archive], dtype=object)
    strategy_data = np.array(strategy_pool.strategies, dtype=object) if strategy_pool.strategies else np.array([])
    meta_successes = dict(meta_learner.successes) if meta_learner else {}
    meta_trials = dict(meta_learner.trials) if meta_learner else {}
    meta_best_strategy = meta_learner.best_strategy if meta_learner else None
    meta_best_score = meta_learner.best_score if meta_learner else -np.inf

    np.savez(
        path,
        objective=np.array(["loss_only"], dtype=object),
        gen=int(gen),
        G=int(bounds_G),
        pop=np.array(pop, dtype=object),
        fit=np.array(fit, dtype=np.float32),
        asr_cache=np.array(asr_cache, dtype=np.float32),
        edge_ratio_cache=np.array(edge_ratio_cache, dtype=np.float32),
        edge_cnt_cache=np.array(edge_cnt_cache, dtype=np.int32),
        Fv=np.array(Fv, dtype=np.float32),
        CRv=np.array(CRv, dtype=np.float32),
        dnv=np.array(dnv, dtype=np.int32),
        csv=np.array(csv, dtype=np.int32),
        blv=np.array(blv, dtype=np.int32),
        brv=np.array(brv, dtype=np.int32),
        best_global_theta=np.array(best_global_theta, dtype=np.float32) if best_global_theta is not None else np.array([]),
        best_global_fit=float(best_global_fit),
        best_global_asr=float(best_global_asr),
        best_global_edge_ratio=float(best_global_edge_ratio),
        best_global_edge_cnt=int(best_global_edge_cnt),
        best_global_gen=int(best_global_gen),
        stagnation=int(stagnation),
        elite_archive=elite_data,
        strategy_pool=strategy_data,
        current_best_theta=np.array(dual_archive.current_best_theta, dtype=np.float32) if dual_archive.current_best_theta is not None else np.array([]),
        current_best_fit=float(dual_archive.current_best_fit),
        current_best_asr=float(dual_archive.current_best_asr),
        current_best_er=float(dual_archive.current_best_er),
        meta_successes=meta_successes,
        meta_trials=meta_trials,
        meta_best_strategy=meta_best_strategy,
        meta_best_score=float(meta_best_score),
        is_meta_guided=np.array(is_meta_guided, dtype=bool) if is_meta_guided is not None else np.array([]),
        best_hist_fit=float(best_hist_fit),
        best_hist_asr=float(best_hist_asr),
        best_hist_theta=np.array(best_hist_theta, dtype=np.float32) if best_hist_theta is not None else np.array([]),
        best_full_theta=np.array(best_full_theta, dtype=np.float32) if best_full_theta is not None else np.array([]),
        best_full_asr=float(best_full_asr),
        best_full_fit=float(best_full_fit),
        calib_residual_scaled=np.array(calibrator.state(), dtype=np.float32) if calibrator is not None else np.array([]),
    )


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        pop = [np.array(x, dtype=np.float32) for x in d["pop"]]
        fit = np.array(d["fit"], dtype=np.float32)
        asr_cache = np.array(d["asr_cache"], dtype=np.float32)
        edge_ratio_cache = np.array(d["edge_ratio_cache"], dtype=np.float32) if "edge_ratio_cache" in d.files else np.zeros((len(pop),), dtype=np.float32)
        edge_cnt_cache = np.array(d["edge_cnt_cache"], dtype=np.int32) if "edge_cnt_cache" in d.files else np.zeros((len(pop),), dtype=np.int32)
        Fv = np.array(d["Fv"], dtype=np.float32)
        CRv = np.array(d["CRv"], dtype=np.float32)
        dnv = np.array(d["dnv"], dtype=np.int32)
        csv = np.array(d["csv"], dtype=np.int32)
        blv = np.array(d["blv"], dtype=np.int32)
        brv = np.array(d["brv"], dtype=np.int32)

        best_global_theta = None
        if "best_global_theta" in d.files and len(d["best_global_theta"]) > 0:
            best_global_theta = np.array(d["best_global_theta"], dtype=np.float32)
        elif "best_asr_global_theta" in d.files and len(d["best_asr_global_theta"]) > 0:
            best_global_theta = np.array(d["best_asr_global_theta"], dtype=np.float32)

        if "best_global_fit" in d.files:
            best_global_fit = float(d["best_global_fit"])
            best_global_asr = float(d["best_global_asr"])
            best_global_edge_ratio = float(d["best_global_edge_ratio"])
            best_global_edge_cnt = int(d["best_global_edge_cnt"])
            best_global_gen = int(d["best_global_gen"])
        else:
            best_global_fit = float(np.min(fit)) if len(fit) else np.inf
            best_global_asr = float(d["best_asr_global_asr"]) if "best_asr_global_asr" in d.files else -1.0
            best_global_edge_ratio = float(d["best_asr_global_edge_ratio"]) if "best_asr_global_edge_ratio" in d.files else 0.0
            best_global_edge_cnt = int(d["best_asr_global_edge_cnt"]) if "best_asr_global_edge_cnt" in d.files else 0
            best_global_gen = int(d["best_asr_global_gen"]) if "best_asr_global_gen" in d.files else 0

        stagnation = int(d["stagnation"]) if "stagnation" in d.files else 0

        dual_archive = DualEliteArchive(size=ELITE_SIZE)
        if "elite_archive" in d.files:
            for item in d["elite_archive"]:
                theta = np.array(item[0], dtype=np.float32)
                fit_e = float(item[1])
                asr_e = float(item[2])
                er_e = float(item[3])
                dual_archive.archive.append((theta, fit_e, asr_e, er_e))
            dual_archive.archive.sort(key=lambda x: x[1])
            dual_archive.archive = dual_archive.archive[:dual_archive.size]

        if "current_best_theta" in d.files and len(d["current_best_theta"]) > 0:
            dual_archive.current_best_theta = np.array(d["current_best_theta"], dtype=np.float32)
            dual_archive.current_best_fit = float(d["current_best_fit"]) if "current_best_fit" in d.files else float(np.inf)
            dual_archive.current_best_asr = float(d["current_best_asr"]) if "current_best_asr" in d.files else -1.0
            dual_archive.current_best_er = float(d["current_best_er"]) if "current_best_er" in d.files else 1.0

        strategy_pool = StrategyPool(size=20)
        if "strategy_pool" in d.files and len(d["strategy_pool"]) > 0:
            for item in d["strategy_pool"]:
                strategy_pool.strategies.append(tuple(item))

        meta_learner = MetaLearner()
        if "meta_successes" in d.files and "meta_trials" in d.files:
            meta_learner.successes = defaultdict(float, d["meta_successes"].item())
            meta_learner.trials = defaultdict(float, d["meta_trials"].item())
            try:
                mbs = d["meta_best_strategy"]
                if hasattr(mbs, "shape") and mbs.shape == ():
                    meta_learner.best_strategy = mbs.item()
                else:
                    meta_learner.best_strategy = tuple(mbs) if mbs is not None else None
            except Exception:
                meta_learner.best_strategy = None
            meta_learner.best_score = float(d["meta_best_score"]) if "meta_best_score" in d.files else -np.inf

        if "is_meta_guided" in d.files and len(d["is_meta_guided"]) > 0:
            is_meta_guided = np.array(d["is_meta_guided"], dtype=bool)
        else:
            is_meta_guided = np.random.rand(len(pop)) < META_GUIDED_RATIO

        best_hist_theta = None
        best_hist_fit = float(np.inf)
        best_hist_asr = -1.0
        if "best_hist_theta" in d.files and len(d["best_hist_theta"]) > 0:
            best_hist_theta = np.array(d["best_hist_theta"], dtype=np.float32)
        if "best_hist_fit" in d.files:
            best_hist_fit = float(d["best_hist_fit"])
        if "best_hist_asr" in d.files:
            best_hist_asr = float(d["best_hist_asr"])
        elif "best_reeval" in d.files:
            best_hist_asr = float(d["best_reeval"])

        best_full_theta = None
        best_full_asr = -1.0
        best_full_fit = np.inf
        if "best_full_asr" in d.files:
            best_full_asr = float(d["best_full_asr"])
        if "best_full_theta" in d.files and len(d["best_full_theta"]) > 0:
            best_full_theta = np.array(d["best_full_theta"], dtype=np.float32)
        if "best_full_fit" in d.files:
            best_full_fit = float(d["best_full_fit"])

        calib_arr = None
        if "calib_residual_scaled" in d.files and len(d["calib_residual_scaled"]) > 0:
            calib_arr = np.array(d["calib_residual_scaled"], dtype=np.float32)

        return (
            int(d["gen"]), int(d["G"]),
            pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache,
            Fv, CRv, dnv, csv, blv, brv,
            best_global_theta, best_global_fit, best_global_asr, best_global_edge_ratio, best_global_edge_cnt, best_global_gen,
            stagnation, dual_archive, strategy_pool, meta_learner, is_meta_guided,
            best_hist_theta, best_hist_fit, best_hist_asr,
            best_full_theta, best_full_asr, best_full_fit,
            calib_arr,
        )
    except Exception:
        traceback.print_exc()
        return None


def export_theta_pack_from_ckpt(ckpt_path: str, out_dir: str, export_name: str = "exported_uap_theta_pack.npz"):
    if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
        log(f"[ERR] export failed: ckpt not found: {ckpt_path}", level=0)
        return None

    d = np.load(ckpt_path, allow_pickle=True)
    G_used = int(d["G"]) if "G" in d.files else int(G)

    def pick_theta_npz(d_npz):
        if "best_full_theta" in d_npz.files and getattr(d_npz["best_full_theta"], "size", 0) > 0:
            return d_npz["best_full_theta"].astype(np.float32), "best_full_theta"
        if "best_hist_theta" in d_npz.files and getattr(d_npz["best_hist_theta"], "size", 0) > 0:
            return d_npz["best_hist_theta"].astype(np.float32), "best_hist_theta"
        pop = d_npz["pop"]
        fit = d_npz["fit"].astype(np.float32)
        idx = int(np.argmin(fit))
        return np.array(pop[idx], dtype=np.float32), "best_pop_by_loss"

    theta, theta_tag = pick_theta_npz(d)
    export_best_loss = float(d["best_full_fit"]) if "best_full_fit" in d.files else (float(np.min(d["fit"])) if "fit" in d.files else np.inf)
    export_best_asr = float(d["best_full_asr"]) if "best_full_asr" in d.files else -1.0

    EXPORT_CFG = dict(
        MASTER_SIZE=np.int32(MASTER_SIZE),
        EXPANSION_RATIO=np.float32(EXPANSION_RATIO),
        THICKNESS_TO_CELL_RATIO=np.float32(THICKNESS_TO_CELL_RATIO),
        MAX_CURV_RATIO=np.float32(MAX_CURV_RATIO),
        ALPHA_GAMMA=np.float32(ALPHA_GAMMA),
        DARKEN_STRENGTH=np.float32(DARKEN_STRENGTH),
        PATCH_COLOR_RGB=np.array(list(PATCH_COLOR_RGB), dtype=np.int32),
        FAST_RENDER_LINE_SAMPLES=np.int32(FAST_RENDER_LINE_SAMPLES),
        GATE_THR=np.float32(GATE_THR),
        THETA_QUANT=np.float32(THETA_QUANT),
        best_loss=np.float32(export_best_loss),
        asr_of_best_loss=np.float32(export_best_asr),
    )

    os.makedirs(out_dir, exist_ok=True)
    export_path = os.path.join(out_dir, export_name)
    np.savez(
        export_path,
        theta=theta.astype(np.float32),
        G=np.int32(G_used),
        theta_tag=np.array([theta_tag], dtype=object),
        **EXPORT_CFG,
    )
    log(f"[OK] exported theta-pack -> {export_path} tag={theta_tag} theta_dim={theta.shape} G={G_used}", level=0)
    return export_path


def _compute_clean_ev_feats_with_gpu_evaluator(indices=None, batch: int = 64):
    global gpu_evaluator, _clip_model
    if gpu_evaluator is None:
        raise RuntimeError("gpu_evaluator is None")
    if _clip_model is None:
        raise RuntimeError("_clip_model is None")
    x = gpu_evaluator.clean_imgs_gpu
    N = int(x.shape[0])
    if indices is None:
        indices = np.arange(N, dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)

    feats = []
    with torch.inference_mode():
        model_dtype = next(_clip_model.parameters()).dtype
        for b0 in range(0, len(indices), int(batch)):
            b1 = min(len(indices), b0 + int(batch))
            idx = torch.as_tensor(indices[b0:b1], device=DEVICE, dtype=torch.long)
            pv = x.index_select(0, idx).to(model_dtype)
            f = _clip_model.get_image_features(pixel_values=pv).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feats.append(f)
    return torch.cat(feats, dim=0) if feats else torch.empty((0, 1), device=DEVICE)


def _build_ev_subspace_and_topo_sigma(clean_feats: torch.Tensor, rank: int):
    global _ev_mu, _ev_U, _clean_ev_feats, _topo_sigma
    if clean_feats is None or clean_feats.numel() == 0:
        _clean_ev_feats = None
        _ev_mu, _ev_U = None, None
        _topo_sigma = None
        return

    _clean_ev_feats = clean_feats.detach()
    with torch.no_grad():
        mu = clean_feats.mean(dim=0, keepdim=True)
        Rm = (clean_feats - mu)
        U, S, Vh = torch.linalg.svd(Rm, full_matrices=False)
        r = int(max(1, min(int(rank), int(Vh.shape[0]))))
        Q = Vh[:r].t().contiguous()
        _ev_mu = mu.squeeze(0).detach()
        _ev_U = Q.detach()
        dist = torch.cdist(clean_feats, clean_feats, p=2.0)
        dist2 = dist * dist
        _topo_sigma = _compute_sigma_median(dist2)
    log(f"[EV] built subspace: N={clean_feats.shape[0]} D={clean_feats.shape[1]} rank={r} topo_sigma={_topo_sigma:.6f}", level=0)


def de_optimize_pde_metade_hybrid(OUT_DIR: str, CKPT_LATEST_PATH: str, imgs_rgb_in, bboxes_list_in, true_labels_in, kept_paths_in, G: int = 5):
    set_eval_globals(imgs_rgb_in, bboxes_list_in, true_labels_in, G_val=G)

    last_saved_bucket = -1
    last_saved_loss = float(np.inf)
    last_saved_asr = -1.0

    pending_theta = None
    pending_asr = -1.0
    pending_fit = float(np.inf)
    pending_reason = ""
    pending_gen = -1

    best_fit_full_theta = None
    best_fit_full_fit = np.inf
    best_fit_full_asr = -1.0

    best_hist_fit = np.inf
    best_hist_theta = None
    best_hist_asr = -1.0

    bounds = make_bounds(G=G)
    b_lo = np.array([b[0] for b in bounds], dtype=np.float32)
    b_hi = np.array([b[1] for b in bounds], dtype=np.float32)
    calibrator = OnlineCalibrator()

    def _maybe_flush_pending_snapshot(gen: int, force: bool = False):
        nonlocal last_saved_bucket, last_saved_loss, last_saved_asr
        nonlocal pending_theta, pending_asr, pending_fit, pending_reason, pending_gen
        if not SAVE_IMAGES_ON_BEST_FULL:
            return
        if pending_theta is None:
            return
        bucket = int(gen // SAVE_BEST_SNAPSHOT_EVERY_GENS)
        if (not force) and (bucket == last_saved_bucket):
            return
        improved = pending_fit < last_saved_loss - LOSS_EPS
        if (not improved) and (not force):
            pending_theta = None
            return
        snap = f"bestLOSS_{pending_reason}_gen{pending_gen}_loss{pending_fit:.6f}_ASR{pending_asr*100:.2f}"
        save_adv_images_for_theta(OUT_DIR, snap, pending_theta, imgs_rgb_in, bboxes_list_in, kept_paths_in, pending_asr, G=G)
        sync_latest_dir(OUT_DIR, snap)
        last_saved_bucket = bucket
        last_saved_loss = float(pending_fit)
        last_saved_asr = float(pending_asr)
        pending_theta = None

    def update_best_hist_loss(theta, fit_full: float, asr_full: float, gen: int, reason: str = ""):
        nonlocal best_hist_fit, best_hist_theta, best_hist_asr
        nonlocal pending_theta, pending_asr, pending_fit, pending_reason, pending_gen
        if theta is None:
            return False
        fit_full = float(fit_full)
        asr_full = float(asr_full)
        if fit_full < best_hist_fit - LOSS_EPS:
            best_hist_fit = fit_full
            best_hist_theta = theta.copy()
            best_hist_asr = asr_full
            log(f"[BEST_LOSS_UP] loss={best_hist_fit:.6f} ASR_of_best_loss={best_hist_asr*100:.2f}% denom={MAX_SAMPLES} reason={reason}", level=0)
            pending_theta = best_hist_theta.copy()
            pending_asr = float(best_hist_asr)
            pending_fit = float(best_hist_fit)
            pending_reason = str(reason)
            pending_gen = int(gen)
            return True
        return False

    def update_best_fit_full(theta, fit_full: float, asr_full: float, reason: str = ""):
        nonlocal best_fit_full_theta, best_fit_full_fit, best_fit_full_asr
        if theta is None:
            return False
        fit_full = float(fit_full)
        asr_full = float(asr_full)
        improved = fit_full < best_fit_full_fit - LOSS_EPS
        if improved:
            best_fit_full_theta = theta.copy()
            best_fit_full_asr = asr_full
            best_fit_full_fit = fit_full
            log(f"[STAGE1] best_full_loss updated: loss={best_fit_full_fit:.6f} ASR_of_best_loss={best_fit_full_asr*100:.2f}% reason={reason}", level=0)
            return True
        return False

    ckpt = load_checkpoint(CKPT_LATEST_PATH)

    if ckpt is None:
        meta_learner = MetaLearner(decay=META_DECAY, exploration=META_EXPLORATION)
        is_meta_guided = np.random.rand(DE_POP) < META_GUIDED_RATIO
        pop = layered_initialization(bounds, DE_POP, G)
        fit = np.zeros((DE_POP,), dtype=np.float32)
        asr_cache = np.zeros((DE_POP,), dtype=np.float32)
        edge_ratio_cache = np.zeros((DE_POP,), dtype=np.float32)
        edge_cnt_cache = np.zeros((DE_POP,), dtype=np.int32)
        strategy_pool = StrategyPool(size=20)
        dual_archive = DualEliteArchive(size=ELITE_SIZE)
        Fv, CRv, dn, cs, bl, br = pde_init_params(DE_POP, strategy_pool=None, meta_learner=meta_learner, is_meta_guided=is_meta_guided)

        n_eval_init = int(round(0.32 * int(max(1, MAX_SAMPLES))))
        n_eval_init = int(min(int(MAX_SAMPLES), max(1, n_eval_init)))
        log("[STAGE1] Initializing population...", level=0)

        for i in range(DE_POP):
            f, a, ec, er, flags = evaluate_theta_full(pop[i], imgs_rgb_in, bboxes_list_in, true_labels_in, G=G, return_flags=True)
            fit[i] = f
            asr_cache[i] = a
            edge_ratio_cache[i] = er
            edge_cnt_cache[i] = ec
            dual_archive.update(pop[i], f, a, er)
            strategy_pool.add_success(bl[i], br[i], dn[i], cs[i], -float(f))
            calibrator.update_from_flags(flags, a, n_eval_cheap=n_eval_init)
            update_best_fit_full(pop[i], f, a, reason="init")
            update_best_hist_loss(pop[i], f, a, gen=0, reason="init")
            _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=0, tag="init")

        best_idx0 = best_idx_by_loss(fit)
        best_global_theta = pop[best_idx0].copy()
        best_global_fit = float(fit[best_idx0])
        best_global_asr = float(asr_cache[best_idx0])
        best_global_edge_ratio = float(edge_ratio_cache[best_idx0])
        best_global_edge_cnt = int(edge_cnt_cache[best_idx0])
        best_global_gen = 0

        stagnation = 0
        start_gen = 0
        save_checkpoint(
            CKPT_LATEST_PATH, start_gen, G,
            pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache,
            Fv, CRv, dn, cs, bl, br,
            best_global_theta, best_global_fit, best_global_asr, best_global_edge_ratio, best_global_edge_cnt, best_global_gen,
            stagnation, dual_archive, strategy_pool, meta_learner, is_meta_guided,
            best_hist_fit=best_hist_fit,
            best_hist_asr=best_hist_asr,
            best_hist_theta=best_hist_theta,
            best_full_theta=best_fit_full_theta,
            best_full_asr=best_fit_full_asr,
            best_full_fit=best_fit_full_fit,
            calibrator=calibrator,
        )
    else:
        (
            start_gen, G_ckpt,
            pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache,
            Fv, CRv, dn, cs, bl, br,
            best_global_theta, best_global_fit, best_global_asr, best_global_edge_ratio, best_global_edge_cnt, best_global_gen,
            stagnation, dual_archive, strategy_pool, meta_learner, is_meta_guided,
            best_hist_theta_ckpt, best_hist_fit_ckpt, best_hist_asr_ckpt,
            best_full_theta_ckpt, best_full_asr_ckpt, best_full_fit_ckpt,
            calib_arr,
        ) = ckpt

        if best_full_theta_ckpt is not None and len(best_full_theta_ckpt) > 0:
            best_fit_full_theta = best_full_theta_ckpt.copy()
            best_fit_full_asr = float(best_full_asr_ckpt)
            best_fit_full_fit = float(best_full_fit_ckpt)

        if best_hist_theta_ckpt is not None and len(best_hist_theta_ckpt) > 0:
            best_hist_theta = best_hist_theta_ckpt.copy()
        best_hist_fit = float(best_hist_fit_ckpt) if best_hist_fit_ckpt is not None else float(np.inf)
        best_hist_asr = float(best_hist_asr_ckpt) if best_hist_asr_ckpt is not None else -1.0
        calibrator.load_state(calib_arr)

    _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=int(start_gen), tag="resume_or_start")

    prev_best_loss = float(np.min(fit)) if len(fit) else float(np.inf)
    last_gen_time = None
    gen_last = int(start_gen)

    for gen in range(int(start_gen), int(DE_GENS)):
        gen_last = int(gen)
        progress = gen / max(1, int(DE_GENS) - 1)
        n_eval = schedule_n_eval(progress, MAX_SAMPLES)
        totalN = len(imgs_rgb_in)
        gen_superset_idx = make_gen_superset_indices(total=totalN, n_superset=MAX_SAMPLES, seed=RANDOM_SEED, gen=gen, mode=EVAL_MODE)

        k_force = int(min(TOPK_FORCE_FULL_CONFIRM, DE_POP))
        topk_force_set = set(rank_by_loss(fit)[:k_force].tolist())

        t_gen0 = time.time()
        st = pop_stats(asr_cache, fit)
        eta_str = ""
        if last_gen_time is not None:
            left = (DE_GENS - (gen + 1))
            eta = last_gen_time * left
            eta_str = f" ETA≈{eta / 60:.1f}min"

        log(
            f"[GEN_START] gen={gen} ({gen + 1}/{DE_GENS}) n_eval={n_eval} "
            f"best_loss={st.get('best_loss', np.inf):.6f} ASR_of_best_loss={st.get('asr_of_best_loss', -1.0) * 100:.2f}%{eta_str}",
            level=1,
        )

        do_meta_resample = ((gen % META_UPDATE_FREQ) == 0)
        if do_meta_resample:
            resample_prob = RESAMPLE_PROB_BASE * (1 - progress * 0.5)
            for i2 in range(DE_POP):
                if is_meta_guided[i2]:
                    if (stagnation >= RESAMPLE_STAGNATION) or (random.random() < resample_prob):
                        bl[i2], br[i2], dn[i2], cs[i2] = meta_learner.select_strategy()

        F_range, CR_range = get_adaptive_F_CR_range(progress, stagnation)
        pde_mutate_params(Fv, CRv, dn, cs, bl, br, progress, F_range, CR_range, is_meta_guided=is_meta_guided, meta_stability=META_STABILITY_IN_MUTATE)

        accept_cnt = 0
        for i in range(DE_POP):
            trial = pde_make_trial(pop, asr_cache, fit, i, Fv[i], CRv[i], dn[i], cs[i], bl[i], br[i], b_lo, b_hi, progress)
            force_full = (i in topk_force_set)
            accept, n_confirm, cur_fit, cur_asr, cec, cer, tri_fit, tri_asr, tec2, ter2 = smart_confirm_selection_incremental_ab(
                trial, pop[i],
                imgs_rgb_in, bboxes_list_in, true_labels_in, G,
                n_eval, MAX_SAMPLES,
                gen=gen, i=i, seed=RANDOM_SEED,
                calibrator=calibrator,
                eval_mode=EVAL_MODE,
                superset_idx=gen_superset_idx,
                force_full_confirm=force_full,
            )

            if accept:
                accept_cnt += 1
                improvement_confirm = float(cur_fit - tri_fit)
                improvement_stable = float(fit[i] - tri_fit)
                improvement = CREDIT_BLEND_RATIO * improvement_confirm + (1 - CREDIT_BLEND_RATIO) * improvement_stable
                meta_learner.update(bl[i], br[i], dn[i], cs[i], improvement)
                pop[i] = trial
                fit[i] = float(tri_fit)
                asr_cache[i] = float(tri_asr)
                edge_ratio_cache[i] = float(ter2)
                edge_cnt_cache[i] = int(tec2)
                dual_archive.update(trial, float(tri_fit), float(tri_asr), float(ter2))
                strategy_pool.add_success(bl[i], br[i], dn[i], cs[i], -float(tri_fit))

            _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="inner_loop")

        do_elite = ((gen + 1) % ELITE_FULL_EVAL_FREQ == 0) or (gen == DE_GENS - 1)
        if do_elite:
            topk = min(ELITE_SIZE, DE_POP)
            top_idx = rank_by_loss(fit)[:topk]
            for idx in top_idx:
                idx = int(idx)
                f_full, a_full, ec_full, er_full, flags_full = evaluate_theta_on_indices(pop[idx], imgs_rgb_in, bboxes_list_in, true_labels_in, gen_superset_idx, G=G, return_flags=True)
                update_best_fit_full(pop[idx], f_full, a_full, reason=f"elitefull_gen{gen}")
                update_best_hist_loss(pop[idx], f_full, a_full, gen=gen, reason=f"elitefull_gen{gen}")
                calibrator.update_from_flags(flags_full, a_full, n_eval_cheap=n_eval)

                if ELITE_FULLEVAL_CREDIT:
                    improvement_confirm = float(fit[idx] - f_full)
                    improvement_stable = float(fit[idx] - f_full)
                    improvement = CREDIT_BLEND_RATIO * improvement_confirm + (1 - CREDIT_BLEND_RATIO) * improvement_stable
                    meta_learner.update(bl[idx], br[idx], dn[idx], cs[idx], improvement)

                fit[idx] = float(f_full)
                asr_cache[idx] = float(a_full)
                edge_ratio_cache[idx] = float(er_full)
                edge_cnt_cache[idx] = int(ec_full)
                dual_archive.update(pop[idx], float(f_full), float(a_full), float(er_full))
                strategy_pool.add_success(bl[idx], br[idx], dn[idx], cs[idx], -float(f_full))
                _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="elite_eval")

        if random.random() < LOCAL_SEARCH_RATE:
            best_idx = best_idx_by_loss(fit)
            ls_theta, ls_fit, ls_asr, ls_edge_cnt, ls_edge_ratio, ls_improved = local_search_iterative(pop[best_idx], bounds, imgs_rgb_in, bboxes_list_in, true_labels_in, G)
            if ls_improved:
                pop[best_idx] = ls_theta
                fit[best_idx] = float(ls_fit)
                asr_cache[best_idx] = float(ls_asr)
                edge_ratio_cache[best_idx] = float(ls_edge_ratio)
                edge_cnt_cache[best_idx] = int(ls_edge_cnt)
                dual_archive.update(ls_theta, float(ls_fit), float(ls_asr), float(ls_edge_ratio))
                update_best_fit_full(ls_theta, ls_fit, ls_asr, reason=f"localsearch_gen{gen}")
                update_best_hist_loss(ls_theta, ls_fit, ls_asr, gen=gen, reason=f"localsearch_gen{gen}")
                fit_ls, flags_ls, n_valid_ls, asr_ls = eval_success_flags_indices(ls_theta, gen_superset_idx)
                calibrator.update_from_flags(flags_ls, asr_ls, n_eval_cheap=n_eval)
                _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="local_search")

        cur_best_loss = float(np.min(fit)) if len(fit) else float(np.inf)
        if not better_loss_only(cur_best_loss, prev_best_loss):
            stagnation += 1
        else:
            stagnation = 0
            prev_best_loss = cur_best_loss

        if stagnation >= STAGNATION_LIMIT:
            best_idx = best_idx_by_loss(fit)
            global_best_theta = pop[best_idx].copy()
            replaced_indices = diversity_restart_enhanced(pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache, bounds, global_best_theta, dual_archive, G, ratio=RESTART_RATIO)
            stagnation = 0
            if len(replaced_indices) > 0:
                for wi in replaced_indices:
                    wi = int(wi)
                    if random.random() < RESTART_RESET_STRATEGY_PROB:
                        is_meta_guided[wi] = (random.random() < META_GUIDED_RATIO)
                    if is_meta_guided[wi]:
                        bl[wi], br[wi], dn[wi], cs[wi] = meta_learner.select_strategy()
                    else:
                        dn[wi] = random.choice([1, 2, 3])
                        cs[wi] = random.choice([0, 1, 2])
                        bl[wi] = random.choice([0, 1, 2, 3])
                        br[wi] = random.choice([0, 1, 2, 3])
                        if bl[wi] == br[wi]:
                            br[wi] = (br[wi] + 1) % 4
                    Fv[wi] = np.random.uniform(F_range[0], F_range[1])
                    CRv[wi] = np.random.uniform(CR_range[0], CR_range[1])

                    f, a, ec, er, flags_full = evaluate_theta_on_indices(pop[wi], imgs_rgb_in, bboxes_list_in, true_labels_in, gen_superset_idx, G=G, return_flags=True)
                    fit[wi] = float(f)
                    asr_cache[wi] = float(a)
                    edge_ratio_cache[wi] = float(er)
                    edge_cnt_cache[wi] = int(ec)
                    dual_archive.update(pop[wi], float(f), float(a), float(er))
                    strategy_pool.add_success(bl[wi], br[wi], dn[wi], cs[wi], -float(f))
                    update_best_fit_full(pop[wi], f, a, reason=f"restartfull_gen{gen}")
                    update_best_hist_loss(pop[wi], f, a, gen=gen, reason=f"restartfull_gen{gen}")
                    calibrator.update_from_flags(flags_full, a, n_eval_cheap=n_eval)
                    _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="restart")

        if ((gen + 1) % REEVAL_FREQ == 0):
            best_idx = best_idx_by_loss(fit)
            theta_chk = pop[best_idx]
            f_chk, asr_chk_full, _, _, flags_chk = evaluate_theta_on_indices(theta_chk, imgs_rgb_in, bboxes_list_in, true_labels_in, gen_superset_idx, G=G, return_flags=True)
            update_best_fit_full(theta_chk, f_chk, asr_chk_full, reason=f"reeval_gen{gen + 1}")
            update_best_hist_loss(theta_chk, f_chk, asr_chk_full, gen=gen, reason=f"reeval_gen{gen + 1}")
            calibrator.update_from_flags(flags_chk, asr_chk_full, n_eval_cheap=n_eval)
            _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="reeval")

        if (best_fit_full_theta is not None) and (ELITE_INJECT_N > 0):
            _ = inject_elite_best_into_pop(pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache, best_fit_full_theta, best_fit_full_asr, best_fit_full_fit, G, n_inject=ELITE_INJECT_N)

        if ((gen + 1) % 3 == 0) or (gen == DE_GENS - 1):
            best_idx = best_idx_by_loss(fit)
            best_global_theta = pop[best_idx].copy()
            best_global_fit = float(fit[best_idx])
            best_global_asr = float(asr_cache[best_idx])
            best_global_edge_ratio = float(edge_ratio_cache[best_idx])
            best_global_edge_cnt = int(edge_cnt_cache[best_idx])
            best_global_gen = int(gen + 1)
            save_checkpoint(
                CKPT_LATEST_PATH, gen + 1, G,
                pop, fit, asr_cache, edge_ratio_cache, edge_cnt_cache,
                Fv, CRv, dn, cs, bl, br,
                best_global_theta, best_global_fit, best_global_asr, best_global_edge_ratio, best_global_edge_cnt, best_global_gen,
                stagnation, dual_archive, strategy_pool, meta_learner, is_meta_guided,
                best_hist_fit=best_hist_fit,
                best_hist_asr=best_hist_asr,
                best_hist_theta=best_hist_theta,
                best_full_theta=best_fit_full_theta,
                best_full_asr=float(best_fit_full_asr),
                best_full_fit=float(best_fit_full_fit),
                calibrator=calibrator,
            )
            log(f"[CKPT] gen={gen + 1} saved -> {CKPT_LATEST_PATH}", level=1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="ckpt")

        t_gen1 = time.time()
        last_gen_time = (t_gen1 - t_gen0)
        st2 = pop_stats(asr_cache, fit)
        log(
            f"[GEN_END] gen={gen} time={last_gen_time:.2f}s accept={accept_cnt} "
            f"best_loss={st2.get('best_loss', np.inf):.6f} ASR_of_best_loss={st2.get('asr_of_best_loss', -1.0) * 100:.2f}%",
            level=1,
        )
        _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=gen, tag="gen_end")

        should_save = (gen == 0) or (((gen + 1) % SAVE_BEST_SNAPSHOT_EVERY_GENS) == 0)
        if should_save and SAVE_IMAGES_ON_BEST_FULL:
            if pending_theta is None and best_hist_theta is not None:
                pending_theta = best_hist_theta.copy()
                pending_asr = float(best_hist_asr)
                pending_fit = float(best_hist_fit)
                pending_reason = f"periodic_gen{gen}"
                pending_gen = int(gen)
            _maybe_flush_pending_snapshot(gen, force=True)

    final_theta = best_fit_full_theta if best_fit_full_theta is not None else (pop[best_idx_by_loss(fit)] if len(pop) else None)

    if final_theta is not None:
        fit_final, final_asr, _, _ = evaluate_theta_full(final_theta, imgs_rgb_in, bboxes_list_in, true_labels_in, G=G)
        update_best_fit_full(final_theta, fit_final, final_asr, reason="final")
        update_best_hist_loss(final_theta, fit_final, final_asr, gen=gen_last, reason="final")
        log(f"[FINAL] best_loss={best_hist_fit:.6f} ASR_of_best_loss={best_hist_asr * 100:.2f}% (denom={MAX_SAMPLES})", level=0)
    else:
        log("[FINAL] no valid theta", level=0)

    _maybe_log_query_curve(asr_cache, fit, best_hist_fit, gen=DE_GENS, tag="final")
    _maybe_flush_pending_snapshot(DE_GENS, force=True)
    return final_theta, float(best_hist_fit), float(best_hist_asr)


def run_one(cfg: dict, det_model):
    global ROI_CORE_DIV
    global gpu_evaluator
    global MAX_SAMPLES, DE_POP, DE_GENS, G, MASTER_SIZE
    global PATCH_COLOR_RGB, DARKEN_STRENGTH
    global LAMBDA_TOPO, LAMBDA_PHYS, CVAR_Q, EV_SUBSPACE_RANK

    IMG_DIR = cfg.get("img_dir", cfg.get("IMG_DIR", cfg.get("clean_dir", cfg.get("CLEAN_DIR", ""))))
    OUT_DIR = cfg.get("out_dir", cfg.get("OUT_DIR", "./out_run_one"))
    ROI_CORE_DIV = float(cfg.get("roi_core_div", cfg.get("ROI_CORE_DIV", ROI_CORE_DIV)))
    ROI_CORE_DIV = max(1.0, ROI_CORE_DIV)
    os.makedirs(OUT_DIR, exist_ok=True)

    CLIP_MODEL_ID = cfg.get("clip_model_id", cfg.get("CLIP_MODEL_ID", None))
    CLIP_CKPT = cfg.get("clip_ckpt", cfg.get("CLIP_CKPT", None))
    if not IMG_DIR or (not os.path.isdir(IMG_DIR)):
        raise FileNotFoundError(f"[RUN] IMG_DIR not found: {IMG_DIR}")
    if not CLIP_MODEL_ID:
        raise ValueError("[RUN] CLIP_MODEL_ID is None")

    MAX_SAMPLES = int(cfg.get("max_samples", cfg.get("MAX_SAMPLES", MAX_SAMPLES)))
    PERSON_ID = int(cfg.get("person_id", cfg.get("PERSON_ID", 0)))
    G = int(cfg.get("G", G))
    MASTER_SIZE = int(cfg.get("master_size", cfg.get("MASTER_SIZE", MASTER_SIZE)))
    PATCH_COLOR_RGB = tuple(cfg.get("patch_color", cfg.get("PATCH_COLOR_RGB", PATCH_COLOR_RGB)))
    DARKEN_STRENGTH = float(cfg.get("darken_strength", cfg.get("DARKEN_STRENGTH", DARKEN_STRENGTH)))
    DE_POP = int(cfg.get("pop_size", cfg.get("DE_POP", DE_POP)))
    DE_GENS = int(cfg.get("de_gens", cfg.get("DE_GENS", DE_GENS)))
    LAMBDA_TOPO = float(cfg.get("lambda_topo", cfg.get("LAMBDA_TOPO", LAMBDA_TOPO)))
    LAMBDA_PHYS = float(cfg.get("lambda_phys", cfg.get("LAMBDA_PHYS", LAMBDA_PHYS)))
    CVAR_Q = float(cfg.get("cvar_q", cfg.get("CVAR_Q", CVAR_Q)))
    EV_SUBSPACE_RANK = int(cfg.get("ev_rank", cfg.get("EV_SUBSPACE_RANK", EV_SUBSPACE_RANK)))

    log(f"[RUN] out_dir={OUT_DIR}", level=0)
    log(f"[RUN] img_dir={IMG_DIR}", level=0)
    log(f"[RUN] clip_model_id={CLIP_MODEL_ID}", level=0)
    log(f"[RUN] G={G} MASTER_SIZE={MASTER_SIZE} pop={DE_POP} gens={DE_GENS}", level=0)
    log(f"[RUN] objective=LOSS_ONLY | q={CVAR_Q} rank={EV_SUBSPACE_RANK} lam_topo={LAMBDA_TOPO} lam_budget={LAMBDA_PHYS}", level=0)

    init_clip_batch(CLIP_MODEL_ID=CLIP_MODEL_ID, CLIP_CKPT=CLIP_CKPT)
    kept_paths, bboxes_list, true_labels = build_eval_set_sequential_top1_person(
        img_dir=IMG_DIR,
        det_model=det_model,
        max_samples=MAX_SAMPLES,
        recursive=True,
        person_id=PERSON_ID,
        thr=SCORE_THR,
    )

    if len(kept_paths) == 0:
        raise RuntimeError("[RUN] kept_paths=0, screening failed. Check detector/full-image CLIP/top1 logic.")

    imgs_rgb = []
    kept_paths_aligned = []
    bboxes_list_aligned = []
    true_labels_aligned = []
    img_cache = {}
    for p, bbs, lbl in zip(kept_paths, bboxes_list, true_labels):
        rgb = img_cache.get(p, None)
        if rgb is None:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            elif bgr.ndim == 3 and bgr.shape[2] == 4:
                bgr = bgr[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_cache[p] = rgb
        imgs_rgb.append(rgb)
        kept_paths_aligned.append(p)
        bboxes_list_aligned.append(bbs)
        true_labels_aligned.append(lbl)

    kept_paths = kept_paths_aligned
    bboxes_list = bboxes_list_aligned
    true_labels = true_labels_aligned
    imgs_rgb, bboxes_list, true_labels, kept_paths = flatten_to_bbox_samples(imgs_rgb, bboxes_list, true_labels, kept_paths)
    if len(imgs_rgb) == 0:
        raise RuntimeError("[RUN] no bbox-samples after flatten.")
    log(f"[DATA] final eval set (bbox-samples) size={len(imgs_rgb)}", level=0)

    gpu_evaluator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gpu_evaluator = FastGPUFullImageEvaluator(imgs_rgb, bboxes_list, _clip_model, _clip_processor, DEVICE)
    clean_feats = _compute_clean_ev_feats_with_gpu_evaluator(batch=64)
    _build_ev_subspace_and_topo_sigma(clean_feats, rank=EV_SUBSPACE_RANK)
    CKPT_LATEST_PATH = os.path.join(OUT_DIR, "de_ckpt_latest.npz")

    try:
        final_theta, best_loss, asr_of_best_loss = de_optimize_pde_metade_hybrid(
            OUT_DIR=OUT_DIR,
            CKPT_LATEST_PATH=CKPT_LATEST_PATH,
            imgs_rgb_in=imgs_rgb,
            bboxes_list_in=bboxes_list,
            true_labels_in=true_labels,
            kept_paths_in=kept_paths,
            G=G,
        )
        export_theta_pack_from_ckpt(ckpt_path=CKPT_LATEST_PATH, out_dir=OUT_DIR, export_name="exported_uap_theta_pack.npz")
        log(f"[OK] run finished: best_loss={best_loss:.6f} ASR_of_best_loss={asr_of_best_loss * 100:.2f}% out={OUT_DIR}", level=0)
        return final_theta, float(best_loss), float(asr_of_best_loss)
    finally:
        try:
            if gpu_evaluator is not None:
                try:
                    gpu_evaluator.clear_mask_cache()
                except Exception:
                    pass
        except Exception:
            pass
        reset_per_run_globals()


def _write_json_file(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_text_file(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _save_single_run_result(run_out_dir: str, result: dict):
    payload = dict(result)
    _write_json_file(os.path.join(run_out_dir, "result_summary.json"), payload)

    asr = payload.get("asr_of_best_loss", None)
    asr_str = "NA" if asr is None else f"{float(asr) * 100.0:.2f}%"
    loss = payload.get("best_loss", None)
    loss_str = "NA" if loss is None else f"{float(loss):.6f}"
    text = (
        f"patch_color_rgb: {tuple(payload.get('patch_color_rgb', (0, 0, 0)))}\n"
        f"model: {payload.get('model', '')}\n"
        f"lambda_topo: {float(payload.get('lambda_topo', 0.0)):.2f}\n"
        f"lambda_budget: {float(payload.get('lambda_budget', 0.0)):.2f}\n"
        f"asr_of_best_loss: {asr_str}\n"
        f"best_loss: {loss_str}\n"
        f"status: {payload.get('status', 'unknown')}\n"
        f"out_dir: {payload.get('out_dir', '')}\n"
    )
    if payload.get("error"):
        text += f"error: {payload['error']}\n"
    _write_text_file(os.path.join(run_out_dir, "result_summary.txt"), text)


RUNS = []


def configure_detector(config: str = None, checkpoint: str = None, score_thr: float = None):
    global DETECTOR_CONFIG, DETECTOR_CKPT, SCORE_THR
    if config is not None:
        DETECTOR_CONFIG = str(config)
    if checkpoint is not None:
        DETECTOR_CKPT = str(checkpoint)
    if score_thr is not None:
        SCORE_THR = float(score_thr)


def configure_runtime(
    seed: int = None,
    device: str = None,
    log_level: int = None,
    clip_batch_size: int = None,
    save_images_on_best_full: bool = None,
):
    global RANDOM_SEED, DEVICE, LOG_LEVEL, CLIP_BATCH_SIZE, SAVE_IMAGES_ON_BEST_FULL
    if seed is not None:
        RANDOM_SEED = int(seed)
    if device is not None:
        DEVICE = str(device)
    if log_level is not None:
        LOG_LEVEL = int(log_level)
    if clip_batch_size is not None:
        CLIP_BATCH_SIZE = int(clip_batch_size)
    if save_images_on_best_full is not None:
        SAVE_IMAGES_ON_BEST_FULL = bool(save_images_on_best_full)


def init_detector_from_config(config: str = None, checkpoint: str = None, score_thr: float = None):
    configure_detector(config=config, checkpoint=checkpoint, score_thr=score_thr)
    det_ckpt_local = _maybe_copy_ckpt_to_local(DETECTOR_CKPT)
    log(f"[DET] init_detector: cfg={DETECTOR_CONFIG} ckpt={det_ckpt_local}", level=0)
    if init_detector is None:
        raise RuntimeError("[DET] init_detector is None. mmdet import failed; cannot run detector.")
    if not DETECTOR_CONFIG or not os.path.isfile(DETECTOR_CONFIG):
        raise FileNotFoundError(f"[DET] detector config not found: {DETECTOR_CONFIG}")
    if not det_ckpt_local or not os.path.isfile(det_ckpt_local):
        raise FileNotFoundError(f"[DET] detector checkpoint not found: {det_ckpt_local}")
    return init_detector(DETECTOR_CONFIG, det_ckpt_local, device=DEVICE)


def run_many(runs, detector_config: str, detector_ckpt: str, score_thr: float = SCORE_THR):
    set_global_seed(RANDOM_SEED)
    det_model = init_detector_from_config(detector_config, detector_ckpt, score_thr=score_thr)
    results = []
    for model_idx, cfg in enumerate(list(runs), start=1):
        name = cfg.get("name", f"run_{model_idx - 1}")
        run_cfg = dict(cfg)
        result_record = {
            "model": name,
            "patch_color_rgb": list(run_cfg.get("patch_color", run_cfg.get("PATCH_COLOR_RGB", PATCH_COLOR_RGB))),
            "lambda_topo": float(run_cfg.get("lambda_topo", run_cfg.get("LAMBDA_TOPO", LAMBDA_TOPO))),
            "lambda_budget": float(run_cfg.get("lambda_phys", run_cfg.get("LAMBDA_PHYS", LAMBDA_PHYS))),
            "G": int(run_cfg.get("G", G)),
            "roi_core_div": float(run_cfg.get("roi_core_div", run_cfg.get("ROI_CORE_DIV", ROI_CORE_DIV))),
            "out_dir": run_cfg.get("out_dir", run_cfg.get("OUT_DIR", "./outputs/run_one")),
            "status": "failed",
            "best_loss": None,
            "asr_of_best_loss": None,
            "error": None,
        }
        try:
            reset_per_run_globals()
            log(f"\n>>> [MODEL {model_idx}/{len(runs)}] {name}", level=0)
            _, best_loss, asr_of_best_loss = run_one(run_cfg, det_model)
            result_record["best_loss"] = float(best_loss)
            result_record["asr_of_best_loss"] = float(asr_of_best_loss)
            result_record["status"] = "ok"
        except Exception as e:
            result_record["error"] = repr(e)
            log(f"[ERR] Run failed for {name}: {repr(e)}", level=0)
            traceback.print_exc()
        finally:
            _save_single_run_result(result_record["out_dir"], result_record)
            reset_per_run_globals()
        results.append(result_record)
    return results

def main():
    set_global_seed(RANDOM_SEED)
    det_model = init_detector_from_config(DETECTOR_CONFIG, DETECTOR_CKPT, score_thr=SCORE_THR)

    if not RUNS:
        log("[ERR] RUNS list is empty. Use `python -m ucgp.cli --config configs/paper_default.yaml`.", level=0)
        return

    log(
        f"\n[RUN START] Total Models: {len(RUNS)} | "
        f"G=5 | delta={EXPANSION_RATIO:.2f} | pop={DE_POP} | gens={DE_GENS} | "
        f"lambda_topo={LAMBDA_TOPO:.2f} | lambda_budget={LAMBDA_PHYS:.2f} | "
        f"roi_core_div={ROI_CORE_DIV:.2f} | thickness_ratio={THICKNESS_TO_CELL_RATIO:.2f} | "
        f"patch_color={PATCH_COLOR_RGB} | max_samples={MAX_SAMPLES} | Objective=LOSS_ONLY\n",
        level=0,
    )

    for model_idx, cfg in enumerate(RUNS, start=1):
        name = cfg.get("name", f"run_{model_idx - 1}")
        run_cfg = dict(cfg)
        run_cfg["G"] = 5
        run_cfg["roi_core_div"] = 4.0
        run_cfg["lambda_topo"] = 0.12
        run_cfg["lambda_phys"] = 0.03
        run_cfg["patch_color"] = (0, 0, 0)
        run_cfg["max_samples"] = 300
        run_cfg["DE_POP"] = 50
        run_cfg["DE_GENS"] = 100

        result_record = {
            "model": name,
            "patch_color_rgb": [0, 0, 0],
            "lambda_topo": 0.12,
            "lambda_budget": 0.03,
            "G": 5,
            "roi_core_div": 4.0,
            "out_dir": run_cfg.get("out_dir", run_cfg.get("OUT_DIR", "./out_run_one")),
            "status": "failed",
            "best_loss": None,
            "asr_of_best_loss": None,
            "error": None,
        }

        try:
            reset_per_run_globals()
            log(
                f"\n>>> [MODEL {model_idx}/{len(RUNS)}] {name} | "
                f"G=5 | delta={EXPANSION_RATIO:.2f} | pop=50 | gens=100 | "
                f"lambda_topo=0.12 | lambda_budget=0.03 | roi_core_div=4.0 | patch_color=(0, 0, 0)",
                level=0,
            )
            log(f"[RUN] IMG_DIR={run_cfg.get('img_dir', run_cfg.get('IMG_DIR', ''))}", level=0)
            log(f"[RUN] OUT_DIR={run_cfg.get('out_dir', run_cfg.get('OUT_DIR', ''))}\n", level=0)

            _, best_loss, asr_of_best_loss = run_one(run_cfg, det_model)
            result_record["best_loss"] = float(best_loss)
            result_record["asr_of_best_loss"] = float(asr_of_best_loss)
            result_record["status"] = "ok"
        except Exception as e:
            result_record["error"] = repr(e)
            log(f"[ERR] Run failed for {name}: {repr(e)}", level=0)
            traceback.print_exc()
        finally:
            _save_single_run_result(result_record["out_dir"], result_record)
            reset_per_run_globals()

    log(
        f"[DONE] Fixed-parameter runs finished | G=5 | delta={EXPANSION_RATIO:.2f} | pop=50 | gens=100 | "
        f"lambda_topo=0.12 | lambda_budget=0.03 | roi_core_div=4.0 | patch_color=(0, 0, 0) | max_samples=300 | objective=LOSS_ONLY",
        level=0
    )

if __name__ == "__main__":
    main()
