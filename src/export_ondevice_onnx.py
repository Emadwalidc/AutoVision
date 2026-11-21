"""Export, dynamically quantize, and validate AutoVision v2 ONNX models for offline Android inference."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
BACKEND_DIR = ROOT / "backend"
for _p in (str(ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))

import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
import torchvision.transforms.functional as TF
from ultralytics import YOLO

from backend.config import load_settings
from backend.ml.classifier import PyTorchAutoVisionClassifier
from color_expert import (
    AutoVisionColorExpert,
    COLOR_CLASSES_V2,
    COLOR_GROUPS_V2,
    COLOR_HEX_V2,
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_neutral_gray_mask,
    extract_masked_lab_features,
)
from multitask_model import (
    CLIP_MEAN,
    CLIP_STD,
    AutoVisionMultiTaskNet,
    year_logits_to_expected,
)


class AutoVisionV2CoreOnnxWrapper(nn.Module):
    """Lean ViT-B/16 ONNX wrapper omitting unused Phase-1 color head and hooks."""

    def __init__(self, core: AutoVisionMultiTaskNet) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_feat = self.core.visual(x)
        emb = F.normalize(raw_feat.float(), dim=-1)
        make_logits = self.core.make_head(emb)
        model_logits = self.core.model_head(emb, targets=None)
        body_logits = self.core.body_head(emb)
        year_logits = self.core.year_head(emb)
        return emb, make_logits, model_logits, body_logits, year_logits


def _make_ort_session(onnx_path: Path) -> InferenceSession:
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    return InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _quantize_to_uint8(fp32_path: Path, dst_path: Path) -> None:
    if dst_path.is_file():
        dst_path.unlink()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dst_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
        use_external_data_format=False,
    )


def _compress_weights_int8_dequant(
    fp32_path: Path,
    dst_path: Path,
    min_numel: int = 1024,
) -> int:
    """Compresses 2D/4D FLOAT initializers into per-channel INT8 with DequantizeLinear (FP32 compute)."""
    if dst_path.is_file():
        dst_path.unlink()

    model = onnx.load(str(fp32_path), load_external_data=False)
    output_names = {o.name for o in model.graph.output}

    kept_inits = []
    added_inits = []
    dequant_nodes = []
    converted = 0

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT and init.name not in output_names:
            arr = numpy_helper.to_array(init).astype(np.float32, copy=False)
            if arr.ndim >= 2 and arr.size >= min_numel:
                max_abs = np.max(np.abs(arr), axis=tuple(range(1, arr.ndim))).astype(np.float32)
                scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
                bcast_scale = scale.reshape((-1,) + (1,) * (arr.ndim - 1))
                q = np.clip(np.round(arr / bcast_scale), -127, 127).astype(np.int8)
                zp = np.zeros(arr.shape[0], dtype=np.int8)

                q_name = f"{init.name}__qint8"
                s_name = f"{init.name}__scale"
                z_name = f"{init.name}__zp"

                added_inits.append(numpy_helper.from_array(q, name=q_name))
                added_inits.append(numpy_helper.from_array(scale, name=s_name))
                added_inits.append(numpy_helper.from_array(zp, name=z_name))
                dequant_nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        inputs=[q_name, s_name, z_name],
                        outputs=[init.name],
                        name=f"dequant_{converted}",
                        axis=0,
                    )
                )
                converted += 1
                continue
        kept_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.extend(added_inits)

    existing_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dequant_nodes)
    model.graph.node.extend(existing_nodes)

    model.ir_version = min(int(model.ir_version), 10)
    onnx.save(model, str(dst_path))
    return converted


def _compress_weights_fp16_cast(
    fp32_path: Path,
    dst_path: Path,
    min_numel: int = 1024,
) -> int:
    """Fallback weight-only compression storing large initializers as FLOAT16 + Cast to FLOAT."""
    if dst_path.is_file():
        dst_path.unlink()

    model = onnx.load(str(fp32_path), load_external_data=False)
    output_names = {o.name for o in model.graph.output}

    kept_inits = []
    added_inits = []
    cast_nodes = []
    converted = 0

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT and init.name not in output_names:
            arr = numpy_helper.to_array(init)
            if arr.ndim >= 2 and arr.size >= min_numel:
                arr_f16 = np.clip(arr, -65504.0, 65504.0).astype(np.float16)
                f16_name = f"{init.name}__fp16"
                added_inits.append(numpy_helper.from_array(arr_f16, name=f16_name))
                cast_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[f16_name],
                        outputs=[init.name],
                        name=f"cast_fp16_{converted}",
                        to=TensorProto.FLOAT,
                    )
                )
                converted += 1
                continue
        kept_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.extend(added_inits)

    existing_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(cast_nodes)
    model.graph.node.extend(existing_nodes)

    model.ir_version = min(int(model.ir_version), 10)
    onnx.save(model, str(dst_path))
    return converted


def _quick_core_embedding_cos(
    clf: PyTorchAutoVisionClassifier,
    core_onnx: Path,
    base_zip: Path,
    eu_zip: Path | None,
) -> float:
    sampled = _sample_zip_images(base_zip, 4, eu_zip, 1, seed=42)
    crops = [im for _, im in sampled]
    try:
        pt_out = clf.classify_crops(crops, use_tta=False)
        x_clip = _preprocess_clip_batch(crops)
        sess = _make_ort_session(core_onnx)
        onnx_embs = [sess.run(None, {"image": x_clip[i : i + 1]})[0][0] for i in range(len(crops))]
        onnx_arr = np.stack(onnx_embs, axis=0)
        return float(np.mean(np.sum(pt_out.embeddings * onnx_arr, axis=1)))
    finally:
        for im in crops:
            im.close()


def _export_fp32_core(
    models_dir: Path,
    lmaps: dict[str, Any],
    fp32_out: Path,
) -> None:
    ckpt_path = models_dir / "autovision_multitask_best.pt"
    year_min = int(lmaps["year_min"])
    year_max = int(lmaps["year_max"])

    net = AutoVisionMultiTaskNet(
        num_makes=len(lmaps["make_classes"]),
        num_models=len(lmaps["model_classes"]),
        num_bodies=len(lmaps["body_classes"]),
        num_years=year_max - year_min + 1,
        num_colors=len(lmaps["color_classes"]),
        pretrained=None,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    wrapper = AutoVisionV2CoreOnnxWrapper(net).eval()
    dummy_x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_x,),
            str(fp32_out),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "embedding",
                "make_logits",
                "model_logits",
                "body_logits",
                "year_logits",
            ],
            dynamic_axes={
                "image": {0: "batch"},
                "embedding": {0: "batch"},
                "make_logits": {0: "batch"},
                "model_logits": {0: "batch"},
                "body_logits": {0: "batch"},
                "year_logits": {0: "batch"},
            },
            dynamo=False,
        )


def _export_fp32_color(
    color_models_dir: Path,
    clm: dict[str, Any],
    fp32_out: Path,
) -> None:
    ckpt_path = color_models_dir / "autovision_color_expert_v2.pt"
    color_classes = list(clm.get("color_classes") or COLOR_CLASSES_V2)

    net = AutoVisionColorExpert(
        num_colors=len(color_classes),
        lab_dim=16,
        pretrained=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    dummy_x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    dummy_lab = torch.randn(1, 16, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            net,
            (dummy_x, dummy_lab),
            str(fp32_out),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image", "lab_feats"],
            output_names=["color_logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "lab_feats": {0: "batch"},
                "color_logits": {0: "batch"},
            },
            dynamo=False,
        )


def _export_yolo_seg(yolo_pt: Path, dst_onnx: Path, tmp_dir: Path) -> dict[str, Any]:
    local_pt = tmp_dir / yolo_pt.name
    shutil.copy2(yolo_pt, local_pt)
    model = YOLO(str(local_pt))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            simplify=False,
            half=False,
            dynamic=False,
        )
    )
    shutil.copy2(exported, dst_onnx)

    sess = _make_ort_session(dst_onnx)
    inputs = [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in sess.get_inputs()]
    outputs = [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in sess.get_outputs()]
    onnx_model = onnx.load(str(dst_onnx))
    info = {
        "ir_version": int(onnx_model.ir_version),
        "opset": [int(op.version) for op in onnx_model.opset_import],
        "inputs": inputs,
        "outputs": outputs,
    }
    return info


def _build_prototypes_and_meta(
    models_dir: Path,
    eu_dir: Path | None,
    lmaps: dict[str, Any],
    clm: dict[str, Any],
    out_dir: Path,
) -> tuple[int, int]:
    with np.load(models_dir / "prototypes_v2.npz", allow_pickle=False) as d:
        protos = d["prototypes"].astype(np.float32)
        p_classes = [str(x) for x in d["class_names"]]
        p_makes = [str(x) for x in d["makes"]]
        p_bodies = [str(x) for x in d["body_types"]]
        p_splits = [str(x) for x in d["split_types"]]

    make_display_names = dict(lmaps.get("make_display_names") or {})
    model_display_names = dict(lmaps.get("model_display_names") or {})

    if eu_dir is not None and eu_dir.is_dir():
        disp_path = eu_dir / "eu_display_names.json"
        if disp_path.is_file():
            disp = json.loads(disp_path.read_text(encoding="utf-8"))
            make_display_names.update(disp.get("make_display_names", {}))
            model_display_names.update(disp.get("model_display_names", {}))

        eu_proto_path = eu_dir / "eu_prototypes.npz"
        if eu_proto_path.is_file():
            with np.load(eu_proto_path, allow_pickle=False) as d:
                ep = d["prototypes"].astype(np.float32)
                protos = np.vstack([protos, ep])
                p_classes.extend(str(x) for x in d["class_names"])
                p_makes.extend(str(x) for x in d["makes"])
                p_bodies.extend(str(x) for x in d["body_types"])
                p_splits.extend(str(x) for x in d["split_types"])

    norms = np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
    protos = (protos / norms).astype("<f4", copy=False)
    (out_dir / "prototypes_fp32.bin").write_bytes(protos.tobytes(order="C"))

    color_classes = list(clm.get("color_classes") or COLOR_CLASSES_V2)
    color_hex = dict(clm.get("color_hex") or COLOR_HEX_V2)
    color_groups = {c: COLOR_GROUPS_V2.get(c, c) for c in color_classes}

    unique_classes = len(dict.fromkeys(p_classes))
    meta = {
        "model_version": "autovision-v2-ondevice",
        "year_min": int(lmaps["year_min"]),
        "year_max": int(lmaps["year_max"]),
        "model_temperature": 2.15,
        "cosface_scale": 30.0,
        "num_prototypes": int(protos.shape[0]),
        "embed_dim": int(protos.shape[1]),
        "num_unique_prototype_classes": int(unique_classes),
        "make_classes": list(lmaps["make_classes"]),
        "model_classes": list(lmaps["model_classes"]),
        "body_classes": list(lmaps["body_classes"]),
        "color_classes": color_classes,
        "color_hex": color_hex,
        "color_groups": color_groups,
        "make_display_names": make_display_names,
        "model_display_names": model_display_names,
        "proto_classes": p_classes,
        "proto_makes": p_makes,
        "proto_bodies": p_bodies,
        "proto_split_types": p_splits,
    }
    (out_dir / "autovision_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return int(protos.shape[0]), unique_classes


def _sample_zip_images(
    base_zip: Path,
    n_base: int,
    eu_zip: Path | None,
    n_eu: int,
    seed: int = 42,
) -> list[tuple[str, Image.Image]]:
    rng = random.Random(seed)
    images: list[tuple[str, Image.Image]] = []

    with zipfile.ZipFile(base_zip, "r") as zf:
        candidates = [
            n
            for n in zf.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.endswith("/")
        ]
        chosen = rng.sample(candidates, min(n_base, len(candidates)))
        for name in chosen:
            with zf.open(name) as f:
                im = Image.open(io.BytesIO(f.read())).convert("RGB")
                images.append((name, im))

    if eu_zip is not None and eu_zip.is_file() and n_eu > 0:
        with zipfile.ZipFile(eu_zip, "r") as zf:
            candidates = [
                n
                for n in zf.namelist()
                if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.endswith("/")
            ]
            chosen = rng.sample(candidates, min(n_eu, len(candidates)))
            for name in chosen:
                with zf.open(name) as f:
                    im = Image.open(io.BytesIO(f.read())).convert("RGB")
                    images.append((name, im))

    return images


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)


def _preprocess_clip_batch(crops: list[Image.Image]) -> np.ndarray:
    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    arrs = []
    for c in crops:
        r = TF.resize(c, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        t = (np.asarray(r, dtype=np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        arrs.append(t.astype(np.float32))
        r.close()
    return np.stack(arrs, axis=0)


def _preprocess_color_batch(crops: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    imgs = []
    labs = []
    for c in crops:
        h, w = c.height, c.width
        mask = np.zeros((h, w), dtype=bool)
        mask[int(0.12 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)] = True
        masked_rgb, final_mask = apply_neutral_gray_mask(c, mask, erode_px=2)
        rgb_224 = TF.resize(masked_rgb, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        mask_pil = Image.fromarray((final_mask.astype(np.uint8) * 255), mode="L")
        mask_224_pil = TF.resize(mask_pil, [224, 224], interpolation=TF.InterpolationMode.NEAREST)
        mask_224 = np.asarray(mask_224_pil) >= 128
        if int(mask_224.sum()) < 32:
            mask_224 = np.ones((224, 224), dtype=bool)
        lab_vec = extract_masked_lab_features(np.asarray(rgb_224, dtype=np.uint8), mask_224)
        img_t = (np.asarray(rgb_224, dtype=np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        imgs.append(img_t.astype(np.float32))
        labs.append(lab_vec.astype(np.float32))
        for tmp in (mask_224_pil, mask_pil, rgb_224, masked_rgb):
            tmp.close()
    return np.stack(imgs, axis=0), np.stack(labs, axis=0)


def _letterbox_640(im: Image.Image) -> np.ndarray:
    w, h = im.size
    scale = min(640.0 / max(w, 1), 640.0 / max(h, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x = (640 - nw) // 2
    pad_y = (640 - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    resized.close()
    canvas.close()
    return np.expand_dims(arr.astype(np.float32), axis=0)


def validate_onnx_parity(
    clf: PyTorchAutoVisionClassifier,
    core_onnx: Path,
    color_onnx: Path,
    yolo_onnx: Path,
    base_zip: Path,
    eu_zip: Path | None,
) -> dict[str, float]:
    sampled = _sample_zip_images(base_zip, 25, eu_zip, 5, seed=42)
    crops = [im for _, im in sampled]
    try:
        pt_out = clf.classify_crops(crops, use_tta=False)

        core_sess = _make_ort_session(core_onnx)
        x_clip = _preprocess_clip_batch(crops)
        onnx_embs = []
        onnx_mk = []
        onnx_md = []
        onnx_bd = []
        onnx_yr = []
        for i in range(len(crops)):
            emb_i, mk_i, md_i, bd_i, yr_i = core_sess.run(None, {"image": x_clip[i : i + 1]})
            onnx_embs.append(emb_i[0])
            onnx_mk.append(_softmax(mk_i[0]))
            onnx_md.append(_softmax(md_i[0] / 2.15))
            onnx_bd.append(_softmax(bd_i[0]))
            yr_e, _ = year_logits_to_expected(
                torch.from_numpy(yr_i),
                year_min=int(clf.label_maps["year_min"]),
                year_max=int(clf.label_maps["year_max"]),
            )
            onnx_yr.append(float(yr_e[0].item()))

        onnx_embs_arr = np.stack(onnx_embs, axis=0)
        onnx_mk_arr = np.stack(onnx_mk, axis=0)
        onnx_md_arr = np.stack(onnx_md, axis=0)
        onnx_bd_arr = np.stack(onnx_bd, axis=0)
        onnx_yr_arr = np.asarray(onnx_yr, dtype=np.float32)

        color_sess = _make_ort_session(color_onnx)
        x_col, lab_col = _preprocess_color_batch(crops)
        onnx_col = []
        for i in range(len(crops)):
            (c_logits,) = color_sess.run(
                None,
                {"image": x_col[i : i + 1], "lab_feats": lab_col[i : i + 1]},
            )
            onnx_col.append(_softmax(c_logits[0]))
        onnx_col_arr = np.stack(onnx_col, axis=0)

        cos_sims = np.sum(pt_out.embeddings * onnx_embs_arr, axis=1)
        mean_cos = float(np.mean(cos_sims))
        make_agree = float(np.mean(np.argmax(pt_out.make_probs, 1) == np.argmax(onnx_mk_arr, 1)))

        pt_md_top1 = np.argmax(pt_out.model_probs, axis=1)
        onnx_md_top1 = np.argmax(onnx_md_arr, axis=1)
        model_agree = float(np.mean(pt_md_top1 == onnx_md_top1))
        onnx_md_top3 = np.argsort(-onnx_md_arr, axis=1)[:, :3]
        model_top3_agree = float(np.mean([t in onnx_md_top3[i] for i, t in enumerate(pt_md_top1)]))

        body_agree = float(np.mean(np.argmax(pt_out.body_probs, 1) == np.argmax(onnx_bd_arr, 1)))
        year_diff = float(np.mean(np.abs(pt_out.year_expected - onnx_yr_arr)))
        color_agree = float(np.mean(np.argmax(pt_out.color_probs, 1) == np.argmax(onnx_col_arr, 1)))

        yolo_sess = _make_ort_session(yolo_onnx)
        yolo_ok = 0
        for i in range(min(5, len(crops))):
            inp = _letterbox_640(crops[i])
            out0, out1 = yolo_sess.run(None, {"images": inp})
            assert out0.shape == (1, 116, 8400), f"Unexpected YOLO output0 shape: {out0.shape}"
            assert out1.shape == (1, 32, 160, 160), f"Unexpected YOLO output1 shape: {out1.shape}"
            veh_scores = out0[0, [4 + 2, 4 + 5, 4 + 7], :]
            if float(np.max(veh_scores)) >= 0.25:
                yolo_ok += 1

        metrics = {
            "num_samples": float(len(crops)),
            "mean_emb_cosine": mean_cos,
            "make_top1_agree": make_agree,
            "model_top1_agree": model_agree,
            "model_top3_agree": model_top3_agree,
            "body_top1_agree": body_agree,
            "year_mae_diff": year_diff,
            "color_top1_agree": color_agree,
            "yolo_vehicle_hits_of_5": float(yolo_ok),
        }
        print("ONNX_PARITY_METRICS:", json.dumps(metrics, indent=2))

        assert mean_cos >= 0.97, f"Embedding cosine too low: {mean_cos:.4f}"
        assert make_agree >= 0.93, f"Make Top-1 agreement too low: {make_agree:.4f}"
        assert model_agree >= 0.90, f"Model Top-1 agreement too low: {model_agree:.4f}"
        assert model_top3_agree >= 0.96, f"Model Top-3 agreement too low: {model_top3_agree:.4f}"
        assert body_agree >= 0.93, f"Body Top-1 agreement too low: {body_agree:.4f}"
        assert year_diff <= 0.50, f"Year MAE delta too high: {year_diff:.4f}"
        assert color_agree >= 0.93, f"Color Top-1 agreement too low: {color_agree:.4f}"
        assert yolo_ok >= 3, f"YOLOv8n-seg ONNX detected vehicles on too few sample images: {yolo_ok}/5"
        return metrics
    finally:
        for im in crops:
            im.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate AutoVision v2 ONNX assets for Android.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "mobile" / "ondevice_assets",
        help="Destination folder for quantized ONNX models, prototypes, and metadata.",
    )
    args = parser.parse_args()

    settings = load_settings(ROOT)
    assert settings.color_models_dir is not None, "v2_color models directory not found."
    assert settings.yolo_seg_weights is not None and settings.yolo_seg_weights.is_file(), "yolov8n-seg.pt missing."
    assert settings.gallery_zip_path is not None and settings.gallery_zip_path.is_file(), "gallery_v2_thumbs.zip missing."

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lmaps = json.loads((settings.models_dir / "label_maps.json").read_text(encoding="utf-8"))
    clm = json.loads((settings.color_models_dir / "color_label_maps_v2.json").read_text(encoding="utf-8"))

    core_onnx = out_dir / "autovision_v2_core.onnx"
    color_onnx = out_dir / "autovision_color_v2.onnx"
    yolo_onnx = out_dir / "yolov8n_seg.onnx"

    clf = PyTorchAutoVisionClassifier(
        models_dir=settings.models_dir,
        device="cpu",
        color_models_dir=settings.color_models_dir,
        model_temperature=settings.model_logit_temperature,
    )
    eu_zip = (settings.eu_extension_dir / "eu_gallery_thumbs.zip") if settings.eu_extension_dir else None

    with tempfile.TemporaryDirectory(prefix="av_onnx_") as tmp_str:
        tmp_dir = Path(tmp_str)
        core_fp32 = tmp_dir / "core_fp32.onnx"
        color_fp32 = tmp_dir / "color_fp32.onnx"

        print("[1/5] Exporting ViT-B/16 multi-task core to FP32 ONNX...")
        _export_fp32_core(settings.models_dir, lmaps, core_fp32)
        print("[2/5] Compressing core weights (per-channel INT8 DequantizeLinear, FP32 compute)...")
        n_q = _compress_weights_int8_dequant(core_fp32, core_onnx)
        quick_cos = _quick_core_embedding_cos(clf, core_onnx, settings.gallery_zip_path, eu_zip)
        print(f"  -> weight-only INT8 ({n_q} tensors): 5-sample mean cosine = {quick_cos:.5f}")
        if quick_cos < 0.99:
            print("  -> mean cosine < 0.99; falling back to FP16 weight compression (Cast -> FLOAT)...")
            n_f16 = _compress_weights_fp16_cast(core_fp32, core_onnx)
            quick_cos_f16 = _quick_core_embedding_cos(clf, core_onnx, settings.gallery_zip_path, eu_zip)
            print(f"  -> weight-only FP16 ({n_f16} tensors): 5-sample mean cosine = {quick_cos_f16:.5f}")

        print("[3/5] Exporting Color Expert v2 to FP32 ONNX + QUInt8 quantization...")
        _export_fp32_color(settings.color_models_dir, clm, color_fp32)
        _quantize_to_uint8(color_fp32, color_onnx)

        print("[4/5] Exporting YOLOv8n-seg FP32 ONNX...")
        yolo_info = _export_yolo_seg(settings.yolo_seg_weights, yolo_onnx, tmp_dir)
        print("YOLO_ONNX_INFO:", json.dumps(yolo_info, indent=2))

        for name, p in [("core", core_onnx), ("color", color_onnx), ("yolo", yolo_onnx)]:
            m = onnx.load(str(p), load_external_data=False)
            assert int(m.ir_version) <= 10, f"{p.name} ir_version={m.ir_version} exceeds ORT Android limit (10)"
            print(
                f"  -> {name}: {p.name} ({p.stat().st_size / (1024 * 1024):.2f} MB), "
                f"ir={m.ir_version}, opsets={[o.version for o in m.opset_import]}"
            )

    n_protos, n_unique = _build_prototypes_and_meta(
        settings.models_dir,
        settings.eu_extension_dir,
        lmaps,
        clm,
        out_dir,
    )
    print(f"[5/5] Built prototypes_fp32.bin ({n_protos} rows, {n_unique} unique classes) & autovision_meta.json")

    validate_onnx_parity(
        clf=clf,
        core_onnx=core_onnx,
        color_onnx=color_onnx,
        yolo_onnx=yolo_onnx,
        base_zip=settings.gallery_zip_path,
        eu_zip=eu_zip,
    )

    total_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print(f"SUCCESS: Total mobile/ondevice_assets size = {total_bytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
