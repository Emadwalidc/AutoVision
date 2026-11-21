"""AutoVision v2 (Phase 1) - Multi-Task Training on Kaggle GPU (49 A-Z Makes, 1950-2026).

Trains a fresh OpenCLIP ViT-B/16 multi-task network (Make, Model CosFace, Body Type,
Interval-Aware DLDL Year 1950-2026) on:
  1. Auto-downloaded `timdwms/us-car-models-1948-2026` (49 makes, 555 models, up to 2026)
  2. Optional attached `autovision-vmmrdb-clean-89k` under `/kaggle/input/**` (everyday street photos)

Kaggle Notebook settings:
  - Accelerator: GPU T4 x2 (recommended) or GPU P100
  - Internet: ON
  - Install if needed: `!pip install -q open_clip_torch`

Outputs (`/kaggle/working/autovision_v2_phase1/`):
  - autovision_multitask_best.pt
  - label_maps.json
  - prototypes_v2.npz
  - features_finetuned_openclip_vitb16.npz
  - autovision_v2_manifest.csv
  - gallery_v2_thumbs.zip
  - training_history.csv
  - training_summary.json
  - v2_test_predictions.npz
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- Hyperparameters & Paths ---
HF_DATASET_BASE = "https://huggingface.co/datasets/timdwms/us-car-models-1948-2026/resolve/main"
WORK_CACHE_DIR = Path("/tmp/autovision_v2_data")
OUT_DIR = Path("/kaggle/working/autovision_v2_phase1")

SEED = 42
CROP_MAX_EDGE = 320
CROP_JPEG_QUALITY = 88
CROP_PAD = 0.05
MIN_SAMPLES_CLOSED = 20
MAX_TRAIN_PER_MODEL = 450
GALLERY_PER_MODEL = 35
MAX_GALLERY_ITEMS = 25000

PHASE1_WARMUP_EPOCHS = 1
WARMUP_HEAD_LR = 3e-3
PHASE1_EPOCHS = 9
BACKBONE_LR = 2.5e-5
PROJ_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.02
EFFECTIVE_BATCH_SIZE = 128
NUM_WORKERS = 2

# --- Embedded Taxonomies (Parity with src/multitask_model.py & src/v2_taxonomy.py) ---
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

YEAR_MIN: int = 1950
YEAR_MAX: int = 2026
NUM_YEAR_BINS: int = YEAR_MAX - YEAR_MIN + 1

BODY_TYPE_CLASSES: tuple[str, ...] = (
    "Sedan",
    "SUV",
    "Pickup",
    "Minivan",
    "Coupe",
    "Hatchback",
    "Wagon",
    "Convertible",
)

COLOR_CLASSES: tuple[str, ...] = (
    "Beige", "Black", "Blue", "Brown", "Gold", "Gray", "Green", "Orange",
    "Pink", "Purple", "Red", "Silver", "Tan", "White", "Yellow",
)

COLOR_GROUPS_11: dict[str, str] = {
    "Beige": "Beige/Gold", "Black": "Black", "Blue": "Blue", "Brown": "Brown",
    "Gold": "Beige/Gold", "Gray": "Gray", "Green": "Green", "Orange": "Orange",
    "Pink": "Purple/Pink", "Purple": "Purple/Pink", "Red": "Red",
    "Silver": "Silver", "Tan": "Beige/Gold", "White": "White", "Yellow": "Yellow",
}

COLOR_HEX: dict[str, str] = {
    "Beige": "#E5D3B3", "Black": "#1C1C1E", "Blue": "#2563EB", "Brown": "#7C4A2D",
    "Gold": "#D4AF37", "Gray": "#6B7280", "Green": "#16A34A", "Orange": "#EA580C",
    "Pink": "#EC4899", "Purple": "#7E22CE", "Red": "#DC2626", "Silver": "#C0C7CE",
    "Tan": "#C8A47E", "White": "#F8FAFC", "Yellow": "#EAB308",
}

_MODEL_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("dodge", "ram_1500"): ("ram", "1500"),
    ("dodge", "ram_hd"): ("ram", "ram_hd"),
    ("mercedes-benz", "sl"): ("mercedes-benz", "sl-class"),
    ("mini", "cooper_countryman"): ("mini", "countryman"),
    ("saab", "9-7"): ("saab", "9-7x"),
    ("toyota", "landcruiser"): ("toyota", "land_cruiser"),
    ("toyota", "fjcruiser"): ("toyota", "fj_cruiser"),
    ("mazda", "mx5_miata"): ("mazda", "mx-5_miata"),
    ("mazda", "rx-7"): ("mazda", "rx7"),
    ("mazda", "rx-8"): ("mazda", "rx8"),
}

_MAKE_DISPLAY_OVERRIDES: dict[str, str] = {
    "alfa-romeo": "Alfa Romeo", "bmw": "BMW", "gmc": "GMC", "land-rover": "Land Rover",
    "mercedes-benz": "Mercedes-Benz", "mg": "MG", "mini": "MINI", "ram": "RAM",
    "rolls-royce": "Rolls-Royce",
}

_TOKEN_DISPLAY_OVERRIDES: dict[str, str] = {
    "amg": "AMG", "ats": "ATS", "brz": "BRZ", "bz4x": "bZ4X", "c-hr": "C-HR",
    "c-max": "C-Max", "cc": "CC", "cl": "CL", "cl-class": "CL-Class",
    "cla-class": "CLA-Class", "clk-class": "CLK-Class", "cls-class": "CLS-Class",
    "cr-v": "CR-V", "cr-z": "CR-Z", "ct": "CT", "ct4": "CT4", "ct5": "CT5",
    "ct6": "CT6", "cts": "CTS", "cx3": "CX-3", "cx30": "CX-30", "cx5": "CX-5",
    "cx50": "CX-50", "cx7": "CX-7", "cx9": "CX-9", "cx90": "CX-90",
    "delsol": "del Sol", "deville": "DeVille", "e-pace": "E-Pace", "e-tron": "e-tron",
    "e350": "E-350", "elr": "ELR", "eqe": "EQE", "eqs": "EQS", "es": "ES", "ev": "EV",
    "ev6": "EV6", "ev9": "EV9", "ex30": "EX30", "ex90": "EX90", "f-pace": "F-Pace",
    "f-type": "F-Type", "f100": "F-100", "f150": "F-150", "f250-f350": "F-250/F-350",
    "fj": "FJ", "frs": "FR-S", "fx35": "FX35", "g70": "G70", "g80": "G80", "g90": "G90",
    "gl-class": "GL-Class", "gla-class": "GLA-Class", "glb-class": "GLB-Class",
    "glc-class": "GLC-Class", "gle-class": "GLE-Class", "glk-class": "GLK-Class",
    "gls-class": "GLS-Class", "grandam": "Grand Am", "grandmarquis": "Grand Marquis",
    "grandprix": "Grand Prix", "granturismo": "GranTurismo", "gs": "GS", "gt": "GT",
    "gti": "GTI", "gto": "GTO", "gtr": "GT-R", "gv60": "GV60", "gv70": "GV70",
    "gv80": "GV80", "gx": "GX", "hd": "HD", "hhr": "HHR", "hr-v": "HR-V", "hs": "HS",
    "i-miev": "i-MiEV", "i-pace": "I-Pace", "i3": "i3", "i30": "I30", "i8": "i8",
    "id.4": "ID.4", "ilx": "ILX", "iq": "iQ", "is": "IS", "ix": "iX", "k900": "K900",
    "lacrosse": "LaCrosse", "lc": "LC", "lesabre": "LeSabre", "ls": "LS", "lx": "LX",
    "m2": "M2", "m3": "M3", "m35": "M35", "m4": "M4", "m5": "M5", "mach-e": "Mach-E",
    "mdx": "MDX", "mkc": "MKC", "mks": "MKS", "mkt": "MKT", "mkx": "MKX", "mkz": "MKZ",
    "montecarlo": "Monte Carlo", "mpv": "MPV", "mr2": "MR2", "mx-5": "MX-5", "nsx": "NSX",
    "nv200": "NV200", "nx": "NX", "parkavenue": "Park Avenue", "promaster": "ProMaster",
    "pt": "PT", "q3": "Q3", "q4": "Q4", "q5": "Q5", "q50": "Q50", "q60": "Q60", "q7": "Q7",
    "q70": "Q70", "q8": "Q8", "qx30": "QX30", "qx4": "QX4", "qx50": "QX50", "qx55": "QX55",
    "qx56": "QX56", "qx60": "QX60", "qx80": "QX80", "r1s": "R1S", "r1t": "R1T", "r8": "R8",
    "ram": "Ram", "rangerover": "Range Rover", "rav4": "RAV4", "rc": "RC", "rdx": "RDX",
    "rl": "RL", "rlx": "RLX", "rsx": "RSX", "rx": "RX", "rx7": "RX-7", "rx8": "RX-8",
    "rz": "RZ", "s10": "S-10", "s2000": "S2000", "s4": "S4", "s40": "S40", "s60": "S60",
    "s70": "S70", "s80": "S80", "s90": "S90", "santafe": "Santa Fe",
    "sl-class": "SL-Class", "slk-class": "SLK-Class", "srx": "SRX", "sts": "STS",
    "suv": "SUV", "sx4": "SX4", "tc": "tC", "tl": "TL", "tlx": "TLX",
    "town&country": "Town & Country", "towncar": "Town Car", "transam": "Trans Am",
    "tsx": "TSX", "tt": "TT", "tx": "TX", "ux": "UX", "v50": "V50", "v60": "V60",
    "v70": "V70", "v90": "V90", "wrx": "WRX", "x1": "X1", "x2": "X2", "x3": "X3",
    "x4": "X4", "x5": "X5", "x6": "X6", "x7": "X7", "xb": "xB", "xc40": "XC40",
    "xc60": "XC60", "xc70": "XC70", "xc90": "XC90", "xd": "xD", "xe": "XE", "xf": "XF",
    "xj": "XJ", "xk": "XK", "xlr": "XLR", "xt4": "XT4", "xt5": "XT5", "xt6": "XT6",
    "xts": "XTS", "yukon": "Yukon", "zdx": "ZDX",
}

_V2_BODY_GROUPS: dict[str, tuple[str, ...]] = {
    "SUV": (
        "acura__mdx", "acura__rdx", "acura__zdx", "alfa-romeo__stelvio", "alfa-romeo__tonale",
        "audi__e-tron", "audi__q3", "audi__q4_e-tron", "audi__q5", "audi__q7", "audi__q8",
        "bmw__ix", "bmw__x1", "bmw__x2", "bmw__x3", "bmw__x4", "bmw__x5", "bmw__x6", "bmw__x7",
        "buick__enclave", "buick__encore", "buick__encore_gx", "buick__envision",
        "buick__envista", "buick__rendezvous", "cadillac__escalade", "cadillac__lyriq",
        "cadillac__srx", "cadillac__xt4", "cadillac__xt5", "cadillac__xt6",
        "chevrolet__blazer", "chevrolet__blazer_ev", "chevrolet__equinox",
        "chevrolet__equinox_ev", "chevrolet__suburban", "chevrolet__tahoe",
        "chevrolet__tracker", "chevrolet__trailblazer", "chevrolet__traverse",
        "chevrolet__trax", "chrysler__aspen", "dodge__durango", "dodge__hornet",
        "dodge__journey", "dodge__nitro", "fiat__500x", "ford__bronco", "ford__bronco_sport",
        "ford__ecosport", "ford__edge", "ford__escape", "ford__excursion", "ford__expedition",
        "ford__explorer", "ford__flex", "ford__mustang_mach-e", "genesis__gv60",
        "genesis__gv70", "genesis__gv80", "gmc__acadia", "gmc__envoy", "gmc__hummer_ev",
        "gmc__jimmy", "gmc__suburban_1500", "gmc__terrain", "gmc__yukon_1500",
        "honda__cr-v", "honda__element", "honda__hr-v", "honda__passport", "honda__pilot",
        "honda__prologue", "hummer__h2", "hummer__h3", "hyundai__kona", "hyundai__palisade",
        "hyundai__santafe", "hyundai__tucson", "hyundai__venue", "hyundai__veracruz",
        "infiniti__fx35", "infiniti__qx30", "infiniti__qx4", "infiniti__qx50",
        "infiniti__qx55", "infiniti__qx56", "infiniti__qx60", "infiniti__qx80",
        "isuzu__rodeo", "isuzu__trooper", "jaguar__e-pace", "jaguar__f-pace", "jaguar__i-pace",
        "jeep__cherokee", "jeep__commander", "jeep__compass", "jeep__grand_cherokee",
        "jeep__liberty", "jeep__patriot", "jeep__renegade", "jeep__wagoneer", "jeep__wrangler",
        "kia__borrego", "kia__ev6", "kia__ev9", "kia__niro", "kia__seltos", "kia__sorento",
        "kia__sportage", "kia__telluride", "land-rover__defender", "land-rover__discovery",
        "land-rover__discovery_sport", "land-rover__freelander", "land-rover__rangerover",
        "land-rover__rangerover_evoque", "land-rover__rangerover_sport",
        "land-rover__rangerover_velar", "lexus__gx", "lexus__lx", "lexus__nx", "lexus__rx",
        "lexus__rz", "lexus__tx", "lexus__ux", "lincoln__aviator", "lincoln__corsair",
        "lincoln__mkc", "lincoln__mkt", "lincoln__mkx", "lincoln__nautilus", "lincoln__navigator",
        "maserati__grecale", "maserati__levante", "mazda__cx3", "mazda__cx30", "mazda__cx5",
        "mazda__cx50", "mazda__cx7", "mazda__cx9", "mazda__cx90", "mazda__tribute",
        "mercedes-benz__eqe_suv", "mercedes-benz__eqs_suv", "mercedes-benz__g-class",
        "mercedes-benz__gl-class", "mercedes-benz__gla-class", "mercedes-benz__glb-class",
        "mercedes-benz__glc-class", "mercedes-benz__gle-class", "mercedes-benz__glk-class",
        "mercedes-benz__gls-class", "mercedes-benz__m-class", "mercury__mariner",
        "mercury__mountaineer", "mini__countryman", "mini__paceman", "mitsubishi__endeavor",
        "mitsubishi__montero", "mitsubishi__outlander", "nissan__ariya", "nissan__armada",
        "nissan__juke", "nissan__kicks", "nissan__murano", "nissan__pathfinder",
        "nissan__rogue", "nissan__x-trail", "nissan__xterra", "oldsmobile__bravada",
        "peugeot__3008", "plymouth__trailduster", "polestar__polestar_3", "pontiac__aztek",
        "pontiac__torrent", "porsche__cayenne", "porsche__macan", "renault__captur",
        "rivian__r1s", "saab__9-7x", "saturn__outlook", "saturn__vue", "subaru__ascent",
        "subaru__crosstrek", "subaru__forester", "subaru__solterra", "subaru__tribeca",
        "suzuki__grand_vitara", "suzuki__samurai", "suzuki__sidekick", "suzuki__vitara",
        "suzuki__x90", "suzuki__xl7", "tesla__model_x", "tesla__model_y", "toyota__4runner",
        "toyota__bz4x", "toyota__c-hr", "toyota__corolla_cross", "toyota__fj40",
        "toyota__fj_cruiser", "toyota__fortuner", "toyota__grand_highlander",
        "toyota__highlander", "toyota__land_cruiser", "toyota__rav4", "toyota__sequoia",
        "toyota__venza", "volkswagen__atlas", "volkswagen__id.4", "volkswagen__taos",
        "volkswagen__tiguan", "volkswagen__touareg", "volvo__c40", "volvo__ex30",
        "volvo__ex90", "volvo__xc40", "volvo__xc60", "volvo__xc90",
        "willys__cj2a", "willys__cj3b",
    ),
    "Pickup": (
        "chevrolet__avalanche", "chevrolet__c10", "chevrolet__colorado", "chevrolet__el_camino",
        "chevrolet__s10", "chevrolet__silverado", "chevrolet__silverado_ev", "dodge__dakota",
        "ford__f100", "ford__f150", "ford__f250-f350", "ford__maverick", "ford__ranger",
        "gmc__canyon", "gmc__sierra_1500", "gmc__sierra_hd", "gmc__sonoma", "honda__ridgeline",
        "hyundai__santa_cruz", "jeep__gladiator", "mitsubishi__mighty_max", "mitsubishi__raider",
        "nissan__720", "nissan__d21_hardbody", "nissan__frontier", "nissan__titan",
        "ram__1500", "ram__2500", "ram__3500", "ram__ram_hd", "rivian__r1t", "subaru__baja",
        "subaru__brat", "suzuki__equator", "tesla__cybertruck", "toyota__1-ton_truck",
        "toyota__hilux", "toyota__pickup", "toyota__t100", "toyota__tacoma", "toyota__tundra",
    ),
    "Minivan": (
        "chevrolet__astro", "chevrolet__express", "chevrolet__uplander", "chevrolet__venture",
        "chrysler__pacifica", "chrysler__town&country", "dodge__grand_caravan", "fiat__500l",
        "ford__c-max", "ford__e350", "ford__freestar", "ford__transit", "ford__transit_connect",
        "ford__windstar", "gmc__savana", "honda__odyssey", "kia__rondo", "kia__sedona",
        "mazda__5", "mazda__mpv", "mercedes-benz__metris", "mercedes-benz__r-class",
        "mercedes-benz__sprinter", "mercury__villager", "mitsubishi__chariot",
        "mitsubishi__delica", "nissan__nv200", "nissan__presage", "nissan__quest",
        "oldsmobile__silhouette", "opel__vivaro", "plymouth__voyager", "pontiac__montana",
        "ram__promaster", "ram__promaster_city", "saturn__relay", "toyota__estima",
        "toyota__hiace", "toyota__previa", "toyota__sienna", "volkswagen__bus",
        "volkswagen__camper", "volkswagen__eurovan", "volkswagen__routan", "volkswagen__vanagon",
    ),
    "Wagon": (
        "dodge__magnum", "mini__clubman", "peugeot__407_wagon", "saab__9-2", "subaru__outback",
        "volkswagen__squareback", "volvo__145_wagon", "volvo__245", "volvo__v40",
        "volvo__v50", "volvo__v60", "volvo__v70", "volvo__v90", "volvo__xc70",
    ),
    "Convertible": (
        "buick__cascada", "cadillac__xlr", "fiat__124_spider", "honda__delsol", "honda__s2000",
        "mazda__mx-5_miata", "mercedes-benz__sl-class", "mercedes-benz__slk-class",
        "mg__mgb", "mg__midget", "plymouth__prowler", "pontiac__solstice", "porsche__914",
        "porsche__boxster", "rolls-royce__corniche", "saturn__sky", "tesla__roadster",
        "triumph__spitfire", "triumph__tr", "volkswagen__cabrio", "volkswagen__dunebuggy",
        "volkswagen__eos",
    ),
    "Coupe": (
        "acura__cl", "acura__nsx", "acura__rsx", "alfa-romeo__4c", "audi__a5", "audi__r8",
        "audi__tt", "bmw__2_series", "bmw__4_series", "bmw__6_series", "bmw__8_series",
        "bmw__i8", "bmw__m2", "bmw__m4", "bmw__z4", "cadillac__eldorado", "cadillac__elr",
        "chevrolet__camaro", "chevrolet__corvette", "chevrolet__montecarlo", "chrysler__crossfire",
        "dodge__challenger", "dodge__viper", "ford__gt", "ford__mustang", "ford__thunderbird",
        "honda__cr-z", "honda__prelude", "hyundai__tiburon", "infiniti__q60", "jaguar__f-type",
        "jaguar__xk", "lexus__lc", "lexus__rc", "maserati__granturismo", "mazda__rx7",
        "mazda__rx8", "mercedes-benz__amg_gt", "mercedes-benz__cl-class",
        "mercedes-benz__clk-class", "mercedes-benz__slr", "mercury__capri", "mercury__cougar",
        "mercury__cyclone", "mitsubishi__3000gt", "mitsubishi__eclipse", "mitsubishi__starion",
        "nash__metropolitan", "nissan__200sx", "nissan__240sx", "nissan__300zx", "nissan__350z",
        "nissan__370z", "nissan__gtr", "nissan__skyline", "nissan__z", "oldsmobile__442",
        "oldsmobile__starfire", "oldsmobile__toronado", "opel__gt", "plymouth__barracuda",
        "plymouth__duster", "plymouth__gtx", "plymouth__laser", "plymouth__roadrunner",
        "plymouth__scamp", "pontiac__fiero", "pontiac__firebird", "pontiac__gto",
        "pontiac__transam", "porsche__911", "porsche__912", "porsche__917", "porsche__924",
        "porsche__928", "porsche__944", "porsche__cayman", "saturn__sc", "scion__frs",
        "scion__tc", "subaru__brz", "subaru__svx", "sunbeam__rapier", "toyota__86",
        "toyota__celica", "toyota__mr2", "toyota__paseo", "toyota__solara", "toyota__supra",
        "triumph__gt6", "volkswagen__corrado", "volkswagen__karmannghia", "volkswagen__scirocco",
        "volvo__142", "volvo__242", "volvo__780", "volvo__c70",
    ),
    "Hatchback": (
        "acura__integra", "audi__a3", "bmw__1_series", "bmw__i3", "chevrolet__aveo",
        "chevrolet__bolt", "chevrolet__hhr", "chevrolet__sonic", "chevrolet__spark",
        "chevrolet__volt", "chrysler__pt_cruiser", "dodge__caliber", "ford__fiesta",
        "ford__focus", "honda__fit", "honda__insight", "hyundai__ioniq", "hyundai__ioniq_5",
        "hyundai__veloster", "kia__rio", "kia__soul", "lexus__ct", "mazda__2", "mazda__3",
        "mazda__speed3", "mercedes-benz__a-class", "mercedes-benz__b-class", "mini__austin",
        "mini__cooper", "mitsubishi__colt", "mitsubishi__i-miev", "mitsubishi__mirage",
        "nissan__cube", "nissan__leaf", "nissan__pulsar", "opel__astra", "plymouth__colt",
        "plymouth__horizon", "plymouth__sundance", "polestar__polestar_2", "pontiac__g39",
        "pontiac__vibe", "pontiac__wave", "saab__9-6", "saturn__astra", "scion__iq",
        "scion__xa", "scion__xb", "scion__xd", "smart__fortwo", "subaru__360",
        "suzuki__reno", "suzuki__swift", "suzuki__sx4", "toyota__matrix", "toyota__prius",
        "toyota__starlet", "toyota__yaris", "volkswagen__beetle", "volkswagen__golf",
        "volkswagen__gti", "volkswagen__r32", "volvo__c30",
    ),
}

_CANONICAL_BODY_LOOKUP: dict[str, str] = {
    cls_name: body_type for body_type, items in _V2_BODY_GROUPS.items() for cls_name in items
}


def _slugify_part(raw: str) -> str:
    return re.sub(r"\s+", "_", raw.strip().lower())


def _format_part_display(slug: str) -> str:
    tokens = slug.replace("_", " ").split()
    out: list[str] = []
    for tok in tokens:
        if tok in _TOKEN_DISPLAY_OVERRIDES:
            out.append(_TOKEN_DISPLAY_OVERRIDES[tok])
        elif "-" in tok:
            sub = tok.split("-")
            if all(len(s) <= 3 and s.isalnum() for s in sub) and not any(s.isalpha() and len(s) > 2 for s in sub):
                out.append(tok.upper())
            else:
                out.append("-".join(p[:1].upper() + p[1:] if p else "" for p in sub))
        elif tok.isdigit():
            out.append(tok)
        elif tok[:1].isdigit() and any(ch.isalpha() for ch in tok):
            out.append(tok.upper())
        else:
            out.append(tok[:1].upper() + tok[1:])
    return " ".join(out)


def canonicalize_make_model(raw_make: str, raw_model: str) -> tuple[str, str, str, str]:
    """Returns `(canonical_make, canonical_class, make_display, model_display)`."""
    mk = raw_make.strip().lower().replace(" ", "-")
    md = _slugify_part(raw_model)
    if (mk, md) in _MODEL_ALIASES:
        mk, md = _MODEL_ALIASES[(mk, md)]
    canonical_class = f"{mk}__{md}"
    make_display = _MAKE_DISPLAY_OVERRIDES.get(
        mk, "-".join(p[:1].upper() + p[1:] for p in mk.split("-") if p)
    )
    if canonical_class == "polestar__polestar_2":
        model_display = "Polestar 2"
    elif canonical_class == "polestar__polestar_3":
        model_display = "Polestar 3"
    else:
        model_display = _format_part_display(md)
    return mk, canonical_class, make_display, model_display


def resolve_default_body_type(canonical_class: str) -> str:
    return _CANONICAL_BODY_LOOKUP.get(canonical_class, "Sedan")


def refine_body_type_from_text(
    default_body: str,
    commons_category: str = "",
    commons_title: str = "",
) -> str:
    if default_body in ("SUV", "Pickup", "Minivan"):
        return default_body
    text = f"{commons_category} {commons_title}".lower()
    if not text.strip():
        return default_body
    text_clean = (
        text.replace("gran coupe", " ")
        .replace("4-door coupe", " ")
        .replace("four-door coupe", " ")
        .replace("suv coupe", " ")
    )
    if any(k in text_clean for k in ("convertible", "cabrio", "cabriolet", "roadster", "spyder", "spider", "drophead")):
        return "Convertible"
    if any(k in text_clean for k in ("wagon", "touring", "avant", "estate", "sportwagen", "variant", "shooting brake", "alltrack")):
        return "Wagon"
    if any(k in text_clean for k in ("hatchback", "5-door", "3-door", "sportback")):
        return "Hatchback"
    if "coupe" in text_clean or "coupé" in text_clean:
        return "Coupe"
    return default_body


def parse_timdwms_year_interval(
    years_str: str,
    phase_str: str = "",
    date_taken_str: str = "",
) -> tuple[float, float, bool]:
    matches = [int(y) for y in re.findall(r"(?:19|20)\d{2}", years_str or "")]
    if not matches:
        return 0.0, 0.0, False
    lo = float(max(YEAR_MIN, min(YEAR_MAX, min(matches))))
    hi = float(max(YEAR_MIN, min(YEAR_MAX, max(matches))))
    if hi < lo:
        hi = lo
    span = hi - lo
    phase = (phase_str or "").strip().lower()
    if span >= 4.0:
        if phase == "pre":
            hi = round(lo + 0.55 * span, 1)
        elif phase == "facelift":
            lo = round(lo + 0.45 * span, 1)
        elif phase in ("facelift2", "facelift3"):
            lo = round(lo + 0.65 * span, 1)
    if date_taken_str:
        dt_matches = [int(y) for y in re.findall(r"(?:19|20)\d{2}", date_taken_str[:10])]
        if dt_matches:
            dt_year = float(dt_matches[0])
            if lo <= dt_year < hi - 1.0:
                hi = min(hi, dt_year + 1.0)
    if hi < lo:
        hi = lo
    return lo, hi, True


def compute_padded_crop_box(
    car_box_str: str,
    img_w: int,
    img_h: int,
    pad: float = 0.05,
) -> tuple[int, int, int, int] | None:
    if img_w <= 16 or img_h <= 16 or not (car_box_str or "").strip():
        return None
    parts = car_box_str.strip().split()
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    if bw < 24.0 or bh < 24.0:
        return None
    px, py = bw * float(pad), bh * float(pad)
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(int(img_w), int(round(x2 + px)))
    cy2 = min(int(img_h), int(round(y2 + py)))
    if (cx2 - cx1) < 24 or (cy2 - cy1) < 24:
        return None
    return cx1, cy1, cx2, cy2


# --- Model Architecture & Losses ---
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_soft_class_weights(
    counts: Sequence[int] | np.ndarray | torch.Tensor,
    power: float = 0.35,
    min_weight: float = 0.5,
    max_weight: float = 4.0,
) -> torch.Tensor:
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    total = float(c.sum())
    weights = (total / c) ** float(power)
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def year_interval_to_dldl_target(
    y_low: torch.Tensor,
    y_high: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    sigma: float = 1.2,
) -> torch.Tensor:
    bins = torch.arange(year_min, year_max + 1, device=y_low.device, dtype=torch.float32).unsqueeze(0)
    lo = torch.minimum(y_low, y_high).float().clamp(min=float(year_min), max=float(year_max)).unsqueeze(1)
    hi = torch.maximum(y_low, y_high).float().clamp(min=float(year_min), max=float(year_max)).unsqueeze(1)
    dist = torch.clamp(lo - bins, min=0.0) + torch.clamp(bins - hi, min=0.0)
    weights = torch.exp(-0.5 * (dist / float(sigma)).pow(2))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def year_logits_to_expected(
    year_logits: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_f = year_logits.float()
    probs = F.softmax(logits_f, dim=-1)
    bins = torch.arange(year_min, year_max + 1, device=logits_f.device, dtype=torch.float32)
    expected = (probs * bins).sum(dim=-1)
    var = (probs * (bins.unsqueeze(0) - expected.unsqueeze(1)).pow(2)).sum(dim=-1).clamp_min(0.0)
    return expected, torch.sqrt(var)


def dldl_interval_year_loss(
    year_logits: torch.Tensor,
    y_low: torch.Tensor,
    y_high: torch.Tensor,
    valid_mask: torch.Tensor,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    sigma: float = 1.2,
    l1_weight: float = 0.2,
) -> torch.Tensor:
    mask = valid_mask.bool()
    if not torch.any(mask):
        return year_logits.sum() * 0.0
    logits_v = year_logits[mask].float()
    lo_v = y_low[mask]
    hi_v = y_high[mask]
    target_probs = year_interval_to_dldl_target(lo_v, hi_v, year_min=year_min, year_max=year_max, sigma=sigma)
    kl_loss = -(target_probs * F.log_softmax(logits_v, dim=-1)).sum(dim=-1).mean()
    expected_v, _ = year_logits_to_expected(logits_v, year_min=year_min, year_max=year_max)
    midpoint = (0.5 * (lo_v.float() + hi_v.float())).clamp(min=float(year_min), max=float(year_max))
    return kl_loss + float(l1_weight) * F.smooth_l1_loss(expected_v, midpoint, beta=1.0)


class CosFaceHead(nn.Module):
    def __init__(self, in_features: int = 512, out_features: int = 550, scale: float = 32.0, margin: float = 0.35) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = F.normalize(x.float(), dim=-1)
        w_norm = F.normalize(self.weight.float(), dim=-1)
        cosine = F.linear(x_norm, w_norm).clamp(-1.0, 1.0)
        if targets is not None:
            one_hot = F.one_hot(targets.long(), num_classes=self.out_features).to(cosine.dtype)
            cosine = cosine - self.margin * one_hot
        return cosine * self.scale


class SpatialAttentionColorHead(nn.Module):
    def __init__(self, vit_dim: int = 768, num_colors: int = len(COLOR_CLASSES), hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1))
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1, bias=False), nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.attn_gate = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim, 64), nn.GELU(), nn.Linear(64, 1))
        fused_dim = 128 + vit_dim * 2 + 6
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_colors),
        )

    def forward(self, x: torch.Tensor, tok_mid: torch.Tensor, tok_late: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_rgb = (x.float() * self.clip_std + self.clip_mean).clamp(0.0, 1.0)
        conv_feat = self.conv_stem(x_rgb)
        bsz, _, gh, gw = conv_feat.shape
        conv_tokens = conv_feat.flatten(2).transpose(1, 2)
        attn = F.softmax(self.attn_gate(tok_late.float()).squeeze(-1), dim=-1)
        attn_unsq = attn.unsqueeze(-1)
        conv_pooled = (conv_tokens * attn_unsq).sum(dim=1)
        mid_pooled = (tok_mid.float() * attn_unsq).sum(dim=1)
        late_pooled = tok_late.float().mean(dim=1)
        rgb_patches = F.avg_pool2d(x_rgb, 16, 16).flatten(2).transpose(1, 2)
        rgb_mean = (rgb_patches * attn_unsq).sum(dim=1)
        rgb_std = torch.sqrt(((rgb_patches - rgb_mean.unsqueeze(1)).pow(2) * attn_unsq).sum(dim=1).clamp_min(1e-6))
        fused = torch.cat([conv_pooled, mid_pooled, late_pooled, rgb_mean, rgb_std], dim=-1)
        return self.classifier(fused), attn.view(bsz, gh, gw)


def extract_vit_features_and_tokens(
    visual: nn.Module,
    x: torch.Tensor,
    indices: tuple[int, int] = (5, 11),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(visual, "forward_intermediates"):
        try:
            out = visual.forward_intermediates(
                x, indices=list(indices), stop_early=False, normalize_intermediates=False, output_fmt="NLC"
            )
            intermediates = out["image_intermediates"]
            return out["image_features"], intermediates[0], intermediates[1]
        except Exception:
            pass
    captured: dict[int, torch.Tensor] = {}
    handles = []
    bsz = x.shape[0]
    resblocks = visual.transformer.resblocks
    for blk_idx in indices:
        def _make_hook(i: int):
            def _hook(_mod, _inp, output):
                t = output[0] if isinstance(output, tuple) else output
                if t.dim() == 3 and t.shape[0] != bsz and t.shape[1] == bsz:
                    t = t.transpose(0, 1)
                captured[i] = t[:, 1:, :]
            return _hook
        handles.append(resblocks[blk_idx].register_forward_hook(_make_hook(blk_idx)))
    try:
        feat = visual(x)
    finally:
        for h in handles:
            h.remove()
    return feat, captured[indices[0]], captured[indices[1]]


def _set_module_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class AutoVisionMultiTaskNet(nn.Module):
    def __init__(
        self,
        num_makes: int,
        num_models: int,
        num_bodies: int = len(BODY_TYPE_CLASSES),
        num_years: int = NUM_YEAR_BINS,
        num_colors: int = len(COLOR_CLASSES),
        pretrained: str | None = "laion2b_s34b_b88k",
        cosface_scale: float = 32.0,
        cosface_margin: float = 0.35,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        clip_model = open_clip.create_model("ViT-B-16", pretrained=pretrained)
        self.visual = clip_model.visual
        embed_dim = int(getattr(self.visual, "output_dim", 512))
        vit_dim = int(getattr(getattr(self.visual, "transformer", None), "width", 768))

        self.make_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, num_makes))
        self.model_head = CosFaceHead(embed_dim, num_models, scale=cosface_scale, margin=cosface_margin)
        self.body_head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, num_bodies))
        self.year_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(256, num_years)
        )
        self.color_head = SpatialAttentionColorHead(
            vit_dim=vit_dim, num_colors=num_colors, hidden_dim=256, dropout=head_dropout
        )
        self._phase_mode = "full"

    def set_phase1_warmup_mode(self) -> None:
        self._phase_mode = "phase1_warmup"
        _set_module_requires_grad(self.visual, False)
        for head in (self.make_head, self.model_head, self.body_head, self.year_head):
            _set_module_requires_grad(head, True)
        _set_module_requires_grad(self.color_head, False)
        self.train(self.training)

    def set_phase1_finetune_mode(self) -> None:
        self._phase_mode = "phase1_finetune"
        _set_module_requires_grad(self.visual, True)
        for head in (self.make_head, self.model_head, self.body_head, self.year_head):
            _set_module_requires_grad(head, True)
        _set_module_requires_grad(self.color_head, False)
        self.train(self.training)

    def train(self, mode: bool = True) -> "AutoVisionMultiTaskNet":
        super().train(mode)
        if mode:
            if self._phase_mode == "phase1_warmup":
                self.visual.eval()
                self.color_head.eval()
            elif self._phase_mode == "phase1_finetune":
                self.color_head.eval()
        return self

    def build_phase1_param_groups(
        self,
        backbone_lr: float = BACKBONE_LR,
        proj_lr: float = PROJ_LR,
        head_lr: float = HEAD_LR,
        weight_decay: float = WEIGHT_DECAY,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, float, float], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = (
                param.ndim < 2 or name.endswith(".bias")
                or "positional_embedding" in name or "class_embedding" in name
            )
            wd = 0.0 if is_no_decay else float(weight_decay)
            if name.startswith("visual."):
                is_proj = any(k in name for k in ("visual.proj", "visual.ln_post", "visual.attn_pool"))
                lr = float(proj_lr if is_proj else backbone_lr)
                tier = "proj" if is_proj else "backbone"
            else:
                lr = float(head_lr)
                tier = "head"
            buckets.setdefault((tier, lr, wd), []).append(param)
        return [
            {"params": params, "lr": lr, "weight_decay": wd, "group_name": f"{tier}_wd{wd}"}
            for (tier, lr, wd), params in buckets.items()
        ]

    def forward(
        self,
        x: torch.Tensor,
        model_targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw_feat, tok_mid, tok_late = extract_vit_features_and_tokens(self.visual, x, indices=(5, 11))
        emb = F.normalize(raw_feat.float(), dim=-1)
        color_logits, color_attn = self.color_head(x, tok_mid.detach(), tok_late.detach())
        out = {
            "embedding": emb,
            "make_logits": self.make_head(emb),
            "model_logits": self.model_head(emb, targets=model_targets),
            "body_logits": self.body_head(emb),
            "year_logits": self.year_head(emb),
            "color_logits": color_logits,
        }
        if return_attn:
            out["color_attn"] = color_attn
        return out


class MultiTaskV2Loss(nn.Module):
    def __init__(
        self,
        make_weights: torch.Tensor | None = None,
        body_weights: torch.Tensor | None = None,
        w_model: float = 1.0,
        w_make: float = 0.5,
        w_body: float = 0.7,
        w_year: float = 0.7,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_model = float(w_model)
        self.w_make = float(w_make)
        self.w_body = float(w_body)
        self.w_year = float(w_year)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("make_weights", make_weights.float() if make_weights is not None else None)
        self.register_buffer("body_weights", body_weights.float() if body_weights is not None else None)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        make_y: torch.Tensor,
        model_y: torch.Tensor,
        body_y: torch.Tensor,
        year_low: torch.Tensor,
        year_high: torch.Tensor,
        year_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_model = F.cross_entropy(outputs["model_logits"].float(), model_y.long(), label_smoothing=self.label_smoothing)
        l_make = F.cross_entropy(outputs["make_logits"].float(), make_y.long(), weight=self.make_weights, label_smoothing=self.label_smoothing)
        l_body = F.cross_entropy(outputs["body_logits"].float(), body_y.long(), weight=self.body_weights, label_smoothing=self.label_smoothing)
        l_year = dldl_interval_year_loss(
            outputs["year_logits"], year_low, year_high, year_valid, year_min=YEAR_MIN, year_max=YEAR_MAX
        )
        total = self.w_model * l_model + self.w_make * l_make + self.w_body * l_body + self.w_year * l_year
        return total, {
            "total": float(total.detach().item()),
            "model": float(l_model.detach().item()),
            "make": float(l_make.detach().item()),
            "body": float(l_body.detach().item()),
            "year": float(l_year.detach().item()),
        }


# --- Data Acquisition & Preparation ---
def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AutoVision-v2-Trainer/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=1024 * 1024)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
            time.sleep(2.0 * attempt)


def _crop_and_save_jpeg(raw_bytes: bytes, box_str: str, dest_path: Path) -> bool:
    if dest_path.is_file():
        return True
    try:
        from io import BytesIO
        with Image.open(BytesIO(raw_bytes)) as im:
            im = im.convert("RGB")
            w, h = im.size
            crop = compute_padded_crop_box(box_str, w, h, pad=CROP_PAD)
            if crop is not None:
                im = im.crop(crop)
            cw, ch = im.size
            if max(cw, ch) > CROP_MAX_EDGE:
                scale = CROP_MAX_EDGE / float(max(cw, ch))
                im = im.resize((max(16, int(round(cw * scale))), max(16, int(round(ch * scale)))), Image.Resampling.BICUBIC)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest_path, format="JPEG", quality=CROP_JPEG_QUALITY, optimize=True)
        return True
    except Exception:
        return False


def prepare_timdwms_samples(cache_dir: Path) -> list[dict[str, Any]]:
    meta_csv = cache_dir / "timdwms_metadata.csv"
    if not meta_csv.is_file():
        print(f"[timdwms] Downloading metadata.csv -> {meta_csv}")
        _download_file(f"{HF_DATASET_BASE}/metadata.csv", meta_csv)

    by_zip: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("filter") != "ok" or row.get("zip") == "_Multiple cars.zip":
                continue
            if not row.get("make", "").strip() or not row.get("model", "").strip():
                continue
            by_zip[row["zip"]].append(row)

    img_root = cache_dir / "timdwms_crops"
    samples: list[dict[str, Any]] = []
    zip_names = sorted(by_zip.keys())
    print(f"[timdwms] Preparing {sum(len(v) for v in by_zip.values()):,} clean photos across {len(zip_names)} make archives...")

    for idx, zip_name in enumerate(zip_names, start=1):
        rows = by_zip[zip_name]
        # Skip download if every crop in this brand zip already exists on disk.
        pending = [r for r in rows if not (img_root / r["file_name"]).with_suffix(".jpg").is_file()]
        if pending:
            tmp_zip = cache_dir / "_active_brand.zip"
            _download_file(f"{HF_DATASET_BASE}/images/{quote(zip_name)}", tmp_zip)
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                def _process_one(r: dict[str, str]) -> bool:
                    out_p = (img_root / r["file_name"]).with_suffix(".jpg")
                    try:
                        raw = zf.read(r["file_name"])
                    except KeyError:
                        return False
                    return _crop_and_save_jpeg(raw, r.get("car_box", ""), out_p)

                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(_process_one, pending))
            tmp_zip.unlink(missing_ok=True)
            if idx % 10 == 0 or idx == len(zip_names):
                print(f"  [{idx}/{len(zip_names)}] {zip_name} complete ({len(rows):,} rows)")

        for r in rows:
            crop_path = (img_root / r["file_name"]).with_suffix(".jpg")
            if not crop_path.is_file():
                continue
            mk, cls_name, _, _ = canonicalize_make_model(r["make"], r["model"])
            y_lo, y_hi, y_ok = parse_timdwms_year_interval(
                r.get("years", ""), r.get("phase", ""), r.get("date_taken", "")
            )
            body = refine_body_type_from_text(
                resolve_default_body_type(cls_name),
                r.get("commons_category", ""),
                r.get("commons_title", ""),
            )
            samples.append({
                "source": "timdwms",
                "rel_path": f"timdwms/{r['file_name']}",
                "abs_path": str(crop_path),
                "make": mk,
                "canonical_class": cls_name,
                "body_type": body,
                "year_low": y_lo,
                "year_high": y_hi,
                "year_valid": y_ok,
                "hint_split": r.get("used_in_run6", "").strip().lower(),
            })

    return samples


def _bounded_walk_files(search_root: Path, filename: str, max_depth: int = 5) -> list[Path]:
    matches: list[Path] = []
    if not search_root.is_dir():
        return matches
    for root, dirs, files in os.walk(search_root):
        rel_parts = Path(root).relative_to(search_root).parts
        if len(rel_parts) > max_depth:
            dirs[:] = []
            continue
        if filename in files:
            matches.append(Path(root) / filename)
    return sorted(matches)


def discover_and_load_vmmrdb(
    cache_dir: Path,
    input_root: Path = Path("/kaggle/input"),
) -> list[dict[str, Any]]:
    if not input_root.is_dir():
        return []

    mt_cands = _bounded_walk_files(input_root, "local_vmmrdb_multitask_manifest.csv", max_depth=5)
    base_cands = _bounded_walk_files(input_root, "local_vmmrdb_manifest.csv", max_depth=5)
    manifest_cands = mt_cands or base_cands
    if not manifest_cands:
        print("[VMMRdb] Optional clean-89k manifest not attached; proceeding with timdwms only.")
        return []
    manifest_path = manifest_cands[0]

    bbox_cands = _bounded_walk_files(input_root, "bboxes_yolo.npz", max_depth=5)
    bboxes_path = bbox_cands[0] if bbox_cands else None

    bbox_map: dict[str, tuple[float, float, float, float]] = {}
    if bboxes_path is not None and bboxes_path.is_file():
        with np.load(bboxes_path, allow_pickle=False) as d:
            for rp, box, used in zip(d["rel_paths"], d["bboxes"], d["used_yolo"]):
                if bool(used) and len(box) >= 4:
                    bbox_map[str(rp).replace("\\", "/")] = (
                        float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    )

    parsed_rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_clean", "true")).strip().lower() != "true":
                continue
            orig_rel = row["rel_path"].replace("\\", "/")
            img_rel = (row.get("kaggle_rel_path") or orig_rel).replace("\\", "/")
            mk_raw = row.get("make", "").strip()
            md_raw = row.get("model", "").strip()
            if not (mk_raw and md_raw) and "__" in row.get("canonical_class", ""):
                mk_raw, md_raw = row["canonical_class"].split("__", 1)
            mk, cls_name, _, _ = canonicalize_make_model(mk_raw, md_raw)
            yr_str = row.get("raw_year", "").strip()
            yr_valid = str(row.get("year_valid", "")).strip().lower() == "true" and yr_str.isdigit()
            yr_val = float(max(YEAR_MIN, min(YEAR_MAX, int(yr_str)))) if yr_valid else 0.0
            body = row.get("body_type", "").strip()
            if body not in BODY_TYPE_CLASSES:
                body = resolve_default_body_type(cls_name)
            parsed_rows.append({
                "orig_rel": orig_rel,
                "img_rel": img_rel,
                "mk": mk,
                "cls": cls_name,
                "body": body,
                "yr": yr_val,
                "yr_ok": yr_valid,
                "split": row.get("split", "").strip(),
            })

    if not parsed_rows:
        return []

    sample_rels = [
        parsed_rows[0]["img_rel"],
        parsed_rows[len(parsed_rows) // 2]["img_rel"],
        parsed_rows[-1]["img_rel"],
    ]
    dir_cands = [manifest_path.parent, manifest_path.parent / "images"]
    for root, dirs, _ in os.walk(input_root):
        if len(Path(root).relative_to(input_root).parts) > 4:
            dirs[:] = []
            continue
        dir_cands.append(Path(root))

    images_dir: Path | None = None
    seen_dirs: set[Path] = set()
    for cand in dir_cands:
        if cand in seen_dirs or not cand.is_dir():
            continue
        seen_dirs.add(cand)
        if all((cand / rp).is_file() for rp in sample_rels):
            images_dir = cand
            break

    if images_dir is None:
        print(f"[VMMRdb] Manifest found at {manifest_path}, but image root did not match sample files; skipping VMMRdb.")
        return []

    raw_rows = [r for r in parsed_rows if (images_dir / r["img_rel"]).is_file()]
    out_root = cache_dir / "vmmrdb_crops"

    def _prep_vmmr(item: dict[str, Any]) -> dict[str, Any] | None:
        dest = (out_root / item["orig_rel"]).with_suffix(".jpg")
        src_img = images_dir / item["img_rel"]
        if not dest.is_file():
            try:
                with Image.open(src_img) as im:
                    im = im.convert("RGB")
                    box = bbox_map.get(item["orig_rel"])
                    if box is not None:
                        iw, ih = im.size
                        x1 = max(0, min(iw - 1, int(round(box[0]))))
                        y1 = max(0, min(ih - 1, int(round(box[1]))))
                        x2 = max(x1 + 1, min(iw, int(round(box[2]))))
                        y2 = max(y1 + 1, min(ih, int(round(box[3]))))
                        if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                            im = im.crop((x1, y1, x2, y2))
                    cw, ch = im.size
                    if max(cw, ch) > CROP_MAX_EDGE:
                        scale = CROP_MAX_EDGE / float(max(cw, ch))
                        im = im.resize((max(16, int(round(cw * scale))), max(16, int(round(ch * scale)))), Image.Resampling.BICUBIC)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    im.save(dest, format="JPEG", quality=CROP_JPEG_QUALITY, optimize=True)
            except Exception:
                return None
        return {
            "source": "vmmrdb",
            "rel_path": f"vmmrdb/{item['orig_rel']}",
            "abs_path": str(dest),
            "make": item["mk"],
            "canonical_class": item["cls"],
            "body_type": item["body"],
            "year_low": item["yr"],
            "year_high": item["yr"],
            "year_valid": item["yr_ok"],
            "hint_split": item["split"],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_prep_vmmr, raw_rows))
    valid_samples = [r for r in results if r is not None]
    print(f"[VMMRdb] Prepared {len(valid_samples):,} clean street-photo samples ({len(bbox_map):,} YOLO crops aligned).")
    return valid_samples


def build_splits_and_balance(samples: list[dict[str, Any]], seed: int = SEED) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_class[s["canonical_class"]].append(s)

    closed_classes = sorted(c for c, items in by_class.items() if len(items) >= MIN_SAMPLES_CLOSED)
    few_shot_classes = sorted(c for c, items in by_class.items() if 2 <= len(items) < MIN_SAMPLES_CLOSED)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": [], "few_shot": []}

    for cls_name in closed_classes:
        items = list(by_class[cls_name])
        rng.shuffle(items)
        n_total = len(items)
        n_val_target = max(1, min(int(round(0.08 * n_total)), max(1, n_total // 8)))
        n_test_target = max(1, min(int(round(0.08 * n_total)), max(1, n_total // 8)))

        # Respect non-overlapping upstream run6 test/val hints where reasonable.
        val_bucket = [s for s in items if s["hint_split"] == "val"][:n_val_target]
        used_ids = {id(s) for s in val_bucket}
        test_bucket = [s for s in items if id(s) not in used_ids and s["hint_split"] == "test"][:n_test_target]
        used_ids.update(id(s) for s in test_bucket)

        rem = [s for s in items if id(s) not in used_ids]
        while len(val_bucket) < n_val_target and rem:
            val_bucket.append(rem.pop())
        while len(test_bucket) < n_test_target and rem:
            test_bucket.append(rem.pop())

        # Prioritize 2017+ and VMMRdb street photos when subsampling mega-classes.
        if len(rem) > MAX_TRAIN_PER_MODEL:
            modern = [s for s in rem if s["year_valid"] and s["year_high"] >= 2017]
            street = [s for s in rem if s["source"] == "vmmrdb" and not (s["year_valid"] and s["year_high"] >= 2017)]
            other = [s for s in rem if id(s) not in {id(x) for x in modern} and id(s) not in {id(x) for x in street}]
            selected: list[dict[str, Any]] = []
            selected.extend(modern[: int(0.55 * MAX_TRAIN_PER_MODEL)])
            selected.extend(street[: max(0, MAX_TRAIN_PER_MODEL - len(selected)) // 2])
            pool = other + modern[int(0.55 * MAX_TRAIN_PER_MODEL):] + street
            rng.shuffle(pool)
            seen = {id(x) for x in selected}
            for x in pool:
                if len(selected) >= MAX_TRAIN_PER_MODEL:
                    break
                if id(x) not in seen:
                    selected.append(x)
                    seen.add(id(x))
            train_bucket = selected
        else:
            train_bucket = rem

        for s in train_bucket:
            s["split"] = "train"
        for s in val_bucket:
            s["split"] = "val"
        for s in test_bucket:
            s["split"] = "test"
        splits["train"].extend(train_bucket)
        splits["val"].extend(val_bucket)
        splits["test"].extend(test_bucket)

    for cls_name in few_shot_classes:
        items = list(by_class[cls_name])
        for s in items:
            s["split"] = "few_shot"
        splits["few_shot"].extend(items)

    return splits, closed_classes, few_shot_classes


# --- PyTorch Dataset & Transforms ---
class AutoVisionV2Dataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        make_to_id: dict[str, int],
        model_to_id: dict[str, int],
        body_to_id: dict[str, int],
        transform: T.Compose,
    ) -> None:
        self.samples = samples
        self.make_to_id = make_to_id
        self.model_to_id = model_to_id
        self.body_to_id = body_to_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int, float, float, int]:
        s = self.samples[idx]
        with Image.open(s["abs_path"]) as im:
            img_t = self.transform(im.convert("RGB"))
        mk_y = self.make_to_id.get(s["make"], -1)
        md_y = self.model_to_id.get(s["canonical_class"], -1)
        bd_y = self.body_to_id.get(s["body_type"], 0)
        return (
            img_t, mk_y, md_y, bd_y,
            float(s["year_low"]), float(s["year_high"]), 1 if s["year_valid"] else 0,
        )


def build_train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(224, scale=(0.68, 1.0), ratio=(0.85, 1.18), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05)], p=0.65),
        T.RandomAffine(degrees=10, translate=(0.04, 0.04)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])


def build_eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# --- Evaluation & Gallery Builders ---
def _safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _topk_correct(logits: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(int(k), logits.shape[1])
    topk = np.argpartition(logits, -k_eff, axis=1)[:, -k_eff:]
    return np.any(topk == targets[:, None], axis=1)


@torch.no_grad()
def evaluate_v2_split(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_tta: bool = True,
) -> dict[str, Any]:
    model.eval()
    use_cuda = device.startswith("cuda")
    embs, mk_l, md_l, bd_l, yr_l = [], [], [], [], []
    mk_y, md_y, bd_y, y_lo, y_hi, y_val = [], [], [], [], [], []

    for x, mk, md, bd, lo, hi, val in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x)
            emb = out["embedding"].float()
            l_mk = out["make_logits"].float()
            l_md = out["model_logits"].float()
            l_bd = out["body_logits"].float()
            l_yr = out["year_logits"].float()
            if use_tta:
                out_f = model(torch.flip(x, dims=[-1]))
                emb = F.normalize(emb + out_f["embedding"].float(), dim=-1)
                l_mk = 0.5 * (l_mk + out_f["make_logits"].float())
                l_md = 0.5 * (l_md + out_f["model_logits"].float())
                l_bd = 0.5 * (l_bd + out_f["body_logits"].float())
                l_yr = 0.5 * (l_yr + out_f["year_logits"].float())

        embs.append(emb.cpu().numpy())
        mk_l.append(l_mk.cpu().numpy())
        md_l.append(l_md.cpu().numpy())
        bd_l.append(l_bd.cpu().numpy())
        yr_l.append(l_yr.cpu().numpy())
        mk_y.append(mk.numpy())
        md_y.append(md.numpy())
        bd_y.append(bd.numpy())
        y_lo.append(lo.numpy())
        y_hi.append(hi.numpy())
        y_val.append(val.numpy())

    arr_emb = np.concatenate(embs, axis=0)
    arr_mk_l = np.concatenate(mk_l, axis=0)
    arr_md_l = np.concatenate(md_l, axis=0)
    arr_bd_l = np.concatenate(bd_l, axis=0)
    arr_yr_l = np.concatenate(yr_l, axis=0)

    arr_mk_y = np.concatenate(mk_y, axis=0)
    arr_md_y = np.concatenate(md_y, axis=0)
    arr_bd_y = np.concatenate(bd_y, axis=0)
    arr_lo = np.concatenate(y_lo, axis=0)
    arr_hi = np.concatenate(y_hi, axis=0)
    arr_val = np.concatenate(y_val, axis=0).astype(bool)

    pred_mk = np.argmax(arr_mk_l, axis=1)
    pred_md = np.argmax(arr_md_l, axis=1)
    pred_bd = np.argmax(arr_bd_l, axis=1)
    exp_yr, _ = year_logits_to_expected(torch.from_numpy(arr_yr_l), YEAR_MIN, YEAR_MAX)
    pred_yr = exp_yr.numpy()

    mid_yr = 0.5 * (arr_lo + arr_hi)
    if np.any(arr_val):
        # Zero error if prediction falls inside the true generation interval [lo, hi].
        interval_err = np.maximum(0.0, np.maximum(arr_lo[arr_val] - pred_yr[arr_val], pred_yr[arr_val] - arr_hi[arr_val]))
        mid_err = np.abs(pred_yr[arr_val] - mid_yr[arr_val])
        year_mae_mid = float(np.mean(mid_err))
        year_within2 = float(np.mean(interval_err <= 2.0))
    else:
        year_mae_mid = 0.0
        year_within2 = 0.0

    make_top1 = float(np.mean(pred_mk == arr_mk_y))
    model_top1 = float(np.mean(pred_md == arr_md_y))
    body_top1 = float(np.mean(pred_bd == arr_bd_y))
    val_score = 0.45 * model_top1 + 0.20 * make_top1 + 0.20 * body_top1 + 0.15 * year_within2

    return {
        "n_eval": int(len(arr_md_y)),
        "val_score": round(val_score, 4),
        "make_top1": round(make_top1, 4),
        "make_top5": round(float(np.mean(_topk_correct(arr_mk_l, arr_mk_y, 5))), 4),
        "make_macro_f1": round(_safe_macro_f1(arr_mk_y, pred_mk), 4),
        "model_top1": round(model_top1, 4),
        "model_top5": round(float(np.mean(_topk_correct(arr_md_l, arr_md_y, 5))), 4),
        "model_macro_f1": round(_safe_macro_f1(arr_md_y, pred_md), 4),
        "body_top1": round(body_top1, 4),
        "body_top3": round(float(np.mean(_topk_correct(arr_bd_l, arr_bd_y, 3))), 4),
        "body_macro_f1": round(_safe_macro_f1(arr_bd_y, pred_bd), 4),
        "year_mae_mid": round(year_mae_mid, 4),
        "year_within2": round(year_within2, 4),
        "_raw": {
            "embeddings": arr_emb,
            "pred_make": pred_mk, "true_make": arr_mk_y,
            "pred_model": pred_md, "true_model": arr_md_y,
            "pred_body": pred_bd, "true_body": arr_bd_y,
            "pred_year": pred_yr, "year_low": arr_lo, "year_high": arr_hi, "year_valid": arr_val,
        },
    }


def _metrics_subset(raw: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"n_eval": 0}
    val_mask = mask & raw["year_valid"]
    mid = 0.5 * (raw["year_low"][val_mask] + raw["year_high"][val_mask])
    err = np.maximum(0.0, np.maximum(raw["year_low"][val_mask] - raw["pred_year"][val_mask], raw["pred_year"][val_mask] - raw["year_high"][val_mask]))
    return {
        "n_eval": int(np.sum(mask)),
        "make_top1": round(float(np.mean(raw["pred_make"][mask] == raw["true_make"][mask])), 4),
        "model_top1": round(float(np.mean(raw["pred_model"][mask] == raw["true_model"][mask])), 4),
        "body_top1": round(float(np.mean(raw["pred_body"][mask] == raw["true_body"][mask])), 4),
        "year_mae_mid": round(float(np.mean(np.abs(raw["pred_year"][val_mask] - mid))), 4) if np.any(val_mask) else 0.0,
        "year_within2": round(float(np.mean(err <= 2.0)), 4) if np.any(val_mask) else 0.0,
    }


@torch.no_grad()
def extract_embeddings(model: nn.Module, samples: list[dict[str, Any]], make_to_id: dict[str, int], model_to_id: dict[str, int], body_to_id: dict[str, int], device: str) -> np.ndarray:
    if not samples:
        return np.zeros((0, 512), dtype=np.float32)
    ds = AutoVisionV2Dataset(samples, make_to_id, model_to_id, body_to_id, build_eval_transform())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.startswith("cuda"))
    model.eval()
    use_cuda = device.startswith("cuda")
    chunks: list[np.ndarray] = []
    try:
        for x, *_ in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                emb = model(x)["embedding"].float()
            chunks.append(emb.cpu().numpy().astype(np.float32))
    finally:
        del loader
    return np.concatenate(chunks, axis=0)


def build_curated_gallery(splits: dict[str, list[dict[str, Any]]], seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = splits["train"] + splits["few_shot"]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in pool:
        by_class[s["canonical_class"]].append(s)

    selected: list[dict[str, Any]] = []
    for cls_name in sorted(by_class):
        items = by_class[cls_name]
        modern = [s for s in items if s["year_valid"] and s["year_high"] >= 2017]
        older = [s for s in items if not (s["year_valid"] and s["year_high"] >= 2017)]
        rng.shuffle(modern)
        rng.shuffle(older)
        pick = (modern + older)[:GALLERY_PER_MODEL]
        selected.extend(pick)

    if len(selected) > MAX_GALLERY_ITEMS:
        rng.shuffle(selected)
        selected = selected[:MAX_GALLERY_ITEMS]
    return selected


def copy_v1_color_head_if_present(
    target_net: AutoVisionMultiTaskNet,
    input_root: Path = Path("/kaggle/input"),
) -> bool:
    if not input_root.is_dir():
        return False
    for p in _bounded_walk_files(input_root, "autovision_multitask_best.pt", max_depth=5):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            color_state = {k[len("color_head."):]: v for k, v in state.items() if k.startswith("color_head.")}
            if color_state:
                target_net.color_head.load_state_dict(color_state, strict=True)
                print(f"[Color] Carried over interim v1 color_head weights from {p}")
                return True
        except Exception as exc:
            print(f"[Color] Skipped copying v1 color_head ({exc})")
    return False


def main() -> None:
    t_start = time.perf_counter()
    set_seed(SEED)
    WORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda" if n_gpus > 0 else "cpu"
    batch_size = 128 if n_gpus >= 2 else 64
    grad_accum = max(1, EFFECTIVE_BATCH_SIZE // batch_size)
    print(f"[Env] Device={device} | GPUs={n_gpus} | Batch={batch_size} x Accum={grad_accum}")

    # 1. Gather datasets
    tim_samples = prepare_timdwms_samples(WORK_CACHE_DIR)
    vmmr_samples = discover_and_load_vmmrdb(WORK_CACHE_DIR)
    all_samples = tim_samples + vmmr_samples
    if len(all_samples) < 1000:
        raise RuntimeError(f"Insufficient prepared images ({len(all_samples)}).")

    splits, closed_models, few_shot_models = build_splits_and_balance(all_samples, SEED)
    makes = sorted({s["make"] for s in splits["train"] + splits["val"] + splits["test"]})
    make_to_id = {m: i for i, m in enumerate(makes)}
    model_to_id = {m: i for i, m in enumerate(closed_models)}
    body_classes = list(BODY_TYPE_CLASSES)
    body_to_id = {b: i for i, b in enumerate(body_classes)}

    print(
        f"[Dataset] Total={len(all_samples):,} | ClosedModels={len(closed_models)} | FewShot={len(few_shot_models)} | "
        f"Makes={len(makes)} | Split(tr/va/te/fs)={len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,}/{len(splits['few_shot']):,}"
    )

    # 2. DataLoaders
    tr_ds = AutoVisionV2Dataset(splits["train"], make_to_id, model_to_id, body_to_id, build_train_transform())
    va_ds = AutoVisionV2Dataset(splits["val"], make_to_id, model_to_id, body_to_id, build_eval_transform())
    te_ds = AutoVisionV2Dataset(splits["test"], make_to_id, model_to_id, body_to_id, build_eval_transform())

    pin = device == "cuda"
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    # 3. Fresh AutoVisionMultiTaskNet from OpenCLIP ViT-B/16
    base_net = AutoVisionMultiTaskNet(
        num_makes=len(makes),
        num_models=len(closed_models),
        num_bodies=len(body_classes),
        num_years=NUM_YEAR_BINS,
        num_colors=len(COLOR_CLASSES),
        pretrained="laion2b_s34b_b88k",
        cosface_scale=32.0,
        cosface_margin=0.35,
    ).to(device)

    mk_cnt = Counter(make_to_id[s["make"]] for s in splits["train"])
    bd_cnt = Counter(body_to_id[s["body_type"]] for s in splits["train"])
    mk_weights = compute_soft_class_weights([mk_cnt[i] for i in range(len(makes))], power=0.35)
    bd_weights = compute_soft_class_weights([bd_cnt[i] for i in range(len(body_classes))], power=0.35)
    criterion = MultiTaskV2Loss(make_weights=mk_weights, body_weights=bd_weights).to(device)

    run_net: nn.Module = nn.DataParallel(base_net) if n_gpus >= 2 else base_net
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    history: list[dict[str, Any]] = []
    best_val_score = -1.0
    best_ckpt_path = OUT_DIR / "autovision_multitask_best.pt"
    last_ckpt_path = OUT_DIR / "last_checkpoint.pt"

    def _run_epochs(epochs: int, phase_name: str, optimizer: torch.optim.Optimizer, scheduler: Any) -> None:
        nonlocal best_val_score
        for ep in range(1, epochs + 1):
            base_net.train(True)
            loss_sum, steps = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(enumerate(tr_loader, start=1), total=len(tr_loader), desc=f"{phase_name} Ep {ep}/{epochs}")
            for step_i, (x, mk, md, bd, lo, hi, val) in pbar:
                x = x.to(device, non_blocking=True)
                mk_t = mk.to(device, non_blocking=True)
                md_t = md.to(device, non_blocking=True)
                bd_t = bd.to(device, non_blocking=True)
                lo_t = lo.to(device, non_blocking=True)
                hi_t = hi.to(device, non_blocking=True)
                val_t = val.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=pin):
                    out = run_net(x, md_t)
                    loss, _ = criterion(out, mk_t, md_t, bd_t, lo_t, hi_t, val_t)
                    scaled_loss = loss / float(grad_accum)
                scaler.scale(scaled_loss).backward()

                if step_i % grad_accum == 0 or step_i == len(tr_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_net.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                loss_sum += float(loss.item())
                steps += 1
                if steps % 50 == 0:
                    pbar.set_postfix(loss=f"{loss_sum / steps:.3f}")

            if scheduler is not None:
                scheduler.step()

            val_metrics = evaluate_v2_split(run_net, va_loader, device, use_tta=True)
            val_metrics_clean = {k: v for k, v in val_metrics.items() if k != "_raw"}
            row = {"phase": phase_name, "epoch": ep, "train_loss": round(loss_sum / max(steps, 1), 4), **val_metrics_clean}
            history.append(row)
            print(
                f"  [{phase_name} Ep {ep}] loss={row['train_loss']:.4f} | val_score={row['val_score']*100:.2f}% | "
                f"model={row['model_top1']*100:.2f}% | make={row['make_top1']*100:.2f}% | "
                f"body={row['body_top1']*100:.2f}% | yr_in2={row['year_within2']*100:.2f}%"
            )

            ckpt_payload = {
                "model_state_dict": base_net.state_dict(),
                "model_version": "autovision-v2-vitb16-multitask",
                "val_score": row["val_score"],
            }
            torch.save(ckpt_payload, last_ckpt_path)
            if row["val_score"] > best_val_score:
                best_val_score = row["val_score"]
                torch.save(ckpt_payload, best_ckpt_path)
                print(f"  -> Saved new best checkpoint ({best_val_score*100:.2f}%)")

    # Stage A: Warmup heads (frozen backbone)
    base_net.set_phase1_warmup_mode()
    opt_warmup = torch.optim.AdamW([p for p in base_net.parameters() if p.requires_grad], lr=WARMUP_HEAD_LR, weight_decay=WEIGHT_DECAY)
    _run_epochs(PHASE1_WARMUP_EPOCHS, "Warmup", opt_warmup, scheduler=None)

    # Stage B: Full end-to-end ViT fine-tuning
    base_net.set_phase1_finetune_mode()
    opt_ft = torch.optim.AdamW(base_net.build_phase1_param_groups())
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=PHASE1_EPOCHS, eta_min=1e-6)
    _run_epochs(PHASE1_EPOCHS, "FineTune", opt_ft, scheduler=sched_ft)

    del tr_loader, va_loader
    gc.collect()

    # 4. Reload best checkpoint & optionally attach interim v1 color_head
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    base_net.load_state_dict(best_state, strict=True)
    if copy_v1_color_head_if_present(base_net):
        torch.save({"model_state_dict": base_net.state_dict(), "model_version": "autovision-v2-vitb16-multitask", "val_score": best_val_score}, best_ckpt_path)

    eval_net: nn.Module = nn.DataParallel(base_net) if n_gpus >= 2 else base_net

    # 5. Final Holdout Test Evaluation (+ Modern 2017-2026 & Street subsets)
    test_eval = evaluate_v2_split(eval_net, te_loader, device, use_tta=True)
    raw = test_eval.pop("_raw")
    is_modern = np.array([bool(s["year_valid"] and s["year_high"] >= 2017) for s in splits["test"]], dtype=bool)
    is_vmmr = np.array([s["source"] == "vmmrdb" for s in splits["test"]], dtype=bool)
    modern_metrics = _metrics_subset(raw, is_modern)
    vmmr_metrics = _metrics_subset(raw, is_vmmr)
    del te_loader

    print(f"\n[Test Overall] {test_eval}")
    print(f"[Test 2017-2026] {modern_metrics}")
    if is_vmmr.any():
        print(f"[Test VMMRdb Street] {vmmr_metrics}")

    # 6. Prototypes (all closed + few-shot classes) & Curated Gallery Export
    print("\n[Export] Extracting embeddings for Prototypes & Curated Gallery...")
    gallery_samples = build_curated_gallery(splits, seed=SEED)
    gallery_embs = extract_embeddings(eval_net, gallery_samples, make_to_id, model_to_id, body_to_id, device)

    by_proto_cls: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(gallery_samples):
        by_proto_cls[s["canonical_class"]].append(i)

    # Ensure every closed class also has prototype coverage even if truncated by MAX_GALLERY_ITEMS.
    missing_closed = [c for c in closed_models if c not in by_proto_cls]
    if missing_closed:
        extra = [s for s in splits["train"] if s["canonical_class"] in set(missing_closed)]
        extra_embs = extract_embeddings(eval_net, extra, make_to_id, model_to_id, body_to_id, device)
        base_idx = len(gallery_samples)
        gallery_samples.extend(extra)
        gallery_embs = np.vstack([gallery_embs, extra_embs])
        for j, s in enumerate(extra):
            by_proto_cls[s["canonical_class"]].append(base_idx + j)

    proto_vecs, proto_names, proto_makes, proto_bodies, proto_split_types = [], [], [], [], []
    few_set = set(few_shot_models)
    for cls_name in sorted(by_proto_cls):
        idxs = by_proto_cls[cls_name]
        vec = gallery_embs[idxs].mean(axis=0)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        sample0 = gallery_samples[idxs[0]]
        proto_vecs.append(vec.astype(np.float32))
        proto_names.append(cls_name)
        proto_makes.append(sample0["make"])
        proto_bodies.append(sample0["body_type"])
        proto_split_types.append("few_shot" if cls_name in few_set else "train")

    proto_mat = np.stack(proto_vecs, axis=0)
    # Score test set against nearest prototype
    sims_te = raw["embeddings"] @ proto_mat.T
    cls_to_proto = {n: i for i, n in enumerate(proto_names)}
    te_proto_preds = np.argmax(sims_te, axis=1)
    te_true_cls = np.array([cls_to_proto[closed_models[i]] for i in raw["true_model"]])
    test_eval["nearest_prototype_top1"] = round(float(np.mean(te_proto_preds == te_true_cls)), 4)

    # Save prototypes_v2.npz
    np.savez_compressed(
        OUT_DIR / "prototypes_v2.npz",
        prototypes=proto_mat,
        class_names=np.array(proto_names),
        makes=np.array(proto_makes),
        body_types=np.array(proto_bodies),
        split_types=np.array(proto_split_types),
        is_few_shot=np.array([st == "few_shot" for st in proto_split_types], dtype=bool),
    )

    # Save gallery vectors + manifest + zip thumbnails
    np.savez_compressed(
        OUT_DIR / "features_finetuned_openclip_vitb16.npz",
        embeddings=gallery_embs.astype(np.float32),
        rel_paths=np.array([s["rel_path"] for s in gallery_samples]),
    )
    with open(OUT_DIR / "autovision_v2_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rel_path", "canonical_class", "make", "raw_year", "year_valid", "body_type", "split", "is_clean"])
        w.writeheader()
        for s in gallery_samples:
            yr = int(round(0.5 * (s["year_low"] + s["year_high"]))) if s["year_valid"] else ""
            w.writerow({
                "rel_path": s["rel_path"],
                "canonical_class": s["canonical_class"],
                "make": s["make"],
                "raw_year": yr,
                "year_valid": "true" if s["year_valid"] else "false",
                "body_type": s["body_type"],
                "split": s["split"],
                "is_clean": "true",
            })

    with zipfile.ZipFile(OUT_DIR / "gallery_v2_thumbs.zip", "w", compression=zipfile.ZIP_STORED) as zf:
        for s in gallery_samples:
            zf.write(s["abs_path"], arcname=s["rel_path"])

    # Display name maps for backend schemas
    make_display_names: dict[str, str] = {}
    model_display_names: dict[str, str] = {}
    for cls_name in sorted(set(closed_models) | set(few_shot_models)):
        mk_part, md_part = cls_name.split("__", 1)
        _, _, mk_d, md_d = canonicalize_make_model(mk_part, md_part)
        make_display_names[mk_part] = mk_d
        model_display_names[cls_name] = md_d

    label_maps = {
        "model_version": "autovision-v2-vitb16-multitask",
        "make_classes": makes,
        "model_classes": closed_models,
        "body_classes": body_classes,
        "color_classes": list(COLOR_CLASSES),
        "color_groups_11": COLOR_GROUPS_11,
        "color_hex": COLOR_HEX,
        "year_min": YEAR_MIN,
        "year_max": YEAR_MAX,
        "make_display_names": make_display_names,
        "model_display_names": model_display_names,
    }
    (OUT_DIR / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    # History & summary
    if history:
        with open(OUT_DIR / "training_history.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    np.savez_compressed(
        OUT_DIR / "v2_test_predictions.npz",
        pred_make=raw["pred_make"], true_make=raw["true_make"],
        pred_model=raw["pred_model"], true_model=raw["true_model"],
        pred_body=raw["pred_body"], true_body=raw["true_body"],
        pred_year=raw["pred_year"], year_low=raw["year_low"],
        year_high=raw["year_high"], year_valid=raw["year_valid"],
    )

    summary = {
        "model_version": "autovision-v2-vitb16-multitask",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "num_total_images": len(all_samples),
        "num_timdwms_images": len(tim_samples),
        "num_vmmrdb_images": len(vmmr_samples),
        "num_makes": len(makes),
        "num_models_closed": len(closed_models),
        "num_prototype_classes": len(proto_names),
        "gallery_size": len(gallery_samples),
        "best_val_score": best_val_score,
        "overall_test_metrics": test_eval,
        "modern_2017_2026_test_metrics": modern_metrics,
        "vmmrdb_street_test_metrics": vmmr_metrics,
    }
    (OUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== AUTOVISION v2 PHASE 1 COMPLETE -> {OUT_DIR} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
