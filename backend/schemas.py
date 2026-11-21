"""Typed Pydantic v2 domain and HTTP response contracts for AutoVision API."""

from __future__ import annotations

from pydantic import BaseModel, Field

_SPECIAL_DISPLAY_TOKENS: dict[str, str] = {
    "bmw": "BMW",
    "brz": "BRZ",
    "cj2a": "CJ-2A",
    "cj3b": "CJ-3B",
    "cla": "CLA",
    "clk": "CLK",
    "cls": "CLS",
    "cr-v": "CR-V",
    "crv": "CR-V",
    "fj40": "FJ40",
    "fjcruiser": "FJ Cruiser",
    "frs": "FR-S",
    "gl": "GL",
    "gla": "GLA",
    "glk": "GLK",
    "gt": "GT",
    "gt6": "GT6",
    "gti": "GTI",
    "gto": "GTO",
    "gtr": "GT-R",
    "gtx": "GTX",
    "iq": "iQ",
    "mg": "MG",
    "mgb": "MGB",
    "mr2": "MR2",
    "nv200": "NV200",
    "pv444": "PV444",
    "r32": "R32",
    "rx7": "RX-7",
    "rx8": "RX-8",
    "sl": "SL",
    "slk": "SLK",
    "slr": "SLR",
    "suv": "SUV",
    "svx": "SVX",
    "sx4": "SX4",
    "tc": "tC",
    "tr": "TR",
    "wrx": "WRX",
    "xa": "xA",
    "xb": "xB",
    "xc60": "XC60",
    "xc70": "XC70",
    "xc90": "XC90",
    "xd": "xD",
    "xl7": "XL7",
}


def _format_part(raw_part: str) -> str:
    tokens = []
    for word in raw_part.replace("_", " ").split():
        lower_w = word.lower()
        if lower_w in _SPECIAL_DISPLAY_TOKENS:
            tokens.append(_SPECIAL_DISPLAY_TOKENS[lower_w])
            continue
        subparts = []
        for sub in word.split("-"):
            low_sub = sub.lower()
            if low_sub in _SPECIAL_DISPLAY_TOKENS:
                subparts.append(_SPECIAL_DISPLAY_TOKENS[low_sub])
            elif any(ch.isdigit() for ch in sub) and any(ch.isalpha() for ch in sub):
                subparts.append(sub.upper())
            else:
                subparts.append(sub.capitalize())
        tokens.append("-".join(subparts))
    return " ".join(tokens)


def format_display_name(raw: str) -> str:
    """Converts canonical slugs (e.g. toyota__camry, mazda__rx7) into clean UI titles."""
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if "__" in cleaned:
        brand, model = cleaned.split("__", 1)
        return f"{_format_part(brand)} {_format_part(model)}".strip()
    return _format_part(cleaned)


def format_canonical_display_name(
    raw: str,
    make_display_names: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
) -> str:
    """Formats make or canonical `make__model` labels using label_maps display overrides when present."""
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if "__" in cleaned:
        mk_slug, _ = cleaned.split("__", 1)
        mk_d = (make_display_names or {}).get(mk_slug)
        md_d = (model_display_names or {}).get(cleaned)
        if mk_d and md_d:
            return f"{mk_d} {md_d}".strip()
        if md_d:
            return f"{_format_part(mk_slug)} {md_d}".strip()
        return format_display_name(cleaned)
    if make_display_names and cleaned in make_display_names:
        return make_display_names[cleaned]
    return format_display_name(cleaned)


class ImageSize(BaseModel):
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class BBoxXYXY(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class BBoxNorm(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)


class TopKPrediction(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class FewShotCandidate(BaseModel):
    canonical_class: str
    display_name: str
    make: str
    body_type: str
    similarity: float = Field(..., ge=-1.0, le=1.0)
    is_few_shot: bool


class MakeAttribute(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: list[TopKPrediction]


class ModelAttribute(BaseModel):
    canonical_class: str
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: list[TopKPrediction]
    few_shot_match: FewShotCandidate | None = None


class YearAttribute(BaseModel):
    predicted: float
    predicted_rounded: int
    range_low: int
    range_high: int
    std: float = Field(..., ge=0.0)


class BodyTypeAttribute(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: list[TopKPrediction]


class ColorAttribute(BaseModel):
    label: str
    display_name: str
    group: str
    hex: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: list[TopKPrediction]


class VehicleAttributes(BaseModel):
    make: MakeAttribute
    model: ModelAttribute
    year: YearAttribute
    body_type: BodyTypeAttribute
    color: ColorAttribute


class SimilarVehicle(BaseModel):
    rel_path: str
    canonical_class: str
    display_name: str
    make: str
    year: int | None
    body_type: str
    similarity: float = Field(..., ge=-1.0, le=1.0)
    thumbnail_url: str


class VehicleDetectionResult(BaseModel):
    vehicle_index: int = Field(..., ge=0)
    detected_by_yolo: bool
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    area_ratio: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: BBoxXYXY
    bbox_norm: BBoxNorm
    attributes: VehicleAttributes
    similar_vehicles: list[SimilarVehicle]


class InferenceTimingsMs(BaseModel):
    decode_ms: float = Field(..., ge=0.0)
    detect_ms: float = Field(..., ge=0.0)
    classify_ms: float = Field(..., ge=0.0)
    retrieve_ms: float = Field(..., ge=0.0)
    total_ms: float = Field(..., ge=0.0)


class AnalyzeResponse(BaseModel):
    model_version: str
    device: str
    image_size: ImageSize
    num_vehicles: int = Field(..., ge=1)
    primary_vehicle_index: int = Field(..., ge=0)
    vehicles: list[VehicleDetectionResult]
    timings_ms: InferenceTimingsMs


class HealthResponse(BaseModel):
    status: str
    model_version: str
    device: str
    ready: bool
    num_makes: int
    num_models: int
    num_prototype_classes: int
    num_bodies: int
    num_colors: int
    gallery_size: int
    yolo_available: bool


class CatalogColorItem(BaseModel):
    label: str
    group: str
    hex: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")


class CatalogResponse(BaseModel):
    model_version: str
    makes: list[str]
    models: list[str]
    prototype_classes: list[str]
    body_types: list[str]
    colors: list[CatalogColorItem]
    year_min: int
    year_max: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str | None = None
