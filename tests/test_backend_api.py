"""Unit and live integration tests for AutoVision FastAPI backend."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CACHE_ULTRALYTICS = BASE_DIR / ".cache" / "ultralytics"
CACHE_ULTRALYTICS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ULTRALYTICS))

from backend.config import BackendSettings, load_settings  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.ml.classifier import ClassifierBatchOutput  # noqa: E402
from backend.ml.detector import DetectedBox  # noqa: E402
from backend.ml.pipeline import AutoVisionPipeline  # noqa: E402
from backend.schemas import (  # noqa: E402
    AnalyzeResponse,
    FewShotCandidate,
    SimilarVehicle,
    format_display_name,
)
from build_attributes_manifest import BODY_TYPE_CLASSES  # noqa: E402
from multitask_model import COLOR_CLASSES, COLOR_GROUPS_11, COLOR_HEX, YEAR_MAX, YEAR_MIN  # noqa: E402


class _FakeDetector:
    def __init__(self) -> None:
        self.seen_sizes: list[tuple[int, int]] = []

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        self.seen_sizes.append(image.size)
        return [
            DetectedBox(x1=5, y1=10, x2=75, y2=140, confidence=0.94, area_ratio=0.71, detected_by_yolo=True),
            DetectedBox(x1=2, y1=5, x2=30, y2=50, confidence=0.78, area_ratio=0.10, detected_by_yolo=True),
        ]


class _FakeClassifier:
    def __init__(self) -> None:
        self._lmaps = {
            "make_classes": ["mazda", "toyota"],
            "model_classes": ["mazda__rx7", "toyota__camry", "toyota__rav4"],
            "body_classes": list(BODY_TYPE_CLASSES),
            "color_classes": list(COLOR_CLASSES),
            "color_groups_11": COLOR_GROUPS_11,
            "color_hex": COLOR_HEX,
            "year_min": YEAR_MIN,
            "year_max": YEAR_MAX,
        }

    @property
    def label_maps(self):
        return self._lmaps

    @property
    def device(self) -> str:
        return "cpu"

    def classify_crops(self, crops: list[Image.Image], use_tta: bool = False) -> ClassifierBatchOutput:
        n = len(crops)
        emb = np.tile(np.eye(1, 512, 0, dtype=np.float32), (n, 1))
        mk = np.tile(np.array([0.08, 0.92], dtype=np.float32), (n, 1))
        md = np.tile(np.array([0.05, 0.90, 0.05], dtype=np.float32), (n, 1))
        bd = np.zeros((n, len(BODY_TYPE_CLASSES)), dtype=np.float32)
        bd[:, 0] = 0.89
        bd[:, 4] = 0.11
        cl = np.zeros((n, len(COLOR_CLASSES)), dtype=np.float32)
        cl[:, COLOR_CLASSES.index("Black")] = 0.88
        cl[:, COLOR_CLASSES.index("Gray")] = 0.12
        return ClassifierBatchOutput(
            embeddings=emb,
            make_probs=mk,
            model_probs=md,
            body_probs=bd,
            year_expected=np.full(n, 2010.6, dtype=np.float32),
            year_std=np.full(n, 1.4, dtype=np.float32),
            color_probs=cl,
        )


class _FakeRetriever:
    @property
    def gallery_size(self) -> int:
        return 2

    @property
    def prototype_classes(self) -> list[str]:
        return ["mazda__rx7", "toyota__camry", "toyota__rav4", "tesla__s"]

    def match_prototype(self, embedding: np.ndarray) -> FewShotCandidate:
        return FewShotCandidate(
            canonical_class="toyota__camry",
            display_name="Toyota Camry",
            make="toyota",
            body_type="Sedan",
            similarity=0.9512,
            is_few_shot=False,
        )

    def search_similar(self, embedding: np.ndarray, top_k: int = 5) -> list[SimilarVehicle]:
        base = [
            SimilarVehicle(
                rel_path="toyota_camry_2011/sample.jpg",
                canonical_class="toyota__camry",
                display_name="Toyota Camry",
                make="toyota",
                year=2011,
                body_type="Sedan",
                similarity=0.9731,
                thumbnail_url="/api/v1/gallery/thumbnail?rel_path=" + quote("toyota_camry_2011/sample.jpg", safe="/"),
            ),
            SimilarVehicle(
                rel_path="toyota_camry_2011/sample2.jpg",
                canonical_class="toyota__camry",
                display_name="Toyota Camry",
                make="toyota",
                year=2010,
                body_type="Sedan",
                similarity=0.9610,
                thumbnail_url="/api/v1/gallery/thumbnail?rel_path=" + quote("toyota_camry_2011/sample2.jpg", safe="/"),
            ),
        ]
        return base[:top_k]


def _make_fake_app(data_dir: Path, max_upload_bytes: int = 5_000_000):
    settings = BackendSettings(
        project_root=BASE_DIR,
        models_dir=BASE_DIR / "models",
        manifest_path=BASE_DIR / "data_manifests" / "local_vmmrdb_multitask_manifest.csv",
        data_dir=data_dir,
        yolo_weights=BASE_DIR / "yolov8n.pt",
        max_upload_bytes=max_upload_bytes,
    )
    detector = _FakeDetector()
    pipeline = AutoVisionPipeline(settings, detector, _FakeClassifier(), _FakeRetriever())
    return create_app(settings=settings, pipeline=pipeline), detector


class TestBackendApiUnit(unittest.TestCase):
    def test_format_display_name_conventions(self) -> None:
        self.assertEqual(format_display_name("toyota__camry"), "Toyota Camry")
        self.assertEqual(format_display_name("mercedes-benz__c-class"), "Mercedes-Benz C-Class")
        self.assertEqual(format_display_name("mazda__rx7"), "Mazda RX-7")
        self.assertEqual(format_display_name("mazda__rx8"), "Mazda RX-8")
        self.assertEqual(format_display_name("volkswagen__gti"), "Volkswagen GTI")
        self.assertEqual(format_display_name("nissan__350z"), "Nissan 350Z")
        self.assertEqual(format_display_name("subaru__wrx"), "Subaru WRX")
        self.assertEqual(format_display_name("honda__cr-v"), "Honda CR-V")
        self.assertEqual(format_display_name("nissan__gtr"), "Nissan GT-R")
        self.assertEqual(format_display_name("mg__mgb"), "MG MGB")

    def test_health_and_catalog_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = _make_fake_app(Path(tmp))
            with TestClient(app) as client:
                h_resp = client.get("/health")
                self.assertEqual(h_resp.status_code, 200)
                h = h_resp.json()
                self.assertEqual(h["status"], "ok")
                self.assertTrue(h["ready"])
                self.assertEqual(h["num_makes"], 2)
                self.assertEqual(h["num_models"], 3)
                self.assertEqual(h["gallery_size"], 2)

                c_resp = client.get("/api/v1/catalog")
                self.assertEqual(c_resp.status_code, 200)
                c = c_resp.json()
                self.assertEqual(len(c["colors"]), 15)
                self.assertEqual(c["year_min"], YEAR_MIN)
                self.assertEqual(c["year_max"], YEAR_MAX)

    def test_analyze_multi_vehicle_and_exif_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, detector = _make_fake_app(Path(tmp))
            im = Image.new("RGB", (160, 80), (20, 20, 20))
            exif = im.getexif()
            exif[274] = 6  # Rotate 270 -> upright 80x160
            buf = io.BytesIO()
            im.save(buf, format="JPEG", exif=exif)
            im.close()

            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/analyze?top_k_similar=2",
                    files={"file": ("phone.jpg", buf.getvalue(), "image/jpeg")},
                )
            self.assertEqual(resp.status_code, 200)
            parsed = AnalyzeResponse.model_validate(resp.json())
            self.assertEqual(detector.seen_sizes[0], (80, 160))
            self.assertEqual((parsed.image_size.width, parsed.image_size.height), (80, 160))
            self.assertEqual(parsed.num_vehicles, 2)
            v0 = parsed.vehicles[0]
            self.assertEqual(v0.attributes.make.label, "toyota")
            self.assertEqual(v0.attributes.model.display_name, "Toyota Camry")
            self.assertEqual(v0.attributes.color.label, "Black")
            self.assertEqual(v0.attributes.color.hex, COLOR_HEX["Black"])
            self.assertEqual(len(v0.similar_vehicles), 2)

    def test_analyze_rejects_oversized_and_corrupt_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _ = _make_fake_app(Path(tmp), max_upload_bytes=128)
            with TestClient(app) as client:
                r_empty = client.post("/api/v1/analyze", files={"file": ("e.jpg", b"", "image/jpeg")})
                self.assertEqual(r_empty.status_code, 400)
                self.assertEqual(r_empty.json()["error_code"], "INVALID_IMAGE")

                r_bad = client.post("/api/v1/analyze", files={"file": ("b.jpg", b"not-an-image", "image/jpeg")})
                self.assertEqual(r_bad.status_code, 400)
                self.assertEqual(r_bad.json()["error_code"], "INVALID_IMAGE")

                r_big = client.post("/api/v1/analyze", files={"file": ("big.jpg", b"x" * 200, "image/jpeg")})
                self.assertEqual(r_big.status_code, 413)
                self.assertEqual(r_big.json()["error_code"], "PAYLOAD_TOO_LARGE")

    def test_gallery_thumbnail_serves_jpeg_and_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            sample_path = data_dir / "toyota_camry_2011" / "sample.jpg"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (600, 450), (180, 40, 40)).save(sample_path, format="JPEG")

            app, _ = _make_fake_app(data_dir)
            with TestClient(app) as client:
                ok = client.get("/api/v1/gallery/thumbnail?rel_path=toyota_camry_2011/sample.jpg&max_width=200")
                self.assertEqual(ok.status_code, 200)
                self.assertEqual(ok.headers["content-type"], "image/jpeg")
                with Image.open(io.BytesIO(ok.content)) as thumb:
                    self.assertEqual(thumb.size, (200, 150))

                for bad_rel in ("../secret.jpg", "C:/Windows/win.ini", "/etc/passwd"):
                    r = client.get("/api/v1/gallery/thumbnail", params={"rel_path": bad_rel})
                    self.assertEqual(r.status_code, 403, f"Expected 403 for {bad_rel}")
                    self.assertEqual(r.json()["error_code"], "UNSAFE_PATH")

                r_miss = client.get("/api/v1/gallery/thumbnail?rel_path=toyota_camry_2011/missing.jpg")
                self.assertEqual(r_miss.status_code, 404)
                self.assertEqual(r_miss.json()["error_code"], "NOT_FOUND")


class TestBackendLiveIntegration(unittest.TestCase):
    def test_live_pipeline_on_real_checkpoint_and_photo(self) -> None:
        settings = load_settings(BASE_DIR)
        sample_img = Path(r"D:\Projects\bitirme\data\toyota_camry_2011\2011 Toyota Camry_00D0D_78Nwt055JIt_600x450.jpg")
        ckpt = settings.models_dir / "autovision_multitask_best.pt"
        lmaps = settings.models_dir / "label_maps.json"
        if not (ckpt.is_file() and lmaps.is_file() and sample_img.is_file()):
            self.skipTest("Real Stage 4 weights or sample image not present")

        app = create_app(settings=settings)
        with TestClient(app) as client:
            h = client.get("/health")
            self.assertEqual(h.status_code, 200)
            self.assertGreater(h.json()["gallery_size"], 70000)

            resp = client.post(
                "/api/v1/analyze?top_k_similar=5",
                files={"file": (sample_img.name, sample_img.read_bytes(), "image/jpeg")},
            )
            self.assertEqual(resp.status_code, 200)
            parsed = AnalyzeResponse.model_validate(resp.json())
            attrs = parsed.vehicles[0].attributes
            self.assertEqual(attrs.make.label, "toyota")
            self.assertEqual(attrs.model.canonical_class, "toyota__camry")
            self.assertEqual(attrs.body_type.label, "Sedan")
            self.assertTrue(2008.0 <= attrs.year.predicted <= 2014.0)
            self.assertEqual(attrs.color.label, "Black")
            self.assertEqual(len(parsed.vehicles[0].similar_vehicles), 5)

            thumb_url = parsed.vehicles[0].similar_vehicles[0].thumbnail_url
            t_resp = client.get(thumb_url)
            self.assertEqual(t_resp.status_code, 200)
            self.assertEqual(t_resp.headers["content-type"], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
