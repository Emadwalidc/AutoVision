"""Unit tests for Body Type taxonomy mapping and multitask manifest generation."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_attributes_manifest import (  # noqa: E402
    BODY_TYPE_CLASSES,
    MODEL_BODY_TYPE,
    resolve_body_type,
    verify_classes_coverage,
)


class TestBodyTypeMapping(unittest.TestCase):
    def test_taxonomy_size_and_valid_labels(self) -> None:
        self.assertEqual(len(MODEL_BODY_TYPE), 342)
        self.assertEqual(len(BODY_TYPE_CLASSES), 8)
        for cls_name, btype in MODEL_BODY_TYPE.items():
            self.assertIn(btype, BODY_TYPE_CLASSES, f"Invalid body type for {cls_name}: {btype}")

    def test_full_coverage_of_classes_csv(self) -> None:
        classes_csv = BASE_DIR / "data_manifests" / "local_vmmrdb_classes.csv"
        if not classes_csv.is_file():
            self.skipTest("local_vmmrdb_classes.csv not found")
        verify_classes_coverage(classes_csv)

    def test_row_level_overrides(self) -> None:
        self.assertEqual(
            resolve_body_type("volkswagen__passat", "passat_wagon", "Wagon", 2005, ""),
            "Wagon",
        )
        self.assertEqual(
            resolve_body_type("toyota__solara", "solara_convertable", "Convertible", 2005, ""),
            "Convertible",
        )
        self.assertEqual(
            resolve_body_type("saturn__s-series", "sw2", "", 2000, ""),
            "Wagon",
        )
        self.assertEqual(
            resolve_body_type("saturn__l-series", "lw200", "", 2002, ""),
            "Wagon",
        )
        self.assertEqual(
            resolve_body_type("mercury__monterey", "monterey", "", 2005, ""),
            "Minivan",
        )
        self.assertEqual(
            resolve_body_type("mercury__monterey", "monterey", "", 1955, ""),
            "Sedan",
        )
        self.assertEqual(
            resolve_body_type("suzuki__boulevard", "boulevard", "", 2006, "non_car"),
            "",
        )

    def test_generated_multitask_manifest_integrity(self) -> None:
        mt_manifest = BASE_DIR / "data_manifests" / "local_vmmrdb_multitask_manifest.csv"
        if not mt_manifest.is_file():
            self.skipTest("local_vmmrdb_multitask_manifest.csv not generated yet")

        with open(mt_manifest, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 91678)
        clean_rows = [r for r in rows if r["is_clean"].strip().lower() == "true"]
        self.assertEqual(len(clean_rows), 89535)
        for r in clean_rows:
            self.assertIn(r["body_type"], BODY_TYPE_CLASSES, r["rel_path"])


if __name__ == "__main__":
    unittest.main()
