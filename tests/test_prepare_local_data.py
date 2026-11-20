"""Unit tests for src/prepare_local_data.py."""

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_local_data import (  # noqa: E402
    assign_splits,
    deduplicate_records,
    normalize_folder_label,
)


def _make_record(
    rel_path: str,
    folder: str,
    dhash: str,
    width: int = 600,
    height: int = 450,
    file_size: int = 50_000,
    is_corrupt: bool = False,
    is_tiny: bool = False,
) -> dict:
    meta = normalize_folder_label(folder)
    return {
        "rel_path": rel_path,
        **meta,
        "width": width,
        "height": height,
        "file_size": file_size,
        "dhash": dhash,
        "is_corrupt": is_corrupt,
        "is_tiny": is_tiny,
        "is_duplicate": False,
        "is_cross_class_dup": False,
        "is_clean": False,
        "split": "excluded",
    }


class TestNormalizeFolderLabel(unittest.TestCase):
    def test_mercedes_series_and_wagon(self) -> None:
        r1 = normalize_folder_label("mercedes benz_c300_2012")
        self.assertEqual(r1["make"], "mercedes-benz")
        self.assertEqual(r1["model"], "c-class")
        self.assertEqual(r1["canonical_class"], "mercedes-benz__c-class")
        self.assertEqual(r1["body_hint"], "")
        self.assertTrue(r1["year_valid"])

        r2 = normalize_folder_label("mercedes benz_e320_wagon_2003")
        self.assertEqual(r2["make"], "mercedes-benz")
        self.assertEqual(r2["model"], "e-class")
        self.assertEqual(r2["body_hint"], "Wagon")

        r3 = normalize_folder_label("mercedes benz_ml350_2010")
        self.assertEqual(r3["model"], "m-class")

    def test_toyota_trims_and_solara_typos(self) -> None:
        r1 = normalize_folder_label("toyota_camry_le_2005")
        self.assertEqual(r1["canonical_class"], "toyota__camry")

        r2 = normalize_folder_label("toyota_solara_convertable_2005")
        self.assertEqual(r2["canonical_class"], "toyota__solara")
        self.assertEqual(r2["body_hint"], "Convertible")

        r3 = normalize_folder_label("toyota_camry_solara_convertible_2006")
        self.assertEqual(r3["canonical_class"], "toyota__solara")
        self.assertEqual(r3["body_hint"], "Convertible")

        r4 = normalize_folder_label("toyota_corolla_wagon_1996")
        self.assertEqual(r4["canonical_class"], "toyota__corolla")
        self.assertEqual(r4["body_hint"], "Wagon")

    def test_vw_saturn_volvo_porsche_aliases(self) -> None:
        self.assertEqual(
            normalize_folder_label("volkswagen_bug_1970")["canonical_class"],
            "volkswagen__beetle",
        )
        self.assertEqual(
            normalize_folder_label("volkswagen_newbeetle_2004")["canonical_class"],
            "volkswagen__beetle",
        )
        self.assertEqual(
            normalize_folder_label("volkswagen_rabbit_2007")["canonical_class"],
            "volkswagen__golf",
        )
        vw_cab = normalize_folder_label("volkswagen_cabriolet_1990")
        self.assertEqual(vw_cab["canonical_class"], "volkswagen__cabrio")
        self.assertEqual(vw_cab["body_hint"], "Convertible")

        self.assertEqual(
            normalize_folder_label("saturn_view_2005")["canonical_class"],
            "saturn__vue",
        )
        self.assertEqual(
            normalize_folder_label("volvo_cx90_2004")["canonical_class"],
            "volvo__xc90",
        )
        self.assertEqual(
            normalize_folder_label("porsche_997_2008")["canonical_class"],
            "porsche__911",
        )

    def test_exclusions(self) -> None:
        for folder in (
            "suzuki_boulevard_2006",
            "polaris_slingshot_2015",
            "volvo_vnl_2007",
            "mitsubishi_fuso_2005",
            "toyota_flatbed_1986",
            "studebaker_flatbed_1959",
        ):
            self.assertEqual(
                normalize_folder_label(folder)["exclude_reason"],
                "non_car",
                folder,
            )

        for folder in (
            "toyota_scion_2008",
            "nissan_truck_1995",
            "toyota_van_1987",
            "mercedes benz_ht350_2006",
        ):
            self.assertEqual(
                normalize_folder_label(folder)["exclude_reason"],
                "ambiguous_label",
                folder,
            )

    def test_bogus_year_flagging(self) -> None:
        for folder in (
            "toyota_camry_1938",
            "porsche_911_1900",
            "volkswagen_beetle_1908",
            "mazda_rx8_2002",
        ):
            res = normalize_folder_label(folder)
            self.assertFalse(res["year_valid"], folder)
            self.assertEqual(res["exclude_reason"], "")


class TestDeduplicationAndSplits(unittest.TestCase):
    def test_same_class_dedup_keeps_highest_resolution(self) -> None:
        r_low = _make_record("toyota_camry_2005/a.jpg", "toyota_camry_2005", "abcd" * 4, 300, 225, 10_000)
        r_high = _make_record("toyota_camry_le_2005/b.jpg", "toyota_camry_le_2005", "abcd" * 4, 600, 450, 20_000)
        r_same_res_smaller = _make_record("toyota_camry_se_2005/c.jpg", "toyota_camry_se_2005", "abcd" * 4, 600, 450, 15_000)

        deduplicate_records([r_low, r_high, r_same_res_smaller])
        self.assertFalse(r_high["is_duplicate"])
        self.assertTrue(r_low["is_duplicate"])
        self.assertTrue(r_same_res_smaller["is_duplicate"])
        self.assertFalse(r_high["is_cross_class_dup"])

    def test_cross_class_dedup_excludes_all_colliding(self) -> None:
        r1 = _make_record("toyota_camry_2005/a.jpg", "toyota_camry_2005", "1234" * 4)
        r2 = _make_record("nissan_altima_2005/b.jpg", "nissan_altima_2005", "1234" * 4)
        r3 = _make_record("nissan_altima_2005/c.jpg", "nissan_altima_2005", "5678" * 4)

        deduplicate_records([r1, r2, r3])
        self.assertTrue(r1["is_cross_class_dup"])
        self.assertTrue(r2["is_cross_class_dup"])
        self.assertFalse(r3["is_cross_class_dup"])

    def test_stratified_split_and_few_shot(self) -> None:
        records = []
        for i in range(20):
            records.append(
                _make_record(f"toyota_camry_2007/img_{i:02d}.jpg", "toyota_camry_2007", f"{i:016x}")
            )
        for i in range(6):
            records.append(
                _make_record(f"porsche_912_1976/img_{i:02d}.jpg", "porsche_912_1976", f"{1000 + i:016x}")
            )
        bad = _make_record("suzuki_boulevard_2006/x.jpg", "suzuki_boulevard_2006", "ffff" * 4)
        corrupt = _make_record("toyota_camry_2007/corrupt.jpg", "toyota_camry_2007", "", is_corrupt=True)
        records.extend([bad, corrupt])

        deduplicate_records(records)
        assign_splits(records, min_class_images=15, seed=42)

        camry_splits = [r["split"] for r in records[:20]]
        self.assertEqual(camry_splits.count("val"), 2)
        self.assertEqual(camry_splits.count("test"), 2)
        self.assertEqual(camry_splits.count("train"), 16)

        few_splits = [r["split"] for r in records[20:26]]
        self.assertTrue(all(s == "few_shot" for s in few_splits))
        self.assertEqual(bad["split"], "excluded")
        self.assertEqual(corrupt["split"], "excluded")

        train_hashes = {r["dhash"] for r in records if r["split"] == "train"}
        val_hashes = {r["dhash"] for r in records if r["split"] == "val"}
        test_hashes = {r["dhash"] for r in records if r["split"] == "test"}
        self.assertEqual(train_hashes & val_hashes, set())
        self.assertEqual(train_hashes & test_hashes, set())
        self.assertEqual(val_hashes & test_hashes, set())


if __name__ == "__main__":
    unittest.main()
