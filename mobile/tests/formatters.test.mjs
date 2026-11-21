import assert from "node:assert/strict";
import test from "node:test";

import {
  computeContainedImageRect,
  DEFAULT_SERVER_URL,
  formatCompactCount,
  formatLatencySeconds,
  formatModelTitle,
  formatModelVersionLabel,
  formatPercent,
  formatRelativeTime,
  formatYearRange,
  formatYearSpanOnly,
  getConfidenceTier,
  normalizeServerUrl,
  resolveThumbnailUrl,
  resolveVehicleModelSelection,
  splitCandidateMakeAndModel,
  withThumbnailWidth,
} from "../src/utils/formatters.ts";

test("normalizeServerUrl trims spaces, injects http://, and strips trailing slashes", () => {
  assert.equal(normalizeServerUrl("  192.168.1.13:8000/  "), DEFAULT_SERVER_URL);
  assert.equal(normalizeServerUrl("https://api.example.com///"), "https://api.example.com");
  assert.equal(normalizeServerUrl("   "), "");
});

test("formatPercent clamps bounds and handles non-finite inputs", () => {
  assert.equal(formatPercent(0.9551, 1), "95.5%");
  assert.equal(formatPercent(0.96, 0), "96%");
  assert.equal(formatPercent(1.4, 0), "100%");
  assert.equal(formatPercent(-0.2, 1), "0.0%");
  assert.equal(formatPercent(Number.NaN, 1), "0.0%");
});

test("formatModelTitle strips redundant brand prefixes cleanly", () => {
  assert.equal(formatModelTitle("Toyota", "Toyota Camry"), "Camry");
  assert.equal(formatModelTitle("Mercedes-Benz", "Mercedes-Benz E-Class"), "E-Class");
  assert.equal(formatModelTitle("Honda", "Civic"), "Civic");
  assert.equal(formatModelTitle("Audi", "Audi"), "Audi");
  assert.equal(formatModelTitle("Ford", ""), "Ford");
});

test("formatYearSpanOnly and formatYearRange format year spans accurately", () => {
  assert.equal(formatYearSpanOnly(2018.1, 2020.4), "2018\u20132020");
  assert.equal(formatYearSpanOnly(2020, 2020), "");
  assert.equal(formatYearRange(2019.4, 2018.1, 2020.4), "2019 (2018\u20132020)");
  assert.equal(formatYearRange(2020, 2020, 2020), "2020");
  assert.equal(formatYearRange(Number.NaN, 2018, 2020), "Unknown");
});

test("resolveThumbnailUrl avoids double slashes and respects absolute URLs", () => {
  assert.equal(
    resolveThumbnailUrl("http://192.168.1.13:8000/", "/api/v1/gallery/thumbnail?rel_path=a.jpg"),
    "http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg",
  );
  assert.equal(
    resolveThumbnailUrl("192.168.1.13:8000", "api/v1/gallery/thumbnail?rel_path=a.jpg"),
    "http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg",
  );
  assert.equal(
    resolveThumbnailUrl("http://192.168.1.13:8000", "https://cdn.example.com/x.jpg"),
    "https://cdn.example.com/x.jpg",
  );
});

test("computeContainedImageRect maps landscape and portrait images accurately inside letterboxes", () => {
  const wide = computeContainedImageRect(400, 400, 1600, 800);
  assert.deepEqual(wide, { left: 0, top: 100, width: 400, height: 200 });

  const tall = computeContainedImageRect(400, 200, 500, 1000);
  assert.deepEqual(tall, { left: 150, top: 0, width: 100, height: 200 });

  assert.deepEqual(computeContainedImageRect(0, 200, 500, 500), {
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });
});

test("formatRelativeTime returns readable compact durations", () => {
  const base = Date.parse("2026-11-20T12:00:00.000Z");
  assert.equal(formatRelativeTime("2026-11-20T11:59:30.000Z", base), "Just now");
  assert.equal(formatRelativeTime("2026-11-20T11:45:00.000Z", base), "15m ago");
  assert.equal(formatRelativeTime("2026-11-20T09:00:00.000Z", base), "3h ago");
  assert.equal(formatRelativeTime("2026-11-18T12:00:00.000Z", base), "2d ago");
  assert.equal(formatRelativeTime("2026-11-01T12:00:00.000Z", base), "2026-11-01");
});

test("getConfidenceTier categorizes high, medium, and low bands cleanly", () => {
  assert.equal(getConfidenceTier(0.92), "high");
  assert.equal(getConfidenceTier(0.8), "high");
  assert.equal(getConfidenceTier(0.65), "medium");
  assert.equal(getConfidenceTier(0.5), "medium");
  assert.equal(getConfidenceTier(0.49), "low");
  assert.equal(getConfidenceTier(Number.NaN), "low");
});

test("formatModelVersionLabel, formatCompactCount, formatLatencySeconds, and withThumbnailWidth work cleanly", () => {
  assert.equal(formatModelVersionLabel("autovision-v2-vitb16-multitask"), "AutoVision v2");
  assert.equal(formatModelVersionLabel("autovision-v1-vitb16-multitask"), "AutoVision v1");
  assert.equal(formatModelVersionLabel(""), "AutoVision");

  assert.equal(formatCompactCount(784), "784");
  assert.equal(formatCompactCount(22652), "22.7k");
  assert.equal(formatCompactCount(2000), "2k");

  assert.equal(formatLatencySeconds(842), "842 ms");
  assert.equal(formatLatencySeconds(1520), "1.5s");
  assert.equal(formatLatencySeconds(0), "");

  assert.equal(
    withThumbnailWidth("http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg", 720),
    "http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg&max_width=720",
  );
  assert.equal(
    withThumbnailWidth("http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg&max_width=360", 720),
    "http://192.168.1.13:8000/api/v1/gallery/thumbnail?rel_path=a.jpg&max_width=720",
  );
  assert.equal(withThumbnailWidth("", 720), "");
});

test("splitCandidateMakeAndModel resolves same-make, makeTop3, multi-word, hyphenated, and diacritic makes", () => {
  const sameMake = splitCandidateMakeAndModel(
    { label: "volkswagen__golf", display_name: "Volkswagen Golf" },
    "volkswagen",
    "Volkswagen",
  );
  assert.deepEqual(sameMake, { makeDisplay: "Volkswagen", modelTitle: "Golf" });

  const viaMakeTop3 = splitCandidateMakeAndModel(
    { label: "audi__a3", display_name: "Audi A3" },
    "volkswagen",
    "Volkswagen",
    [{ label: "audi", display_name: "Audi" }],
  );
  assert.deepEqual(viaMakeTop3, { makeDisplay: "Audi", modelTitle: "A3" });

  const hyphenated = splitCandidateMakeAndModel(
    { label: "mercedes-benz__a-class", display_name: "Mercedes-Benz A-Class" },
    "audi",
    "Audi",
  );
  assert.deepEqual(hyphenated, { makeDisplay: "Mercedes-Benz", modelTitle: "A-Class" });

  const multiWord = splitCandidateMakeAndModel(
    { label: "land_rover__range_rover", display_name: "Land Rover Range Rover" },
    "jeep",
    "Jeep",
  );
  assert.deepEqual(multiWord, { makeDisplay: "Land Rover", modelTitle: "Range Rover" });

  const diacritic = splitCandidateMakeAndModel(
    { label: "skoda__octavia", display_name: "Škoda Octavia" },
    "volkswagen",
    "Volkswagen",
  );
  assert.deepEqual(diacritic, { makeDisplay: "Škoda", modelTitle: "Octavia" });
});

test("resolveVehicleModelSelection switches candidates and clamps invalid indices to 0", () => {
  const sampleVehicle = {
    vehicle_index: 0,
    detected_by_yolo: true,
    detection_confidence: 0.95,
    area_ratio: 0.5,
    bbox_xyxy: { x1: 0, y1: 0, x2: 100, y2: 100 },
    bbox_norm: { x1: 0, y1: 0, x2: 1, y2: 1 },
    attributes: {
      make: {
        label: "volkswagen",
        display_name: "Volkswagen",
        confidence: 0.91,
        top3: [
          { label: "volkswagen", display_name: "Volkswagen", confidence: 0.91 },
          { label: "audi", display_name: "Audi", confidence: 0.04 },
        ],
      },
      model: {
        canonical_class: "volkswagen__gti",
        display_name: "Volkswagen GTI",
        confidence: 0.36,
        top3: [
          { label: "volkswagen__gti", display_name: "Volkswagen GTI", confidence: 0.36 },
          { label: "volkswagen__golf", display_name: "Volkswagen Golf", confidence: 0.27 },
          { label: "audi__a3", display_name: "Audi A3", confidence: 0.08 },
        ],
        few_shot_match: null,
      },
      year: { predicted: 2021, predicted_rounded: 2021, range_low: 2018, range_high: 2025, std: 2 },
      body_type: { label: "Hatchback", display_name: "Hatchback", confidence: 0.86, top3: [] },
      color: { label: "Blue", display_name: "Blue", group: "Blue", hex: "#2563EB", confidence: 0.96, top3: [] },
    },
    similar_vehicles: [],
  };

  const def = resolveVehicleModelSelection(sampleVehicle);
  assert.equal(def.index, 0);
  assert.equal(def.isOverride, false);
  assert.equal(def.canonicalClass, "volkswagen__gti");
  assert.equal(def.makeDisplay, "Volkswagen");
  assert.equal(def.modelTitle, "GTI");

  const switched = resolveVehicleModelSelection(sampleVehicle, 1);
  assert.equal(switched.index, 1);
  assert.equal(switched.isOverride, true);
  assert.equal(switched.canonicalClass, "volkswagen__golf");
  assert.equal(switched.makeDisplay, "Volkswagen");
  assert.equal(switched.modelTitle, "Golf");
  assert.equal(switched.confidence, 0.27);

  const crossMake = resolveVehicleModelSelection(sampleVehicle, 2);
  assert.equal(crossMake.index, 2);
  assert.equal(crossMake.makeDisplay, "Audi");
  assert.equal(crossMake.modelTitle, "A3");

  assert.equal(resolveVehicleModelSelection(sampleVehicle, -1).index, 0);
  assert.equal(resolveVehicleModelSelection(sampleVehicle, 99).index, 0);
  assert.equal(resolveVehicleModelSelection(sampleVehicle, Number.NaN).index, 0);

  const emptyTop3Vehicle = {
    ...sampleVehicle,
    attributes: {
      ...sampleVehicle.attributes,
      model: {
        ...sampleVehicle.attributes.model,
        top3: [],
      },
    },
  };
  const fallback = resolveVehicleModelSelection(emptyTop3Vehicle, 1);
  assert.equal(fallback.index, 0);
  assert.equal(fallback.isOverride, false);
  assert.equal(fallback.canonicalClass, "volkswagen__gti");
  assert.equal(fallback.candidates.length, 1);
});
