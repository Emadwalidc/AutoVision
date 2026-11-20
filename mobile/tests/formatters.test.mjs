import assert from "node:assert/strict";
import test from "node:test";

import {
  computeContainedImageRect,
  DEFAULT_SERVER_URL,
  formatModelTitle,
  formatPercent,
  formatRelativeTime,
  formatYearRange,
  formatYearSpanOnly,
  normalizeServerUrl,
  resolveThumbnailUrl,
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
