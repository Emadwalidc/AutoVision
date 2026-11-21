import type { TopKPrediction, VehicleDetectionResult } from "../types/api";

export const DEFAULT_SERVER_URL = "http://192.168.1.13:8000";

export interface ContainedRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Normalizes a LAN server address (trims spaces, adds `http://` when omitted,
 * and removes trailing slashes).
 */
export function normalizeServerUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) {
    return "";
  }
  const withScheme = /^https?:\/\//i.test(trimmed)
    ? trimmed
    : `http://${trimmed}`;
  return withScheme.replace(/\/+$/, "");
}

/**
 * Formats a score in [0, 1] as a bounded percentage string.
 */
export function formatPercent(value: number, decimals = 0): string {
  if (!Number.isFinite(value)) {
    return decimals > 0 ? `0.${"0".repeat(decimals)}%` : "0%";
  }
  const clamped = Math.min(1, Math.max(0, value));
  return `${(clamped * 100).toFixed(decimals)}%`;
}

export type ConfidenceTier = "high" | "medium" | "low";

/**
 * Categorizes model prediction confidence into visual UI confidence bands.
 */
export function getConfidenceTier(confidence: number): ConfidenceTier {
  if (!Number.isFinite(confidence)) {
    return "low";
  }
  if (confidence >= 0.8) {
    return "high";
  }
  if (confidence >= 0.5) {
    return "medium";
  }
  return "low";
}

/**
 * Formats internal backend model version strings for compact UI badges.
 */
export function formatModelVersionLabel(rawVersion: string): string {
  const clean = rawVersion.trim().toLowerCase();
  if (!clean) {
    return "AutoVision";
  }
  if (clean.includes("v2")) {
    return "AutoVision v2";
  }
  if (clean.includes("v1")) {
    return "AutoVision v1";
  }
  return "AutoVision";
}

/**
 * Compacts large gallery or catalog counts (e.g. 22652 -> "22.7k").
 */
export function formatCompactCount(count: number): string {
  if (!Number.isFinite(count) || count < 0) {
    return "0";
  }
  if (count < 1000) {
    return String(Math.round(count));
  }
  return `${(count / 1000).toFixed(1).replace(/\.0$/, "")}k`;
}

/**
 * Formats pipeline latency cleanly ("840 ms" or "1.4s").
 */
export function formatLatencySeconds(totalMs: number): string {
  if (!Number.isFinite(totalMs) || totalMs <= 0) {
    return "";
  }
  if (totalMs < 1000) {
    return `${Math.round(totalMs)} ms`;
  }
  return `${(totalMs / 1000).toFixed(1)}s`;
}

/**
 * Attaches or overrides `max_width` query parameter on gallery thumbnail URLs.
 */
export function withThumbnailWidth(url: string, maxWidth: number): string {
  const clean = url.trim();
  if (!clean) {
    return "";
  }
  const width = Math.max(64, Math.round(maxWidth));
  if (/([?&])max_width=\d+/i.test(clean)) {
    return clean.replace(/([?&])max_width=\d+/i, `$1max_width=${width}`);
  }
  return `${clean}${clean.includes("?") ? "&" : "?"}max_width=${width}`;
}

/**
 * Strips a redundant leading brand prefix from a model display string so
 * "Toyota" + "Toyota Camry" renders cleanly as eyebrow "Toyota" / title "Camry".
 */
export function formatModelTitle(
  makeDisplay: string,
  modelDisplay: string,
): string {
  const cleanMake = makeDisplay.trim();
  const cleanModel = modelDisplay.trim();
  if (!cleanModel) {
    return cleanMake || "Vehicle";
  }
  if (!cleanMake) {
    return cleanModel;
  }
  const lowerMake = cleanMake.toLowerCase();
  const lowerModel = cleanModel.toLowerCase();
  if (lowerModel.startsWith(lowerMake)) {
    const rest = cleanModel.slice(cleanMake.length).replace(/^[\s\-_]+/, "");
    if (rest.length > 0) {
      return rest;
    }
  }
  return cleanModel;
}

/**
 * Returns a clean year interval like "2018–2020", or "" when equal/unavailable.
 */
export function formatYearSpanOnly(low: number, high: number): string {
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    return "";
  }
  const lowRound = Math.round(Math.min(low, high));
  const highRound = Math.round(Math.max(low, high));
  if (lowRound === highRound) {
    return "";
  }
  return `${lowRound}\u2013${highRound}`;
}

/**
 * Formats predicted year paired with its uncertainty interval when distinct.
 */
export function formatYearRange(
  predicted: number,
  low: number,
  high: number,
): string {
  if (!Number.isFinite(predicted)) {
    return "Unknown";
  }
  const predRound = Math.round(predicted);
  const span = formatYearSpanOnly(low, high);
  return span ? `${predRound} (${span})` : `${predRound}`;
}

/**
 * Joins a server base URL with a relative gallery thumbnail path without
 * duplicating slashes. Passes absolute URLs through untouched.
 */
export function resolveThumbnailUrl(
  baseUrl: string,
  thumbnailUrl: string,
): string {
  const cleanThumb = thumbnailUrl.trim();
  if (!cleanThumb) {
    return "";
  }
  if (/^https?:\/\//i.test(cleanThumb)) {
    return cleanThumb;
  }
  const cleanBase = normalizeServerUrl(baseUrl);
  const prefixedThumb = cleanThumb.startsWith("/")
    ? cleanThumb
    : `/${cleanThumb}`;
  return `${cleanBase}${prefixedThumb}`;
}

/**
 * Computes the rendered image rectangle inside a `resizeMode="contain"` frame
 * so normalized bounding boxes align accurately with letterboxed photographs.
 */
export function computeContainedImageRect(
  containerW: number,
  containerH: number,
  imgW: number,
  imgH: number,
): ContainedRect {
  if (
    !Number.isFinite(containerW) ||
    !Number.isFinite(containerH) ||
    !Number.isFinite(imgW) ||
    !Number.isFinite(imgH) ||
    containerW <= 0 ||
    containerH <= 0 ||
    imgW <= 0 ||
    imgH <= 0
  ) {
    return { left: 0, top: 0, width: 0, height: 0 };
  }

  const containerAspect = containerW / containerH;
  const imageAspect = imgW / imgH;

  if (imageAspect > containerAspect) {
    const width = containerW;
    const height = containerW / imageAspect;
    return {
      left: 0,
      top: (containerH - height) / 2,
      width,
      height,
    };
  }

  const height = containerH;
  const width = containerH * imageAspect;
  return {
    left: (containerW - width) / 2,
    top: 0,
    width,
    height,
  };
}

/**
 * Formats an ISO timestamp into a compact relative age label for Garage cards.
 */
export function formatRelativeTime(
  isoTimestamp: string,
  nowMs = Date.now(),
): string {
  const parsedMs = Date.parse(isoTimestamp);
  if (!Number.isFinite(parsedMs)) {
    return "Unknown";
  }
  const diffSec = Math.max(0, Math.floor((nowMs - parsedMs) / 1000));
  if (diffSec < 60) {
    return "Just now";
  }
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) {
    return `${diffMin}m ago`;
  }
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) {
    return `${diffHours}h ago`;
  }
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) {
    return `${diffDays}d ago`;
  }
  return new Date(parsedMs).toISOString().slice(0, 10);
}

function slugifyToken(raw: string): string {
  return raw
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

/**
 * Splits a Top-K model candidate display name into clean brand eyebrow and
 * concise model title, even when a candidate belongs to another make.
 */
export function splitCandidateMakeAndModel(
  candidate: { label: string; display_name: string },
  fallbackMakeLabel: string,
  fallbackMakeDisplay: string,
  makeTop3?: Array<{ label: string; display_name: string }>,
): { makeDisplay: string; modelTitle: string } {
  const makeSlug = candidate.label.includes("__")
    ? candidate.label.split("__")[0].trim()
    : "";
  const normalizedSlug = makeSlug ? slugifyToken(makeSlug) : "";

  if (normalizedSlug && normalizedSlug === slugifyToken(fallbackMakeLabel)) {
    return {
      makeDisplay: fallbackMakeDisplay,
      modelTitle: formatModelTitle(fallbackMakeDisplay, candidate.display_name),
    };
  }

  if (normalizedSlug && makeTop3 && makeTop3.length > 0) {
    const matched = makeTop3.find(
      (m) => slugifyToken(m.label) === normalizedSlug,
    );
    if (matched) {
      return {
        makeDisplay: matched.display_name,
        modelTitle: formatModelTitle(
          matched.display_name,
          candidate.display_name,
        ),
      };
    }
  }

  if (normalizedSlug) {
    const tokens = candidate.display_name.trim().split(/\s+/);
    for (let len = 1; len < tokens.length; len += 1) {
      const prefix = tokens.slice(0, len).join(" ");
      if (slugifyToken(prefix) === normalizedSlug) {
        return {
          makeDisplay: prefix,
          modelTitle: tokens.slice(len).join(" "),
        };
      }
    }
  }

  return {
    makeDisplay: fallbackMakeDisplay,
    modelTitle: formatModelTitle(fallbackMakeDisplay, candidate.display_name),
  };
}

export interface ResolvedModelSelection {
  index: number;
  isOverride: boolean;
  canonicalClass: string;
  displayName: string;
  makeDisplay: string;
  modelTitle: string;
  confidence: number;
  candidates: TopKPrediction[];
}

/**
 * Resolves the active model candidate (default #1 or user-selected Top-K index)
 * consistently across VehicleCard, SimilarCarsStrip, PhotoFrame, and Garage.
 */
export function resolveVehicleModelSelection(
  vehicle: VehicleDetectionResult,
  selectedIndex?: number,
): ResolvedModelSelection {
  const { make, model } = vehicle.attributes;
  const candidates: TopKPrediction[] =
    model.top3 && model.top3.length > 0
      ? model.top3
      : [
          {
            label: model.canonical_class,
            display_name: model.display_name,
            confidence: model.confidence,
          },
        ];

  const rawIdx =
    typeof selectedIndex === "number" && Number.isFinite(selectedIndex)
      ? Math.trunc(selectedIndex)
      : 0;
  const index = rawIdx >= 0 && rawIdx < candidates.length ? rawIdx : 0;

  const active = candidates[index];
  const { makeDisplay, modelTitle } = splitCandidateMakeAndModel(
    active,
    make.label,
    make.display_name,
    make.top3,
  );

  return {
    index,
    isOverride: index > 0,
    canonicalClass: active.label,
    displayName: active.display_name,
    makeDisplay,
    modelTitle,
    confidence: active.confidence,
    candidates,
  };
}
