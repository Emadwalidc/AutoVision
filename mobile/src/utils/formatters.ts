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
