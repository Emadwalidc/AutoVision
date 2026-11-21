import AsyncStorage from "@react-native-async-storage/async-storage";

import type { ScanHistoryItem } from "../types/api";
import { DEFAULT_SERVER_URL, normalizeServerUrl } from "../utils/formatters";

const SERVER_URL_KEY = "@autovision/server_url_v2";
const LEGACY_SETTINGS_KEY = "@autovision/settings_v1";
const HISTORY_KEY = "@autovision/history_v1";
const MAX_HISTORY_ITEMS = 30;

export async function loadServerUrl(): Promise<string> {
  try {
    const stored = await AsyncStorage.getItem(SERVER_URL_KEY);
    if (stored) {
      const normalized = normalizeServerUrl(stored);
      if (normalized) {
        return normalized;
      }
    }

    const legacyRaw = await AsyncStorage.getItem(LEGACY_SETTINGS_KEY);
    if (legacyRaw) {
      const legacy = JSON.parse(legacyRaw) as { serverUrl?: unknown };
      if (typeof legacy.serverUrl === "string") {
        const normalized = normalizeServerUrl(legacy.serverUrl);
        if (normalized) {
          return normalized;
        }
      }
    }
  } catch {
    // Fall back silently to default LAN address.
  }
  return DEFAULT_SERVER_URL;
}

export async function saveServerUrl(rawUrl: string): Promise<string> {
  const clean = normalizeServerUrl(rawUrl) || DEFAULT_SERVER_URL;
  await AsyncStorage.setItem(SERVER_URL_KEY, clean);
  return clean;
}

export async function loadHistory(): Promise<ScanHistoryItem[]> {
  try {
    const raw = await AsyncStorage.getItem(HISTORY_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter(
        (item): item is ScanHistoryItem =>
          Boolean(item) &&
          typeof item === "object" &&
          typeof (item as ScanHistoryItem).id === "string" &&
          typeof (item as ScanHistoryItem).imageUri === "string" &&
          Boolean((item as ScanHistoryItem).response?.vehicles),
      )
      .slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

export async function appendHistoryItem(
  item: ScanHistoryItem,
  current: ScanHistoryItem[],
): Promise<ScanHistoryItem[]> {
  const deduped = current.filter((existing) => existing.id !== item.id);
  const next = [item, ...deduped].slice(0, MAX_HISTORY_ITEMS);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

export async function removeHistoryItem(
  id: string,
  current: ScanHistoryItem[],
): Promise<ScanHistoryItem[]> {
  const next = current.filter((item) => item.id !== id);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

export async function updateHistorySelection(
  id: string,
  selectedModelIndices: Record<number, number>,
  current: ScanHistoryItem[],
): Promise<ScanHistoryItem[]> {
  const next = current.map((item) =>
    item.id === id
      ? { ...item, selectedModelIndices: { ...selectedModelIndices } }
      : item,
  );
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

export async function clearHistory(): Promise<void> {
  await AsyncStorage.removeItem(HISTORY_KEY);
}
