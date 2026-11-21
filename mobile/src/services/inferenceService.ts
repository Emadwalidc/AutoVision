import { NativeModules } from "react-native";

import type { AnalyzeResponse, HealthResponse } from "../types/api";
import {
  type AnalyzeMeta,
  analyzeVehiclePhoto,
  ApiClientError,
  checkHealth,
} from "./apiClient";

export type InferenceMode = "on-device" | "server";

interface AutoVisionOnDeviceNative {
  getHealth: () => Promise<string | HealthResponse>;
  analyzeImage: (
    imageUri: string,
    useTta: boolean,
  ) => Promise<string | AnalyzeResponse>;
}

function getNativeModule(): AutoVisionOnDeviceNative | null {
  const mod = NativeModules?.AutoVisionOnDevice as
    | Partial<AutoVisionOnDeviceNative>
    | undefined;
  if (
    mod &&
    typeof mod.getHealth === "function" &&
    typeof mod.analyzeImage === "function"
  ) {
    return mod as AutoVisionOnDeviceNative;
  }
  return null;
}

export function isOnDeviceAvailable(): boolean {
  return getNativeModule() !== null;
}

function parseNativePayload<T>(raw: string | T): T {
  if (typeof raw === "string") {
    return JSON.parse(raw) as T;
  }
  return raw;
}

export async function checkInferenceHealth(
  mode: InferenceMode,
  serverUrl: string,
): Promise<HealthResponse> {
  const native = getNativeModule();
  if (mode === "on-device" && native) {
    try {
      const raw = await native.getHealth();
      return parseNativePayload<HealthResponse>(raw);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "On-device engine unavailable.";
      throw new ApiClientError(msg, 0, "ONDEVICE_HEALTH_ERROR");
    }
  }
  return checkHealth(serverUrl);
}

export async function analyzeVehicle(
  mode: InferenceMode,
  serverUrl: string,
  imageUri: string,
  meta?: AnalyzeMeta,
): Promise<AnalyzeResponse> {
  const native = getNativeModule();
  if (mode === "on-device" && native) {
    try {
      const raw = await native.analyzeImage(imageUri, false);
      return parseNativePayload<AnalyzeResponse>(raw);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Could not process that photo on-device. Please try another image.";
      throw new ApiClientError(msg, 0, "ONDEVICE_ERROR");
    }
  }
  return analyzeVehiclePhoto(serverUrl, imageUri, meta);
}
