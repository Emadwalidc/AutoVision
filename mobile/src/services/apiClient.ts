import { Platform } from "react-native";

import type {
  AnalyzeResponse,
  ErrorResponse,
  HealthResponse,
} from "../types/api";
import { normalizeServerUrl } from "../utils/formatters";

const SHORT_TIMEOUT_MS = 10_000;
const ANALYZE_TIMEOUT_MS = 45_000;
const DEFAULT_TOP_K_SIMILAR = 6;

export class ApiClientError extends Error {
  readonly status: number;
  readonly errorCode: string;

  constructor(message: string, status = 0, errorCode = "REQUEST_FAILED") {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.errorCode = errorCode;
  }
}

function parseRawResponseOrThrow<T>(status: number, bodyText: string): T {
  let parsed: unknown = null;
  if (bodyText.length > 0) {
    try {
      parsed = JSON.parse(bodyText);
    } catch {
      parsed = null;
    }
  }

  if (status < 200 || status >= 300) {
    const errPayload = parsed as Partial<ErrorResponse> | null;
    const message =
      errPayload?.message ||
      "Could not process that photo. Please try a different image.";
    throw new ApiClientError(
      message,
      status,
      errPayload?.error_code || `HTTP_${status}`,
    );
  }

  if (parsed === null) {
    throw new ApiClientError(
      "Received an unreadable response. Please try again.",
      status,
      "INVALID_JSON",
    );
  }

  return parsed as T;
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new ApiClientError(
        "The request took too long. Please try again.",
        0,
        "TIMEOUT",
      );
    }
    throw new ApiClientError(
      "Could not connect. Check that your phone and computer are on the same Wi-Fi.",
      0,
      "NETWORK_ERROR",
    );
  } finally {
    clearTimeout(timerId);
  }
}

/**
 * Native multipart upload via React Native's XMLHttpRequest -> RCTNetworking
 * bridge, avoiding Expo SDK 58's WinterCG `fetch` FormData `{ uri }` error.
 */
function uploadFormDataNative<T>(
  url: string,
  formData: FormData,
  timeoutMs: number,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.timeout = timeoutMs;
    xhr.setRequestHeader("Accept", "application/json");

    xhr.onload = () => {
      try {
        const result = parseRawResponseOrThrow<T>(
          xhr.status,
          xhr.responseText || "",
        );
        resolve(result);
      } catch (err: unknown) {
        reject(err);
      }
    };

    xhr.ontimeout = () => {
      reject(
        new ApiClientError(
          "The request took too long. Please try again.",
          0,
          "TIMEOUT",
        ),
      );
    };

    xhr.onerror = () => {
      reject(
        new ApiClientError(
          "Could not connect. Check that your phone and computer are on the same Wi-Fi.",
          0,
          "NETWORK_ERROR",
        ),
      );
    };

    xhr.onabort = () => {
      reject(
        new ApiClientError(
          "The request was canceled. Please try again.",
          0,
          "ABORTED",
        ),
      );
    };

    xhr.send(formData);
  });
}

function assertValidBaseUrl(rawUrl: string): string {
  const clean = normalizeServerUrl(rawUrl);
  if (!clean) {
    throw new ApiClientError(
      "Connection address is empty.",
      0,
      "INVALID_URL",
    );
  }
  return clean;
}

export async function checkHealth(serverUrl: string): Promise<HealthResponse> {
  const baseUrl = assertValidBaseUrl(serverUrl);
  const response = await fetchWithTimeout(
    `${baseUrl}/health`,
    { method: "GET", headers: { Accept: "application/json" } },
    SHORT_TIMEOUT_MS,
  );
  const text = await response.text();
  return parseRawResponseOrThrow<HealthResponse>(response.status, text);
}

export interface AnalyzeMeta {
  mimeType?: string;
  fileName?: string;
}

interface NativeFilePart {
  uri: string;
  name: string;
  type: string;
}

export async function analyzeVehiclePhoto(
  serverUrl: string,
  imageUri: string,
  meta?: AnalyzeMeta,
): Promise<AnalyzeResponse> {
  const baseUrl = assertValidBaseUrl(serverUrl);
  const endpoint = `${baseUrl}/api/v1/analyze?top_k_similar=${DEFAULT_TOP_K_SIMILAR}&use_tta=true`;

  const fileName = meta?.fileName || "vehicle.jpg";
  const mimeType = meta?.mimeType || "image/jpeg";
  const formData = new FormData();

  if (Platform.OS === "web") {
    const fileRes = await fetch(imageUri);
    const blob = await fileRes.blob();
    formData.append("file", blob, fileName);

    const response = await fetchWithTimeout(
      endpoint,
      {
        method: "POST",
        headers: { Accept: "application/json" },
        body: formData,
      },
      ANALYZE_TIMEOUT_MS,
    );
    const text = await response.text();
    return parseRawResponseOrThrow<AnalyzeResponse>(response.status, text);
  }

  const nativeFile: NativeFilePart = {
    uri: imageUri,
    name: fileName,
    type: mimeType,
  };
  formData.append("file", nativeFile as unknown as Blob);
  return uploadFormDataNative<AnalyzeResponse>(
    endpoint,
    formData,
    ANALYZE_TIMEOUT_MS,
  );
}
