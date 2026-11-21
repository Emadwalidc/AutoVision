import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  useFonts,
} from "@expo-google-fonts/inter";
import * as Haptics from "expo-haptics";
import { StatusBar } from "expo-status-bar";
import React, { useCallback, useEffect, useState } from "react";
import { StyleSheet, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { BottomTabBar } from "./src/components/BottomTabBar";
import { ConnectionSheet } from "./src/components/ConnectionSheet";
import { ErrorBoundary } from "./src/components/ErrorBoundary";
import { GarageScreen } from "./src/screens/GarageScreen";
import { ScanScreen } from "./src/screens/ScanScreen";
import {
  analyzeVehiclePhoto,
  ApiClientError,
  checkHealth,
} from "./src/services/apiClient";
import {
  appendHistoryItem,
  clearHistory,
  loadHistory,
  loadServerUrl,
  removeHistoryItem,
  saveServerUrl,
  updateHistorySelection,
} from "./src/services/storage";
import { palette } from "./src/theme/tokens";
import type {
  AnalyzeResponse,
  HealthResponse,
  ScanHistoryItem,
  TabId,
} from "./src/types/api";
import { DEFAULT_SERVER_URL } from "./src/utils/formatters";

interface AssetMeta {
  mimeType?: string;
  fileName?: string;
}

export default function App(): React.ReactElement {
  const [fontsLoaded, fontError] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
  });

  const [activeTab, setActiveTab] = useState<TabId>("scan");
  const [serverUrl, setServerUrl] = useState<string>(DEFAULT_SERVER_URL);
  const [serverHealth, setServerHealth] = useState<HealthResponse | null>(null);
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);

  const [activeImageUri, setActiveImageUri] = useState<string | null>(null);
  const [activeMeta, setActiveMeta] = useState<AssetMeta>({});
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResponse | null>(
    null,
  );
  const [selectedVehicleIndex, setSelectedVehicleIndex] = useState<number>(0);
  const [selectedModelIndices, setSelectedModelIndices] = useState<
    Record<number, number>
  >({});
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isNetworkError, setIsNetworkError] = useState<boolean>(false);
  const [connectionOpen, setConnectionOpen] = useState<boolean>(false);

  // Hydrate saved LAN address, one-shot server health, and recent history; zero idle timers.
  useEffect(() => {
    let active = true;
    void (async () => {
      const [loadedUrl, loadedHistory] = await Promise.all([
        loadServerUrl(),
        loadHistory(),
      ]);
      if (!active) {
        return;
      }
      setServerUrl(loadedUrl);
      setHistory(loadedHistory);

      try {
        const health = await checkHealth(loadedUrl);
        if (active) {
          setServerHealth(health);
        }
      } catch {
        if (active) {
          setServerHealth(null);
        }
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  const executeAnalyze = useCallback(
    async (imageUri: string, meta: AssetMeta, targetUrl: string): Promise<void> => {
      setActiveImageUri(imageUri);
      setActiveMeta(meta);
      setActiveHistoryId(null);
      setIsAnalyzing(true);
      setErrorMessage(null);
      setIsNetworkError(false);
      setAnalyzeResult(null);
      setSelectedVehicleIndex(0);
      setSelectedModelIndices({});

      try {
        const response = await analyzeVehiclePhoto(targetUrl, imageUri, meta);
        setAnalyzeResult(response);
        setSelectedVehicleIndex(response.primary_vehicle_index ?? 0);
        Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Success,
        ).catch(() => {});

        const entry: ScanHistoryItem = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          createdAt: new Date().toISOString(),
          imageUri,
          response,
        };
        setActiveHistoryId(entry.id);
        setHistory((prev) => {
          void appendHistoryItem(entry, prev).catch(() => {});
          return [entry, ...prev].slice(0, 30);
        });
      } catch (err: unknown) {
        const apiErr = err instanceof ApiClientError ? err : null;
        Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Error,
        ).catch(() => {});
        setErrorMessage(
          apiErr?.message || "Something went wrong. Please try again.",
        );
        setIsNetworkError(
          apiErr !== null &&
            (apiErr.errorCode === "NETWORK_ERROR" ||
              apiErr.errorCode === "TIMEOUT" ||
              apiErr.errorCode === "INVALID_URL"),
        );
      } finally {
        setIsAnalyzing(false);
      }
    },
    [],
  );

  const handleAnalyzePickedAsset = (uri: string, meta?: AssetMeta): void => {
    void executeAnalyze(uri, meta ?? {}, serverUrl);
  };

  const handleRetryAnalyze = (): void => {
    if (!activeImageUri) {
      return;
    }
    void executeAnalyze(activeImageUri, activeMeta, serverUrl);
  };

  const handleClearActive = (): void => {
    setActiveImageUri(null);
    setActiveMeta({});
    setActiveHistoryId(null);
    setAnalyzeResult(null);
    setSelectedVehicleIndex(0);
    setSelectedModelIndices({});
    setErrorMessage(null);
    setIsNetworkError(false);
  };

  const handleOpenHistoryItem = (item: ScanHistoryItem): void => {
    setActiveImageUri(item.imageUri);
    setActiveHistoryId(item.id);
    setAnalyzeResult(item.response);
    setSelectedVehicleIndex(item.response.primary_vehicle_index ?? 0);
    setSelectedModelIndices(item.selectedModelIndices ?? {});
    setErrorMessage(null);
    setIsNetworkError(false);
    setActiveTab("scan");
  };

  const handleSelectModelCandidate = (
    vehicleIndex: number,
    candidateIndex: number,
  ): void => {
    const nextIndices: Record<number, number> = {
      ...selectedModelIndices,
      [vehicleIndex]: candidateIndex,
    };
    setSelectedModelIndices(nextIndices);
    if (activeHistoryId) {
      setHistory((prev) => {
        void updateHistorySelection(activeHistoryId, nextIndices, prev).catch(
          () => {},
        );
        return prev.map((item) =>
          item.id === activeHistoryId
            ? { ...item, selectedModelIndices: nextIndices }
            : item,
        );
      });
    }
  };

  const handleClearAllHistory = async (): Promise<void> => {
    await clearHistory();
    setActiveHistoryId(null);
    setHistory([]);
  };

  const handleDeleteHistoryItem = (id: string): void => {
    if (activeHistoryId === id) {
      setActiveHistoryId(null);
    }
    setHistory((prev) => {
      void removeHistoryItem(id, prev).catch(() => {});
      return prev.filter((item) => item.id !== id);
    });
  };

  const handleSaveServerUrl = async (nextUrl: string): Promise<void> => {
    const saved = await saveServerUrl(nextUrl);
    setServerUrl(saved);
  };

  if (!fontsLoaded && !fontError) {
    return (
      <SafeAreaProvider>
        <StatusBar style="light" />
        <View style={styles.root} />
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <ErrorBoundary>
        <SafeAreaView style={styles.root} edges={["top", "left", "right"]}>
          <View style={styles.body}>
            <View
              style={[styles.pane, activeTab !== "scan" && styles.hiddenPane]}
            >
              <ScanScreen
                serverUrl={serverUrl}
                serverHealth={serverHealth}
                activeImageUri={activeImageUri}
                analyzeResult={analyzeResult}
                selectedVehicleIndex={selectedVehicleIndex}
                selectedModelIndices={selectedModelIndices}
                isAnalyzing={isAnalyzing}
                errorMessage={errorMessage}
                isNetworkError={isNetworkError}
                recentItems={history}
                onSelectVehicle={setSelectedVehicleIndex}
                onSelectModelCandidate={handleSelectModelCandidate}
                onAnalyzePickedAsset={handleAnalyzePickedAsset}
                onRetryAnalyze={handleRetryAnalyze}
                onClearActive={handleClearActive}
                onOpenHistoryItem={handleOpenHistoryItem}
                onOpenConnectionSheet={() => setConnectionOpen(true)}
              />
            </View>

            {activeTab === "garage" ? (
              <View style={styles.pane}>
                <GarageScreen
                  items={history}
                  onOpenItem={handleOpenHistoryItem}
                  onDeleteItem={handleDeleteHistoryItem}
                  onClearHistory={handleClearAllHistory}
                  onGoScan={() => setActiveTab("scan")}
                />
              </View>
            ) : null}
          </View>

          <BottomTabBar
            activeTab={activeTab}
            onSelectTab={setActiveTab}
            garageCount={history.length}
          />

          <ConnectionSheet
            visible={connectionOpen}
            serverUrl={serverUrl}
            onSave={handleSaveServerUrl}
            onHealthChecked={setServerHealth}
            onClose={() => setConnectionOpen(false)}
          />
        </SafeAreaView>
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: palette.ground,
  },
  body: {
    flex: 1,
  },
  pane: {
    flex: 1,
  },
  hiddenPane: {
    display: "none",
  },
});
