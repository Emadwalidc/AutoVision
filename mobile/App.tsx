import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  useFonts,
} from "@expo-google-fonts/inter";
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
} from "./src/services/apiClient";
import {
  appendHistoryItem,
  clearHistory,
  loadHistory,
  loadServerUrl,
  saveServerUrl,
} from "./src/services/storage";
import { palette } from "./src/theme/tokens";
import type { AnalyzeResponse, ScanHistoryItem, TabId } from "./src/types/api";
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
  const [history, setHistory] = useState<ScanHistoryItem[]>([]);

  const [activeImageUri, setActiveImageUri] = useState<string | null>(null);
  const [activeMeta, setActiveMeta] = useState<AssetMeta>({});
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResponse | null>(
    null,
  );
  const [selectedVehicleIndex, setSelectedVehicleIndex] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isNetworkError, setIsNetworkError] = useState<boolean>(false);
  const [connectionOpen, setConnectionOpen] = useState<boolean>(false);

  // Hydrate saved LAN address and recent history once on launch; zero idle timers.
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
    })();

    return () => {
      active = false;
    };
  }, []);

  const executeAnalyze = useCallback(
    async (imageUri: string, meta: AssetMeta, targetUrl: string): Promise<void> => {
      setActiveImageUri(imageUri);
      setActiveMeta(meta);
      setIsAnalyzing(true);
      setErrorMessage(null);
      setIsNetworkError(false);
      setAnalyzeResult(null);
      setSelectedVehicleIndex(0);

      try {
        const response = await analyzeVehiclePhoto(targetUrl, imageUri, meta);
        setAnalyzeResult(response);
        setSelectedVehicleIndex(response.primary_vehicle_index ?? 0);

        const entry: ScanHistoryItem = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          createdAt: new Date().toISOString(),
          imageUri,
          response,
        };
        setHistory((prev) => {
          void appendHistoryItem(entry, prev).catch(() => {});
          return [entry, ...prev].slice(0, 30);
        });
      } catch (err: unknown) {
        const apiErr = err instanceof ApiClientError ? err : null;
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
    setAnalyzeResult(null);
    setSelectedVehicleIndex(0);
    setErrorMessage(null);
    setIsNetworkError(false);
  };

  const handleOpenHistoryItem = (item: ScanHistoryItem): void => {
    setActiveImageUri(item.imageUri);
    setAnalyzeResult(item.response);
    setSelectedVehicleIndex(item.response.primary_vehicle_index ?? 0);
    setErrorMessage(null);
    setIsNetworkError(false);
    setActiveTab("scan");
  };

  const handleClearAllHistory = async (): Promise<void> => {
    await clearHistory();
    setHistory([]);
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
                activeImageUri={activeImageUri}
                analyzeResult={analyzeResult}
                selectedVehicleIndex={selectedVehicleIndex}
                isAnalyzing={isAnalyzing}
                errorMessage={errorMessage}
                isNetworkError={isNetworkError}
                recentItems={history}
                onSelectVehicle={setSelectedVehicleIndex}
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
