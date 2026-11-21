import { Feather } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import React, { useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { SimilarCarsStrip } from "../components/SimilarCarsStrip";
import { VehicleCard } from "../components/VehicleCard";
import { VehiclePhotoFrame } from "../components/VehiclePhotoFrame";
import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type {
  AnalyzeResponse,
  HealthResponse,
  ScanHistoryItem,
} from "../types/api";
import {
  formatCompactCount,
  resolveVehicleModelSelection,
} from "../utils/formatters";

interface ScanScreenProps {
  serverUrl: string;
  serverHealth?: HealthResponse | null;
  activeImageUri: string | null;
  analyzeResult: AnalyzeResponse | null;
  selectedVehicleIndex: number;
  selectedModelIndices: Record<number, number>;
  isAnalyzing: boolean;
  errorMessage: string | null;
  isNetworkError: boolean;
  recentItems: ScanHistoryItem[];
  onSelectVehicle: (index: number) => void;
  onSelectModelCandidate: (
    vehicleIndex: number,
    candidateIndex: number,
  ) => void;
  onAnalyzePickedAsset: (
    uri: string,
    meta?: { mimeType?: string; fileName?: string },
  ) => void;
  onRetryAnalyze: () => void;
  onClearActive: () => void;
  onOpenHistoryItem: (item: ScanHistoryItem) => void;
  onOpenConnectionSheet: () => void;
}

export function ScanScreen({
  serverUrl,
  serverHealth,
  activeImageUri,
  analyzeResult,
  selectedVehicleIndex,
  selectedModelIndices,
  isAnalyzing,
  errorMessage,
  isNetworkError,
  recentItems,
  onSelectVehicle,
  onSelectModelCandidate,
  onAnalyzePickedAsset,
  onRetryAnalyze,
  onClearActive,
  onOpenHistoryItem,
  onOpenConnectionSheet,
}: ScanScreenProps): React.ReactElement {
  const [pickerError, setPickerError] = useState<string | null>(null);

  const handleCaptureCamera = async (): Promise<void> => {
    setPickerError(null);
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      setPickerError(
        "Camera permission is required to take a photo. Allow access in settings or choose from library.",
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ["images"],
      quality: 0.85,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets.length > 0) {
      const asset = result.assets[0];
      onAnalyzePickedAsset(asset.uri, {
        mimeType: asset.mimeType ?? "image/jpeg",
        fileName: asset.fileName ?? "camera_capture.jpg",
      });
    }
  };

  const handlePickLibrary = async (): Promise<void> => {
    setPickerError(null);
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setPickerError(
        "Photo library permission is required to select an image.",
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.85,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets.length > 0) {
      const asset = result.assets[0];
      onAnalyzePickedAsset(asset.uri, {
        mimeType: asset.mimeType ?? "image/jpeg",
        fileName: asset.fileName ?? "library_photo.jpg",
      });
    }
  };

  const selectedVehicle = analyzeResult
    ? analyzeResult.vehicles.find(
        (v) => v.vehicle_index === selectedVehicleIndex,
      ) ?? analyzeResult.vehicles[0]
    : null;

  const combinedError = pickerError || errorMessage;

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      {/* Minimal 52px App Bar */}
      <View style={styles.topBar}>
        <Pressable
          accessibilityRole="text"
          onLongPress={onOpenConnectionSheet}
          delayLongPress={650}
          style={styles.brandRow}
        >
          <View style={styles.reticleIcon} accessibilityElementsHidden>
            <View style={[styles.reticleCorner, styles.rcTL]} />
            <View style={[styles.reticleCorner, styles.rcTR]} />
            <View style={[styles.reticleCorner, styles.rcBL]} />
            <View style={[styles.reticleCorner, styles.rcBR]} />
          </View>
          <Text style={styles.brandTitle}>AutoVision</Text>
        </Pressable>

        {activeImageUri && !isAnalyzing ? (
          <View style={styles.topBarActions}>
            <Pressable
              accessibilityRole="button"
              onPress={() => {
                setPickerError(null);
                onClearActive();
              }}
              style={({ pressed }) => [
                styles.newPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Text style={styles.newPillText}>New</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Server connection"
              onPress={onOpenConnectionSheet}
              style={({ pressed }) => [
                styles.connBtn,
                pressed && styles.pressedFade,
              ]}
            >
              <Feather name="sliders" size={15} color={palette.ink} />
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: serverHealth?.ready
                      ? palette.accent
                      : isNetworkError
                        ? palette.negativeInk
                        : palette.faint,
                  },
                ]}
              />
            </Pressable>
          </View>
        ) : (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Server connection"
            onPress={onOpenConnectionSheet}
            style={({ pressed }) => [
              styles.connBtn,
              pressed && styles.pressedFade,
            ]}
          >
            <Feather name="sliders" size={15} color={palette.ink} />
            <View
              style={[
                styles.statusDot,
                {
                  backgroundColor: serverHealth?.ready
                    ? palette.accent
                    : isNetworkError
                      ? palette.negativeInk
                      : palette.faint,
                },
              ]}
            />
          </Pressable>
        )}
      </View>

      {/* Idle State */}
      {!activeImageUri && !isAnalyzing ? (
        <View style={styles.idleBlock}>
          <View style={styles.viewfinderCard}>
            <View style={styles.frameCorners} accessibilityElementsHidden>
              <View style={[styles.frameCorner, styles.fcTL]} />
              <View style={[styles.frameCorner, styles.fcTR]} />
              <View style={[styles.frameCorner, styles.fcBL]} />
              <View style={[styles.frameCorner, styles.fcBR]} />
            </View>

            <View style={styles.crosshairCenter}>
              <Feather name="crosshair" size={28} color={palette.accent} />
            </View>

            <Text style={styles.heroHeadline}>Identify any car</Text>
            <Text style={styles.heroSub}>
              Take a photo or choose one from your library.
            </Text>

            <View style={styles.capRow}>
              <View style={styles.capPill}>
                <Text style={[styles.capText, typography.tabular]}>
                  {`${serverHealth?.num_prototype_classes ?? 784} models`}
                </Text>
              </View>
              <View style={styles.capPill}>
                <Text style={[styles.capText, typography.tabular]}>
                  1950–2026
                </Text>
              </View>
              <View style={styles.capPill}>
                <Text style={[styles.capText, typography.tabular]}>
                  {`${serverHealth?.num_colors ?? 12} colors`}
                </Text>
              </View>
              {serverHealth?.gallery_size ? (
                <View style={styles.capPill}>
                  <Text style={[styles.capText, typography.tabular]}>
                    {`${formatCompactCount(serverHealth.gallery_size)} refs`}
                  </Text>
                </View>
              ) : null}
            </View>
          </View>

          <View style={styles.ctaStack}>
            <Pressable
              accessibilityRole="button"
              onPress={() => {
                void handleCaptureCamera();
              }}
              style={({ pressed }) => [
                styles.primaryPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Feather name="camera" size={18} color={palette.primaryInk} />
              <Text style={styles.primaryPillText}>Take photo</Text>
            </Pressable>

            <Pressable
              accessibilityRole="button"
              onPress={() => {
                void handlePickLibrary();
              }}
              style={({ pressed }) => [
                styles.secondaryPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Feather name="image" size={18} color={palette.ink} />
              <Text style={styles.secondaryPillText}>
                Choose from library
              </Text>
            </Pressable>
          </View>

          {pickerError ? (
            <View style={styles.errorCard}>
              <View style={styles.errorTag}>
                <Text style={styles.errorTagText}>Permission needed</Text>
              </View>
              <Text style={styles.errorBody}>{pickerError}</Text>
            </View>
          ) : null}

          {recentItems.length > 0 ? (
            <View style={styles.recentSection}>
              <Text style={typography.section}>Recent</Text>
              <FlatList
                horizontal
                data={recentItems.slice(0, 8)}
                keyExtractor={(item) => item.id}
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.recentList}
                renderItem={({ item }) => {
                  const primary =
                    item.response.vehicles[
                      item.response.primary_vehicle_index
                    ] ?? item.response.vehicles[0];
                  const selection = primary
                    ? resolveVehicleModelSelection(
                        primary,
                        item.selectedModelIndices?.[primary.vehicle_index],
                      )
                    : null;
                  const year = primary?.attributes.year.predicted_rounded;
                  return (
                    <Pressable
                      accessibilityRole="button"
                      onPress={() => onOpenHistoryItem(item)}
                      style={({ pressed }) => [
                        styles.recentCard,
                        pressed && styles.pressedFade,
                      ]}
                    >
                      <Image
                        source={{ uri: item.imageUri }}
                        style={styles.recentThumb}
                        resizeMode="cover"
                      />
                      <View style={styles.recentMeta}>
                        <Text style={styles.recentTitle} numberOfLines={1}>
                          {selection
                            ? `${selection.makeDisplay} ${selection.modelTitle}`.trim()
                            : "Vehicle"}
                        </Text>
                        {year ? (
                          <Text
                            style={[styles.recentSub, typography.tabular]}
                            numberOfLines={1}
                          >
                            {year}
                          </Text>
                        ) : null}
                      </View>
                    </Pressable>
                  );
                }}
              />
            </View>
          ) : null}
        </View>
      ) : null}

      {/* Active Photo Preview */}
      {activeImageUri ? (
        <VehiclePhotoFrame
          imageUri={activeImageUri}
          imageSize={analyzeResult?.image_size ?? null}
          vehicles={analyzeResult?.vehicles ?? []}
          selectedIndex={selectedVehicleIndex}
          selectedModelIndices={selectedModelIndices}
          onSelectVehicle={onSelectVehicle}
        />
      ) : null}

      {/* Loading State */}
      {isAnalyzing ? (
        <View style={styles.loadingCard}>
          <ActivityIndicator size="small" color={palette.accent} />
          <View style={styles.loadingTextCol}>
            <Text style={styles.loadingText}>Analyzing vehicle...</Text>
            <Text style={styles.loadingSubText}>
              Detecting car, model, year & paint color
            </Text>
          </View>
        </View>
      ) : null}

      {/* Error State */}
      {!isAnalyzing && combinedError && activeImageUri ? (
        <View style={styles.errorCard}>
          <View style={styles.errorTag}>
            <Text style={styles.errorTagText}>
              {isNetworkError ? "Connection error" : "Could not identify"}
            </Text>
          </View>
          <Text style={styles.errorBody}>{combinedError}</Text>
          <View style={styles.errorActions}>
            <Pressable
              accessibilityRole="button"
              onPress={onRetryAnalyze}
              style={({ pressed }) => [
                styles.retryPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Text style={styles.retryPillText}>Try again</Text>
            </Pressable>

            {isNetworkError ? (
              <Pressable
                accessibilityRole="button"
                onPress={onOpenConnectionSheet}
                style={({ pressed }) => [
                  styles.addressPill,
                  pressed && styles.pressedFade,
                ]}
              >
                <Text style={styles.addressPillText}>Change address</Text>
              </Pressable>
            ) : null}
          </View>
        </View>
      ) : null}

      {/* Result State */}
      {!isAnalyzing && analyzeResult ? (
        selectedVehicle ? (
          <>
            <VehicleCard
              vehicle={selectedVehicle}
              modelVersion={analyzeResult.model_version}
              timingsMs={analyzeResult.timings_ms.total_ms}
              selectedCandidateIndex={
                selectedModelIndices[selectedVehicle.vehicle_index] ?? 0
              }
              onSelectCandidate={(idx) =>
                onSelectModelCandidate(selectedVehicle.vehicle_index, idx)
              }
            />

            <SimilarCarsStrip
              similar={selectedVehicle.similar_vehicles}
              serverUrl={serverUrl}
              activeCanonicalClass={
                resolveVehicleModelSelection(
                  selectedVehicle,
                  selectedModelIndices[selectedVehicle.vehicle_index],
                ).canonicalClass
              }
            />

            <View style={styles.bottomCtaRow}>
              <Pressable
                accessibilityRole="button"
                onPress={() => {
                  void handleCaptureCamera();
                }}
                style={({ pressed }) => [
                  styles.bottomPrimaryPill,
                  pressed && styles.pressedFade,
                ]}
              >
                <Feather name="camera" size={16} color={palette.primaryInk} />
                <Text style={styles.primaryPillText}>Take another photo</Text>
              </Pressable>

              <Pressable
                accessibilityRole="button"
                onPress={() => {
                  void handlePickLibrary();
                }}
                style={({ pressed }) => [
                  styles.bottomSecondaryPill,
                  pressed && styles.pressedFade,
                ]}
              >
                <Feather name="image" size={16} color={palette.ink} />
                <Text style={styles.secondaryPillText}>Library</Text>
              </Pressable>
            </View>
          </>
        ) : (
          <View style={styles.errorCard}>
            <Text style={typography.section}>No car found in photo</Text>
            <Text style={styles.errorBody}>
              Try taking a clearer photo showing the car cleanly in frame.
            </Text>
            <Pressable
              accessibilityRole="button"
              onPress={() => {
                void handleCaptureCamera();
              }}
              style={({ pressed }) => [
                styles.retryPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Text style={styles.retryPillText}>Take another photo</Text>
            </Pressable>
          </View>
        )
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: palette.ground,
  },
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xl,
    gap: spacing.md,
  },
  topBar: {
    height: 52,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  brandRow: {
    flex: 1,
    minWidth: 0,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  reticleIcon: {
    width: 22,
    height: 22,
    alignItems: "center",
    justifyContent: "center",
  },
  reticleCorner: {
    position: "absolute",
    width: 7,
    height: 7,
    borderColor: palette.accent,
  },
  rcTL: {
    top: 1,
    left: 1,
    borderTopWidth: 2,
    borderLeftWidth: 2,
  },
  rcTR: {
    top: 1,
    right: 1,
    borderTopWidth: 2,
    borderRightWidth: 2,
  },
  rcBL: {
    bottom: 1,
    left: 1,
    borderBottomWidth: 2,
    borderLeftWidth: 2,
  },
  rcBR: {
    bottom: 1,
    right: 1,
    borderBottomWidth: 2,
    borderRightWidth: 2,
  },
  brandTitle: {
    fontFamily: fonts.semibold,
    fontSize: 20,
    lineHeight: 26,
    letterSpacing: -0.2,
    paddingRight: 10,
    includeFontPadding: false,
    color: palette.ink,
  },
  topBarActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  connBtn: {
    width: 36,
    height: 36,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  statusDot: {
    position: "absolute",
    top: 6,
    right: 6,
    width: 7,
    height: 7,
    borderRadius: radius.pill,
  },
  newPill: {
    backgroundColor: palette.raised,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  newPillText: {
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
    color: palette.ink,
  },
  idleBlock: {
    gap: spacing.md,
  },
  viewfinderCard: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    paddingVertical: 44,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    position: "relative",
  },
  frameCorners: {
    position: "absolute",
    top: 18,
    bottom: 18,
    left: 18,
    right: 18,
  },
  frameCorner: {
    position: "absolute",
    width: 18,
    height: 18,
    borderColor: palette.control,
  },
  fcTL: {
    top: 0,
    left: 0,
    borderTopWidth: 1.5,
    borderLeftWidth: 1.5,
  },
  fcTR: {
    top: 0,
    right: 0,
    borderTopWidth: 1.5,
    borderRightWidth: 1.5,
  },
  fcBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 1.5,
    borderLeftWidth: 1.5,
  },
  fcBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 1.5,
    borderRightWidth: 1.5,
  },
  crosshairCenter: {
    width: 56,
    height: 56,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  heroHeadline: {
    ...typography.title,
    textAlign: "center",
  },
  heroSub: {
    ...typography.body,
    color: palette.muted,
    textAlign: "center",
  },
  capRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 6,
    marginTop: 6,
  },
  capPill: {
    backgroundColor: palette.raised,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  capText: {
    fontFamily: fonts.medium,
    fontSize: 11,
    lineHeight: 15,
    color: palette.muted,
  },
  ctaStack: {
    gap: 10,
  },
  primaryPill: {
    height: 52,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  primaryPillText: {
    fontFamily: fonts.semibold,
    fontSize: 15,
    lineHeight: 20,
    color: palette.primaryInk,
  },
  secondaryPill: {
    height: 48,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  secondaryPillText: {
    fontFamily: fonts.medium,
    fontSize: 15,
    lineHeight: 20,
    color: palette.ink,
  },
  pressedFade: {
    opacity: 0.78,
  },
  recentSection: {
    marginTop: spacing.sm,
    gap: spacing.md,
  },
  recentList: {
    gap: 12,
  },
  recentCard: {
    width: 136,
    backgroundColor: palette.surface,
    borderRadius: 16,
    overflow: "hidden",
  },
  recentThumb: {
    width: "100%",
    height: 84,
    backgroundColor: palette.raised,
  },
  recentMeta: {
    padding: 10,
    gap: 2,
  },
  recentTitle: {
    fontFamily: fonts.semibold,
    fontSize: 13,
    lineHeight: 18,
    color: palette.ink,
  },
  recentSub: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  loadingCard: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  loadingTextCol: {
    flex: 1,
    gap: 2,
  },
  loadingText: {
    fontFamily: fonts.medium,
    fontSize: 15,
    lineHeight: 20,
    color: palette.ink,
  },
  loadingSubText: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  errorCard: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    gap: 12,
  },
  errorTag: {
    alignSelf: "flex-start",
    backgroundColor: palette.negativeBg,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  errorTagText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.negativeInk,
  },
  errorBody: {
    ...typography.body,
    color: palette.muted,
  },
  errorActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 2,
  },
  retryPill: {
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
  },
  retryPillText: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    color: palette.primaryInk,
  },
  addressPill: {
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
  },
  addressPillText: {
    fontFamily: fonts.medium,
    fontSize: 14,
    color: palette.ink,
  },
  bottomCtaRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: spacing.xs,
  },
  bottomPrimaryPill: {
    flex: 1,
    height: 48,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingHorizontal: spacing.md,
  },
  bottomSecondaryPill: {
    height: 48,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingHorizontal: spacing.lg,
  },
});
