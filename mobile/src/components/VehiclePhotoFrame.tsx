import * as Haptics from "expo-haptics";
import React, { useMemo, useState } from "react";
import {
  Image,
  type ImageProps,
  type LayoutChangeEvent,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { ImageSize, VehicleDetectionResult } from "../types/api";
import {
  computeContainedImageRect,
  resolveVehicleModelSelection,
} from "../utils/formatters";

interface VehiclePhotoFrameProps {
  imageUri: string;
  imageSize?: ImageSize | null;
  vehicles: VehicleDetectionResult[];
  selectedIndex: number;
  selectedModelIndices?: Record<number, number>;
  onSelectVehicle: (vehicleIndex: number) => void;
}

/**
 * Displays the captured photograph inside an OLED surface card with
 * cleanly projected viewfinder corner brackets around each detected car.
 */
export function VehiclePhotoFrame({
  imageUri,
  imageSize,
  vehicles,
  selectedIndex,
  selectedModelIndices,
  onSelectVehicle,
}: VehiclePhotoFrameProps): React.ReactElement {
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [naturalSize, setNaturalSize] = useState<ImageSize>({
    width: 0,
    height: 0,
  });

  const effectiveSize =
    imageSize && imageSize.width > 0 && imageSize.height > 0
      ? imageSize
      : naturalSize;

  const containedRect = useMemo(
    () =>
      computeContainedImageRect(
        viewport.width,
        viewport.height,
        effectiveSize.width,
        effectiveSize.height,
      ),
    [viewport.width, viewport.height, effectiveSize.width, effectiveSize.height],
  );

  const handleViewportLayout = (event: LayoutChangeEvent): void => {
    const { width, height } = event.nativeEvent.layout;
    if (width !== viewport.width || height !== viewport.height) {
      setViewport({ width, height });
    }
  };

  const handleImageLoad: NonNullable<ImageProps["onLoad"]> = (event) => {
    const source = event.nativeEvent?.source;
    if (source && source.width > 0 && source.height > 0) {
      setNaturalSize({ width: source.width, height: source.height });
    }
  };

  const handleTapBox = (vehicleIndex: number): void => {
    Haptics.selectionAsync().catch(() => {});
    onSelectVehicle(vehicleIndex);
  };

  const showMultiple = vehicles.length > 1;

  return (
    <View style={styles.wrap}>
      <View style={styles.viewport} onLayout={handleViewportLayout}>
        <Image
          source={{ uri: imageUri }}
          style={StyleSheet.absoluteFill}
          resizeMode="contain"
          onLoad={handleImageLoad}
          accessibilityLabel="Vehicle photo"
        />

        {containedRect.width > 0 &&
        containedRect.height > 0 &&
        vehicles.length > 0 ? (
          <View
            pointerEvents="box-none"
            style={[
              styles.overlayFrame,
              {
                left: containedRect.left,
                top: containedRect.top,
                width: containedRect.width,
                height: containedRect.height,
              },
            ]}
          >
            {vehicles.map((vehicle) => {
              const isSelected = vehicle.vehicle_index === selectedIndex;
              const boxLeft = vehicle.bbox_norm.x1 * containedRect.width;
              const boxTop = vehicle.bbox_norm.y1 * containedRect.height;
              const boxWidth = Math.max(
                24,
                (vehicle.bbox_norm.x2 - vehicle.bbox_norm.x1) *
                  containedRect.width,
              );
              const boxHeight = Math.max(
                24,
                (vehicle.bbox_norm.y2 - vehicle.bbox_norm.y1) *
                  containedRect.height,
              );

              const cornerStyle = isSelected
                ? styles.cornerSelected
                : styles.cornerIdle;

              const selection = resolveVehicleModelSelection(
                vehicle,
                selectedModelIndices?.[vehicle.vehicle_index],
              );

              return (
                <Pressable
                  key={vehicle.vehicle_index}
                  hitSlop={8}
                  disabled={!showMultiple}
                  accessibilityRole={showMultiple ? "button" : "image"}
                  accessibilityLabel={`${selection.makeDisplay} ${selection.modelTitle}`.trim()}
                  onPress={() => handleTapBox(vehicle.vehicle_index)}
                  style={[
                    styles.bbox,
                    {
                      left: boxLeft,
                      top: boxTop,
                      width: boxWidth,
                      height: boxHeight,
                    },
                  ]}
                >
                  <View style={[styles.corner, styles.tl, cornerStyle]} />
                  <View style={[styles.corner, styles.tr, cornerStyle]} />
                  <View style={[styles.corner, styles.bl, cornerStyle]} />
                  <View style={[styles.corner, styles.br, cornerStyle]} />

                  {showMultiple ? (
                    <View
                      style={[
                        styles.numberBadge,
                        isSelected
                          ? styles.numberBadgeActive
                          : styles.numberBadgeIdle,
                      ]}
                    >
                      <Text
                        style={[
                          styles.numberBadgeText,
                          typography.tabular,
                          isSelected
                            ? styles.numberBadgeTextActive
                            : styles.numberBadgeTextIdle,
                        ]}
                      >
                        {vehicle.vehicle_index + 1}
                      </Text>
                    </View>
                  ) : null}
                </Pressable>
              );
            })}
          </View>
        ) : null}
      </View>

      {showMultiple ? (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.switcherRow}
        >
          {vehicles.map((vehicle) => {
            const active = vehicle.vehicle_index === selectedIndex;
            const selection = resolveVehicleModelSelection(
              vehicle,
              selectedModelIndices?.[vehicle.vehicle_index],
            );
            return (
              <Pressable
                key={vehicle.vehicle_index}
                accessibilityRole="button"
                onPress={() => handleTapBox(vehicle.vehicle_index)}
                style={[
                  styles.carChip,
                  active ? styles.carChipActive : styles.carChipIdle,
                ]}
              >
                <Text
                  style={[
                    styles.carChipText,
                    typography.tabular,
                    active ? styles.carChipTextActive : styles.carChipTextIdle,
                  ]}
                >
                  {`Car ${vehicle.vehicle_index + 1} \u00B7 ${selection.makeDisplay} ${selection.modelTitle}`}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    gap: spacing.sm + 2,
  },
  viewport: {
    width: "100%",
    height: 280,
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    overflow: "hidden",
    position: "relative",
  },
  overlayFrame: {
    position: "absolute",
  },
  bbox: {
    position: "absolute",
  },
  corner: {
    position: "absolute",
    width: 14,
    height: 14,
  },
  cornerSelected: {
    borderColor: palette.accent,
  },
  cornerIdle: {
    borderColor: palette.ink,
  },
  tl: {
    top: 0,
    left: 0,
    borderTopWidth: 2.5,
    borderLeftWidth: 2.5,
  },
  tr: {
    top: 0,
    right: 0,
    borderTopWidth: 2.5,
    borderRightWidth: 2.5,
  },
  bl: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 2.5,
    borderLeftWidth: 2.5,
  },
  br: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 2.5,
    borderRightWidth: 2.5,
  },
  numberBadge: {
    position: "absolute",
    top: 6,
    left: 6,
    minWidth: 22,
    height: 22,
    borderRadius: radius.pill,
    paddingHorizontal: 6,
    alignItems: "center",
    justifyContent: "center",
  },
  numberBadgeActive: {
    backgroundColor: palette.primaryBg,
  },
  numberBadgeIdle: {
    backgroundColor: palette.surface,
  },
  numberBadgeText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
  },
  numberBadgeTextActive: {
    color: palette.primaryInk,
  },
  numberBadgeTextIdle: {
    color: palette.ink,
  },
  switcherRow: {
    gap: spacing.sm,
  },
  carChip: {
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md + 2,
    paddingVertical: 8,
  },
  carChipActive: {
    backgroundColor: palette.primaryBg,
  },
  carChipIdle: {
    backgroundColor: palette.raised,
  },
  carChipText: {
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
  },
  carChipTextActive: {
    color: palette.primaryInk,
  },
  carChipTextIdle: {
    color: palette.ink,
  },
});
