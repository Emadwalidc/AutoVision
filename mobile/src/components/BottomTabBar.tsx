import { Feather } from "@expo/vector-icons";
import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { fonts, palette, spacing, typography } from "../theme/tokens";
import type { TabId } from "../types/api";

interface BottomTabBarProps {
  activeTab: TabId;
  onSelectTab: (tab: TabId) => void;
  garageCount: number;
}

export function BottomTabBar({
  activeTab,
  onSelectTab,
  garageCount,
}: BottomTabBarProps): React.ReactElement {
  const insets = useSafeAreaInsets();
  const bottomPad = Math.max(insets.bottom, spacing.xs);

  const scanActive = activeTab === "scan";
  const garageActive = activeTab === "garage";
  const garageLabel =
    garageCount > 0 ? `Garage (${garageCount})` : "Garage";

  return (
    <View style={[styles.container, { paddingBottom: bottomPad }]}>
      <Pressable
        accessibilityRole="tab"
        accessibilityState={{ selected: scanActive }}
        accessibilityLabel="Scan"
        onPress={() => onSelectTab("scan")}
        style={({ pressed }) => [styles.tab, pressed && styles.tabPressed]}
      >
        <Feather
          name="camera"
          size={19}
          color={scanActive ? palette.ink : palette.muted}
        />
        <Text
          style={[
            styles.label,
            scanActive ? styles.labelActive : styles.labelIdle,
          ]}
        >
          Scan
        </Text>
      </Pressable>

      <Pressable
        accessibilityRole="tab"
        accessibilityState={{ selected: garageActive }}
        accessibilityLabel={garageLabel}
        onPress={() => onSelectTab("garage")}
        style={({ pressed }) => [styles.tab, pressed && styles.tabPressed]}
      >
        <Feather
          name="bookmark"
          size={19}
          color={garageActive ? palette.ink : palette.muted}
        />
        <Text
          style={[
            styles.label,
            typography.tabular,
            garageActive ? styles.labelActive : styles.labelIdle,
          ]}
        >
          {garageLabel}
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    backgroundColor: palette.ground,
    borderTopWidth: 1,
    borderTopColor: palette.border,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 10,
    paddingBottom: 6,
    gap: 4,
    minHeight: 54,
  },
  tabPressed: {
    opacity: 0.72,
  },
  label: {
    fontSize: 12,
    lineHeight: 16,
  },
  labelActive: {
    fontFamily: fonts.semibold,
    color: palette.ink,
  },
  labelIdle: {
    fontFamily: fonts.medium,
    color: palette.muted,
  },
});
