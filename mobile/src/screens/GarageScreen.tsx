import { Feather } from "@expo/vector-icons";
import React, { useMemo, useState } from "react";
import {
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { ScanHistoryItem } from "../types/api";
import { formatModelTitle, formatRelativeTime } from "../utils/formatters";

interface GarageScreenProps {
  items: ScanHistoryItem[];
  onOpenItem: (item: ScanHistoryItem) => void;
  onClearHistory: () => Promise<void>;
  onGoScan: () => void;
}

export function GarageScreen({
  items,
  onOpenItem,
  onClearHistory,
  onGoScan,
}: GarageScreenProps): React.ReactElement {
  const [confirmClear, setConfirmClear] = useState(false);
  const [query, setQuery] = useState("");

  const filteredItems = useMemo(() => {
    const clean = query.trim().toLowerCase();
    if (!clean) {
      return items;
    }
    return items.filter((item) => {
      const primary =
        item.response.vehicles[item.response.primary_vehicle_index] ??
        item.response.vehicles[0];
      if (!primary) {
        return false;
      }
      const haystack = [
        primary.attributes.make.display_name,
        primary.attributes.model.display_name,
        String(primary.attributes.year.predicted_rounded),
        primary.attributes.body_type.display_name,
        primary.attributes.color.display_name,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(clean);
    });
  }, [items, query]);

  const handleConfirmClear = (): void => {
    setConfirmClear(false);
    setQuery("");
    void onClearHistory();
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.titleBlock}>
          <Text style={typography.title}>Garage</Text>
          <Text style={[styles.countText, typography.tabular]}>
            {`${items.length} ${items.length === 1 ? "car" : "cars"}`}
          </Text>
        </View>

        {items.length > 0 ? (
          confirmClear ? (
            <View style={styles.confirmRow}>
              <Pressable
                accessibilityRole="button"
                onPress={handleConfirmClear}
                style={styles.confirmPill}
              >
                <Text style={styles.confirmText}>Confirm</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                onPress={() => setConfirmClear(false)}
                style={styles.cancelPill}
              >
                <Text style={styles.cancelText}>Cancel</Text>
              </Pressable>
            </View>
          ) : (
            <Pressable
              accessibilityRole="button"
              onPress={() => setConfirmClear(true)}
              style={styles.cancelPill}
            >
              <Text style={styles.cancelText}>Clear</Text>
            </Pressable>
          )
        ) : null}
      </View>

      {items.length >= 4 ? (
        <View style={styles.searchRow}>
          <View style={styles.searchPill}>
            <Feather name="search" size={15} color={palette.muted} />
            <TextInput
              value={query}
              onChangeText={setQuery}
              placeholder="Search saved cars"
              placeholderTextColor={palette.faint}
              autoCapitalize="none"
              autoCorrect={false}
              style={styles.searchInput}
            />
            {query.length > 0 ? (
              <Pressable
                accessibilityRole="button"
                onPress={() => setQuery("")}
              >
                <Feather name="x" size={15} color={palette.muted} />
              </Pressable>
            ) : null}
          </View>
        </View>
      ) : null}

      {items.length === 0 ? (
        <View style={styles.emptyWrap}>
          <View style={styles.emptyCard}>
            <View style={styles.emptyIconCircle}>
              <Feather name="bookmark" size={22} color={palette.muted} />
            </View>
            <Text style={typography.section}>No saved cars yet</Text>
            <Text style={styles.emptyBody}>
              Cars you identify appear here automatically so you can revisit them
              anytime.
            </Text>
            <Pressable
              accessibilityRole="button"
              onPress={onGoScan}
              style={({ pressed }) => [
                styles.primaryPill,
                pressed && styles.pressedFade,
              ]}
            >
              <Text style={styles.primaryPillText}>Scan a car</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <FlatList
          data={filteredItems}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          ListEmptyComponent={
            <View style={styles.emptyCard}>
              <Text style={typography.small}>No saved cars match your search.</Text>
            </View>
          }
          renderItem={({ item }) => {
            const primary =
              item.response.vehicles[item.response.primary_vehicle_index] ??
              item.response.vehicles[0];
            const makeName = primary?.attributes.make.display_name ?? "";
            const cleanModel = formatModelTitle(
              makeName,
              primary?.attributes.model.display_name ?? "Vehicle",
            );
            const yearRounded =
              primary?.attributes.year.predicted_rounded ?? null;
            const bodyName = primary?.attributes.body_type.display_name ?? "";
            const colorObj = primary?.attributes.color;
            const carCount = item.response.num_vehicles;

            return (
              <Pressable
                accessibilityRole="button"
                onPress={() => onOpenItem(item)}
                style={({ pressed }) => [
                  styles.card,
                  pressed && styles.pressedFade,
                ]}
              >
                <Image
                  source={{ uri: item.imageUri }}
                  style={styles.thumb}
                  resizeMode="cover"
                />

                <View style={styles.cardBody}>
                  <Text style={styles.cardTitle} numberOfLines={1}>
                    {`${makeName} ${cleanModel}`.trim()}
                  </Text>

                  <View style={styles.specLine}>
                    {colorObj ? (
                      <View
                        style={[
                          styles.swatch,
                          { backgroundColor: colorObj.hex || palette.raised },
                        ]}
                      />
                    ) : null}
                    <Text style={styles.specText} numberOfLines={1}>
                      {[
                        yearRounded !== null ? String(yearRounded) : null,
                        bodyName || null,
                        colorObj?.display_name ?? null,
                      ]
                        .filter(Boolean)
                        .join(" \u00B7 ")}
                    </Text>
                  </View>

                  <Text style={[styles.timeText, typography.tabular]}>
                    {carCount > 1
                      ? `${formatRelativeTime(item.createdAt)} \u00B7 ${carCount} cars`
                      : formatRelativeTime(item.createdAt)}
                  </Text>
                </View>

                <Feather
                  name="chevron-right"
                  size={18}
                  color={palette.faint}
                />
              </Pressable>
            );
          }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.ground,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.md,
  },
  titleBlock: {
    gap: 2,
  },
  countText: {
    ...typography.small,
  },
  confirmRow: {
    flexDirection: "row",
    gap: spacing.sm,
  },
  confirmPill: {
    backgroundColor: palette.negativeBg,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  confirmText: {
    fontFamily: fonts.semibold,
    fontSize: 13,
    color: palette.negativeInk,
  },
  cancelPill: {
    backgroundColor: palette.raised,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  cancelText: {
    fontFamily: fonts.medium,
    fontSize: 13,
    color: palette.ink,
  },
  searchRow: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.md,
  },
  searchPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: palette.surface,
    borderRadius: radius.pill,
    paddingHorizontal: 16,
    height: 42,
  },
  searchInput: {
    flex: 1,
    fontFamily: fonts.regular,
    fontSize: 14,
    color: palette.ink,
  },
  emptyWrap: {
    paddingHorizontal: spacing.lg,
  },
  emptyCard: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    alignItems: "flex-start",
    gap: 10,
  },
  emptyIconCircle: {
    width: 44,
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyBody: {
    ...typography.body,
    color: palette.muted,
  },
  primaryPill: {
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 4,
  },
  primaryPillText: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    color: palette.primaryInk,
  },
  listContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
    gap: 12,
  },
  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: 12,
    gap: 14,
  },
  pressedFade: {
    opacity: 0.78,
  },
  thumb: {
    width: 76,
    height: 76,
    borderRadius: 12,
    backgroundColor: palette.raised,
  },
  cardBody: {
    flex: 1,
    gap: 4,
  },
  cardTitle: {
    fontFamily: fonts.semibold,
    fontSize: 16,
    lineHeight: 22,
    color: palette.ink,
  },
  specLine: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  swatch: {
    width: 11,
    height: 11,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.control,
  },
  specText: {
    flex: 1,
    fontFamily: fonts.regular,
    fontSize: 13,
    lineHeight: 18,
    color: palette.muted,
  },
  timeText: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.faint,
  },
});
