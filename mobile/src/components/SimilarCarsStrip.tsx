import { Feather } from "@expo/vector-icons";
import React, { useState } from "react";
import {
  FlatList,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { SimilarVehicle } from "../types/api";
import {
  formatPercent,
  resolveThumbnailUrl,
  withThumbnailWidth,
} from "../utils/formatters";

interface SimilarCarsStripProps {
  similar: SimilarVehicle[];
  serverUrl: string;
  activeCanonicalClass?: string | null;
}

/**
 * Horizontal visual strip of similar-looking vehicles with a tap-to-zoom comparison lightbox.
 */
export function SimilarCarsStrip({
  similar,
  serverUrl,
  activeCanonicalClass,
}: SimilarCarsStripProps): React.ReactElement | null {
  const [selected, setSelected] = useState<SimilarVehicle | null>(null);

  if (similar.length === 0) {
    return null;
  }

  const selectedUri = selected
    ? withThumbnailWidth(
        resolveThumbnailUrl(serverUrl, selected.thumbnail_url),
        720,
      )
    : "";
  const selectedSubtitle = selected
    ? [selected.year ? String(selected.year) : null, selected.body_type]
        .filter(Boolean)
        .join(" \u00B7 ")
    : "";

  return (
    <View style={styles.section}>
      <View style={styles.headerRow}>
        <Text style={styles.heading}>Similar cars</Text>
        <Text style={styles.subheading}>Tap photo to compare</Text>
      </View>

      <FlatList
        horizontal
        data={similar}
        keyExtractor={(item, idx) => `${item.rel_path}-${idx}`}
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => {
          const uri = resolveThumbnailUrl(serverUrl, item.thumbnail_url);
          const subtitle = [item.year ? String(item.year) : null, item.body_type]
            .filter(Boolean)
            .join(" \u00B7 ");
          const isMatched = Boolean(
            activeCanonicalClass && item.canonical_class === activeCanonicalClass,
          );

          return (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Inspect similar ${item.display_name}`}
              onPress={() => setSelected(item)}
              style={({ pressed }) => [
                styles.card,
                isMatched && styles.cardMatched,
                pressed && styles.cardPressed,
              ]}
            >
              <View style={styles.thumbWrap}>
                <Image
                  source={{ uri }}
                  style={styles.thumb}
                  resizeMode="cover"
                  accessibilityLabel={item.display_name}
                />
                <View style={styles.zoomBadge} accessibilityElementsHidden>
                  <Feather name="maximize-2" size={11} color={palette.ink} />
                </View>
              </View>
              <View style={styles.cardBody}>
                <Text style={styles.title} numberOfLines={1}>
                  {item.display_name}
                </Text>
                {subtitle ? (
                  <Text style={styles.meta} numberOfLines={1}>
                    {subtitle}
                  </Text>
                ) : null}
                <Text style={[styles.matchText, typography.tabular]}>
                  {`${formatPercent(item.similarity, 0)} similar`}
                </Text>
              </View>
            </Pressable>
          );
        }}
      />

      <Modal
        visible={selected !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setSelected(null)}
      >
        <Pressable
          style={styles.modalBackdrop}
          onPress={() => setSelected(null)}
        >
          <Pressable style={styles.modalCard} onPress={() => {}}>
            <View style={styles.modalHeader}>
              <View style={styles.modalTitleCol}>
                <Text style={styles.modalTitle} numberOfLines={2}>
                  {selected?.display_name ?? ""}
                </Text>
                {selectedSubtitle ? (
                  <Text style={styles.modalMeta}>{selectedSubtitle}</Text>
                ) : null}
              </View>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Close reference preview"
                onPress={() => setSelected(null)}
                style={styles.modalCloseBtn}
              >
                <Feather name="x" size={18} color={palette.muted} />
              </Pressable>
            </View>

            {selectedUri ? (
              <Image
                source={{ uri: selectedUri }}
                style={styles.modalImage}
                resizeMode="contain"
              />
            ) : null}

            <View style={styles.modalFooter}>
              <View style={styles.simPill}>
                <Text style={[styles.simPillText, typography.tabular]}>
                  {`${formatPercent(selected?.similarity ?? 0, 0)} visual similarity`}
                </Text>
              </View>
              <Pressable
                accessibilityRole="button"
                onPress={() => setSelected(null)}
                style={styles.doneBtn}
              >
                <Text style={styles.doneBtnText}>Close</Text>
              </Pressable>
            </View>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    gap: spacing.md,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "baseline",
    paddingHorizontal: 2,
  },
  heading: {
    ...typography.section,
  },
  subheading: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.faint,
  },
  listContent: {
    gap: spacing.md,
  },
  card: {
    width: 168,
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    borderWidth: 1.5,
    borderColor: "transparent",
    overflow: "hidden",
  },
  cardMatched: {
    borderColor: palette.accent,
  },
  cardPressed: {
    opacity: 0.78,
  },
  thumbWrap: {
    position: "relative",
  },
  thumb: {
    width: "100%",
    height: 110,
    backgroundColor: palette.raised,
  },
  zoomBadge: {
    position: "absolute",
    top: 8,
    right: 8,
    width: 24,
    height: 24,
    borderRadius: radius.pill,
    backgroundColor: "#00000099",
    alignItems: "center",
    justifyContent: "center",
  },
  cardBody: {
    padding: 12,
    gap: 3,
  },
  title: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    lineHeight: 20,
    color: palette.ink,
  },
  meta: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  matchText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.accent,
    marginTop: 4,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "#000000D9",
    justifyContent: "center",
    padding: spacing.lg,
  },
  modalCard: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    gap: spacing.md,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: spacing.md,
  },
  modalTitleCol: {
    flex: 1,
    gap: 2,
  },
  modalTitle: {
    ...typography.title,
  },
  modalMeta: {
    ...typography.small,
  },
  modalCloseBtn: {
    width: 32,
    height: 32,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
  },
  modalImage: {
    width: "100%",
    height: 260,
    borderRadius: radius.tile,
    backgroundColor: palette.ground,
  },
  modalFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.md,
  },
  simPill: {
    backgroundColor: palette.positiveBg,
    borderRadius: radius.pill,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  simPillText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.positiveInk,
  },
  doneBtn: {
    backgroundColor: palette.raised,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.lg,
    paddingVertical: 10,
  },
  doneBtnText: {
    fontFamily: fonts.medium,
    fontSize: 14,
    color: palette.ink,
  },
});
