import React from "react";
import { FlatList, Image, StyleSheet, Text, View } from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { SimilarVehicle } from "../types/api";
import { formatPercent, resolveThumbnailUrl } from "../utils/formatters";

interface SimilarCarsStripProps {
  similar: SimilarVehicle[];
  serverUrl: string;
}

/**
 * Horizontal visual strip of similar-looking vehicles returned with the scan.
 */
export function SimilarCarsStrip({
  similar,
  serverUrl,
}: SimilarCarsStripProps): React.ReactElement | null {
  if (similar.length === 0) {
    return null;
  }

  return (
    <View style={styles.section}>
      <Text style={styles.heading}>Similar cars</Text>

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

          return (
            <View style={styles.card}>
              <Image
                source={{ uri }}
                style={styles.thumb}
                resizeMode="cover"
                accessibilityLabel={item.display_name}
              />
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
            </View>
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    gap: spacing.md,
  },
  heading: {
    ...typography.section,
    paddingHorizontal: 2,
  },
  listContent: {
    gap: spacing.md,
  },
  card: {
    width: 168,
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    overflow: "hidden",
  },
  thumb: {
    width: "100%",
    height: 110,
    backgroundColor: palette.raised,
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
});
