import { Feather } from "@expo/vector-icons";
import React, { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { TopKPrediction, VehicleDetectionResult } from "../types/api";
import {
  formatModelTitle,
  formatPercent,
  formatYearSpanOnly,
} from "../utils/formatters";

interface VehicleCardProps {
  vehicle: VehicleDetectionResult;
}

function CandidateGroup({
  title,
  items,
}: {
  title: string;
  items: TopKPrediction[];
}): React.ReactElement | null {
  if (items.length === 0) {
    return null;
  }

  return (
    <View style={styles.candidateGroup}>
      <Text style={styles.candidateGroupTitle}>{title}</Text>
      {items.map((item, idx) => {
        const widthPct = Math.round(
          Math.max(0, Math.min(1, item.confidence)) * 100,
        );
        return (
          <View key={`${item.label}-${idx}`} style={styles.candidateRow}>
            <View style={styles.candidateLabelRow}>
              <Text style={styles.candidateName} numberOfLines={1}>
                {item.display_name}
              </Text>
              <Text style={[styles.candidateConf, typography.tabular]}>
                {formatPercent(item.confidence, 0)}
              </Text>
            </View>
            <View style={styles.candidateTrack}>
              <View
                style={[styles.candidateFill, { width: `${widthPct}%` }]}
              />
            </View>
          </View>
        );
      })}
    </View>
  );
}

/**
 * Borderless dark surface card presenting the car's Make, Model, Year,
 * Body style, and Paint Color, with optional expandable alternatives.
 */
export function VehicleCard({
  vehicle,
}: VehicleCardProps): React.ReactElement {
  const [expanded, setExpanded] = useState(false);

  const { make, model, year, body_type: bodyType, color } = vehicle.attributes;
  const cleanModel = formatModelTitle(make.display_name, model.display_name);
  const yearSpan = formatYearSpanOnly(year.range_low, year.range_high);

  return (
    <View style={styles.card}>
      {/* Hero identity */}
      <View style={styles.heroRow}>
        <View style={styles.heroLeft}>
          <Text style={styles.makeEyebrow}>{make.display_name}</Text>
          <Text style={styles.modelTitle} numberOfLines={2}>
            {cleanModel}
          </Text>
        </View>

        <View style={styles.matchPill}>
          <Text style={[styles.matchPillText, typography.tabular]}>
            {`${formatPercent(model.confidence, 0)} match`}
          </Text>
        </View>
      </View>

      {model.few_shot_match?.is_few_shot ? (
        <View style={styles.closestPill}>
          <Text style={styles.closestText}>
            {`Closest match: ${model.few_shot_match.display_name}`}
          </Text>
        </View>
      ) : null}

      {/* 3-tile spec grid */}
      <View style={styles.specRow}>
        <View style={styles.specTile}>
          <Text style={styles.specLabel}>Year</Text>
          <Text style={[styles.specValue, typography.tabular]}>
            {year.predicted_rounded}
          </Text>
          {yearSpan ? (
            <Text style={[styles.specSub, typography.tabular]}>
              {yearSpan}
            </Text>
          ) : null}
        </View>

        <View style={styles.specTile}>
          <Text style={styles.specLabel}>Body</Text>
          <Text
            style={styles.specValue}
            numberOfLines={1}
            adjustsFontSizeToFit
            minimumFontScale={0.8}
          >
            {bodyType.display_name}
          </Text>
        </View>

        <View style={styles.specTile}>
          <Text style={styles.specLabel}>Color</Text>
          <View style={styles.colorValueRow}>
            <View
              style={[
                styles.swatch,
                { backgroundColor: color.hex || palette.raised },
              ]}
            />
            <Text
              style={styles.colorValueText}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.8}
            >
              {color.display_name}
            </Text>
          </View>
        </View>
      </View>

      {/* Expandable other likely matches */}
      <View style={styles.expandDivider}>
        <Pressable
          accessibilityRole="button"
          accessibilityState={{ expanded }}
          onPress={() => setExpanded((prev) => !prev)}
          style={({ pressed }) => [
            styles.expandTrigger,
            pressed && styles.expandPressed,
          ]}
        >
          <Text style={styles.expandLabel}>Other likely matches</Text>
          <Feather
            name={expanded ? "chevron-up" : "chevron-down"}
            size={18}
            color={palette.muted}
          />
        </Pressable>

        {expanded ? (
          <View style={styles.expandBody}>
            <CandidateGroup title="Model" items={model.top3} />
            <CandidateGroup title="Make" items={make.top3} />
            <CandidateGroup title="Body" items={bodyType.top3} />
            <CandidateGroup title="Color" items={color.top3} />
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    gap: spacing.md,
  },
  heroRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: spacing.md,
  },
  heroLeft: {
    flex: 1,
    gap: 2,
  },
  makeEyebrow: {
    fontFamily: fonts.medium,
    fontSize: 14,
    lineHeight: 20,
    color: palette.muted,
  },
  modelTitle: {
    ...typography.heroModel,
  },
  matchPill: {
    backgroundColor: palette.positiveBg,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginTop: 2,
  },
  matchPillText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.positiveInk,
  },
  closestPill: {
    alignSelf: "flex-start",
    backgroundColor: palette.warningBg,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  closestText: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.warningInk,
  },
  specRow: {
    flexDirection: "row",
    gap: 10,
  },
  specTile: {
    flex: 1,
    backgroundColor: palette.raised,
    borderRadius: radius.tile,
    padding: 12,
    gap: 4,
    minHeight: 74,
  },
  specLabel: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  specValue: {
    fontFamily: fonts.semibold,
    fontSize: 17,
    lineHeight: 22,
    color: palette.ink,
  },
  specSub: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.faint,
  },
  colorValueRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  swatch: {
    width: 14,
    height: 14,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.control,
  },
  colorValueText: {
    flex: 1,
    fontFamily: fonts.semibold,
    fontSize: 16,
    lineHeight: 22,
    color: palette.ink,
  },
  expandDivider: {
    borderTopWidth: 1,
    borderTopColor: palette.border,
    paddingTop: spacing.md,
    gap: spacing.md,
  },
  expandTrigger: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 2,
  },
  expandPressed: {
    opacity: 0.72,
  },
  expandLabel: {
    fontFamily: fonts.medium,
    fontSize: 14,
    lineHeight: 20,
    color: palette.muted,
  },
  expandBody: {
    gap: spacing.md,
  },
  candidateGroup: {
    backgroundColor: palette.raised,
    borderRadius: radius.tile,
    padding: 12,
    gap: 8,
  },
  candidateGroupTitle: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  candidateRow: {
    gap: 4,
  },
  candidateLabelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm,
  },
  candidateName: {
    flex: 1,
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
    color: palette.ink,
  },
  candidateConf: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.accent,
  },
  candidateTrack: {
    width: "100%",
    height: 3,
    borderRadius: 2,
    backgroundColor: palette.surface,
    overflow: "hidden",
  },
  candidateFill: {
    height: "100%",
    backgroundColor: palette.accent,
    borderRadius: 2,
  },
});
