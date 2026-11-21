import { Feather } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import React, { useEffect, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { TopKPrediction, VehicleDetectionResult } from "../types/api";
import {
  formatLatencySeconds,
  formatModelVersionLabel,
  formatPercent,
  formatYearSpanOnly,
  getConfidenceTier,
  resolveVehicleModelSelection,
  splitCandidateMakeAndModel,
} from "../utils/formatters";

interface VehicleCardProps {
  vehicle: VehicleDetectionResult;
  modelVersion?: string | null;
  timingsMs?: number | null;
  selectedCandidateIndex?: number;
  onSelectCandidate?: (index: number) => void;
}

function CandidateGroup({
  title,
  items,
  selectedIndex,
  onSelectIndex,
}: {
  title: string;
  items: TopKPrediction[];
  selectedIndex?: number;
  onSelectIndex?: (index: number) => void;
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
        const isSelected = selectedIndex === idx;
        const rowBody = (
          <>
            <View style={styles.candidateLabelRow}>
              <View style={styles.candidateNameRow}>
                {isSelected ? (
                  <Feather name="check" size={13} color={palette.accent} />
                ) : null}
                <Text
                  style={[
                    styles.candidateName,
                    isSelected && styles.candidateNameActive,
                  ]}
                  numberOfLines={1}
                >
                  {item.display_name}
                </Text>
              </View>
              <Text style={[styles.candidateConf, typography.tabular]}>
                {formatPercent(item.confidence, 0)}
              </Text>
            </View>
            <View style={styles.candidateTrack}>
              <View
                style={[styles.candidateFill, { width: `${widthPct}%` }]}
              />
            </View>
          </>
        );

        if (onSelectIndex) {
          return (
            <Pressable
              key={`${item.label}-${idx}`}
              accessibilityRole="button"
              accessibilityState={{ selected: isSelected }}
              onPress={() => onSelectIndex(idx)}
              style={({ pressed }) => [
                styles.candidateRow,
                styles.candidateRowInteractive,
                isSelected && styles.candidateRowActive,
                pressed && styles.altPressed,
              ]}
            >
              {rowBody}
            </Pressable>
          );
        }

        return (
          <View key={`${item.label}-${idx}`} style={styles.candidateRow}>
            {rowBody}
          </View>
        );
      })}
    </View>
  );
}

/**
 * Borderless dark surface card presenting the car's Make, Model, Year,
 * Body style, Paint Color, interactive candidate switcher, and Top-3 breakdown.
 */
export function VehicleCard({
  vehicle,
  modelVersion,
  timingsMs,
  selectedCandidateIndex,
  onSelectCandidate,
}: VehicleCardProps): React.ReactElement {
  const { make, model, year, body_type: bodyType, color } = vehicle.attributes;
  const [expanded, setExpanded] = useState(model.confidence < 0.75);

  useEffect(() => {
    setExpanded(model.confidence < 0.75);
  }, [vehicle.vehicle_index, model.canonical_class]);

  const active = resolveVehicleModelSelection(vehicle, selectedCandidateIndex);
  const yearSpan = formatYearSpanOnly(year.range_low, year.range_high);
  const top1Tier = getConfidenceTier(model.confidence);
  const activeTier = getConfidenceTier(active.confidence);
  const matchPct = formatPercent(active.confidence, 0);

  const matchLabel = active.isOverride
    ? `${matchPct} \u00B7 Selected`
    : activeTier === "high"
      ? `${matchPct} match`
      : activeTier === "medium"
        ? `${matchPct} \u00B7 Likely`
        : `${matchPct} \u00B7 Close match`;

  const matchPillStyle = active.isOverride
    ? styles.matchPillPositive
    : activeTier === "high"
      ? styles.matchPillPositive
      : activeTier === "medium"
        ? styles.matchPillNeutral
        : styles.matchPillWarning;
  const matchTextStyle = active.isOverride
    ? styles.matchTextPositive
    : activeTier === "high"
      ? styles.matchTextPositive
      : activeTier === "medium"
        ? styles.matchTextNeutral
        : styles.matchTextWarning;

  const parsedCandidates = active.candidates.map((cand) => ({
    cand,
    parsed: splitCandidateMakeAndModel(
      cand,
      make.label,
      make.display_name,
      make.top3,
    ),
  }));
  const allSameMake = parsedCandidates.every(
    ({ parsed }) =>
      parsed.makeDisplay.trim().toLowerCase() ===
      make.display_name.trim().toLowerCase(),
  );

  const handleSelectCandidate = (idx: number): void => {
    if (idx === active.index) {
      return;
    }
    Haptics.selectionAsync().catch(() => {});
    onSelectCandidate?.(idx);
  };

  const footerMeta = [
    modelVersion ? formatModelVersionLabel(modelVersion) : null,
    timingsMs ? formatLatencySeconds(timingsMs) : null,
  ]
    .filter(Boolean)
    .join(" \u00B7 ");

  return (
    <View style={styles.card}>
      {/* Hero identity */}
      <View style={styles.heroRow}>
        <View style={styles.heroLeft}>
          <Text style={styles.makeEyebrow}>{active.makeDisplay}</Text>
          <Text style={styles.modelTitle} numberOfLines={2}>
            {active.modelTitle}
          </Text>
        </View>

        <View style={[styles.matchPill, matchPillStyle]}>
          <Text style={[styles.matchPillText, typography.tabular, matchTextStyle]}>
            {matchLabel}
          </Text>
        </View>
      </View>

      {top1Tier === "low" && !active.isOverride ? (
        <View style={styles.advisoryBanner}>
          <Feather name="info" size={14} color={palette.warningInk} />
          <Text style={styles.advisoryText}>
            Close match — tap a variant below to switch.
          </Text>
        </View>
      ) : null}

      {model.few_shot_match?.is_few_shot &&
      model.few_shot_match.canonical_class !== active.canonicalClass ? (
        <View style={styles.closestPill}>
          <Text style={styles.closestText}>
            {`Closest prototype: ${model.few_shot_match.display_name} (${formatPercent(model.few_shot_match.similarity, 0)})`}
          </Text>
        </View>
      ) : null}

      {/* Interactive candidate switcher across Top-K model predictions */}
      {active.candidates.length > 1 ? (
        <View style={styles.altRow}>
          <View style={styles.altHeaderRow}>
            <Text style={styles.altLabel}>
              {top1Tier === "low" ? "Possible models" : "Model candidates"}
            </Text>
            {active.isOverride ? (
              <Pressable
                accessibilityRole="button"
                hitSlop={6}
                onPress={() => handleSelectCandidate(0)}
                style={({ pressed }) => [pressed && styles.altPressed]}
              >
                <Text style={styles.resetLink}>Reset to #1</Text>
              </Pressable>
            ) : (
              <Text style={styles.altHint}>Tap to switch</Text>
            )}
          </View>
          <View style={styles.altChips}>
            {parsedCandidates.map(({ cand, parsed }, idx) => {
              const isSelected = idx === active.index;
              const chipLabel = allSameMake
                ? parsed.modelTitle
                : cand.display_name;
              return (
                <Pressable
                  key={`${cand.label}-${idx}`}
                  accessibilityRole="button"
                  accessibilityState={{ selected: isSelected }}
                  onPress={() => handleSelectCandidate(idx)}
                  style={({ pressed }) => [
                    styles.altChip,
                    isSelected && styles.altChipActive,
                    pressed && styles.altPressed,
                  ]}
                >
                  <Text
                    style={[
                      styles.altChipName,
                      isSelected && styles.altChipNameActive,
                    ]}
                    numberOfLines={1}
                  >
                    {chipLabel}
                  </Text>
                  <Text
                    style={[
                      styles.altChipPct,
                      typography.tabular,
                      isSelected && styles.altChipPctActive,
                    ]}
                  >
                    {formatPercent(cand.confidence, 0)}
                  </Text>
                </Pressable>
              );
            })}
          </View>
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
          <Text style={[styles.specSub, typography.tabular]}>
            {formatPercent(bodyType.confidence, 0)}
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
          <Text style={[styles.specSub, typography.tabular]}>
            {formatPercent(color.confidence, 0)}
          </Text>
        </View>
      </View>

      {/* Expandable full Top-3 breakdown */}
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
          <Text style={styles.expandLabel}>All predictions & probabilities</Text>
          <Feather
            name={expanded ? "chevron-up" : "chevron-down"}
            size={18}
            color={palette.muted}
          />
        </Pressable>

        {expanded ? (
          <View style={styles.expandBody}>
            <CandidateGroup
              title="Model (Top 3)"
              items={active.candidates}
              selectedIndex={active.index}
              onSelectIndex={handleSelectCandidate}
            />
            <CandidateGroup title="Make (Top 3)" items={make.top3} />
            <CandidateGroup title="Body (Top 3)" items={bodyType.top3} />
            <CandidateGroup title="Color (Top 3)" items={color.top3} />
            {footerMeta ? (
              <Text style={[styles.footerMeta, typography.tabular]}>
                {footerMeta}
              </Text>
            ) : null}
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
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginTop: 2,
  },
  matchPillPositive: {
    backgroundColor: palette.positiveBg,
  },
  matchPillNeutral: {
    backgroundColor: palette.neutralBg,
  },
  matchPillWarning: {
    backgroundColor: palette.warningBg,
  },
  matchPillText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
  },
  matchTextPositive: {
    color: palette.positiveInk,
  },
  matchTextNeutral: {
    color: palette.neutralInk,
  },
  matchTextWarning: {
    color: palette.warningInk,
  },
  advisoryBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: palette.warningBg,
    borderRadius: radius.tile,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  advisoryText: {
    flex: 1,
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 17,
    color: palette.warningInk,
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
  altRow: {
    gap: 8,
  },
  altHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm,
  },
  altLabel: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  altHint: {
    fontFamily: fonts.regular,
    fontSize: 11,
    lineHeight: 15,
    color: palette.faint,
  },
  resetLink: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.accent,
  },
  altChips: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  altChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: palette.raised,
    borderRadius: radius.pill,
    paddingHorizontal: 12,
    paddingVertical: 7,
    maxWidth: "100%",
  },
  altChipActive: {
    backgroundColor: palette.primaryBg,
  },
  altPressed: {
    opacity: 0.72,
  },
  altChipName: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.ink,
    flexShrink: 1,
  },
  altChipNameActive: {
    fontFamily: fonts.semibold,
    color: palette.primaryInk,
  },
  altChipPct: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.accent,
  },
  altChipPctActive: {
    color: palette.primaryInk,
    opacity: 0.72,
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
  candidateRowInteractive: {
    borderRadius: radius.chip,
    paddingHorizontal: 6,
    paddingVertical: 4,
    marginHorizontal: -6,
  },
  candidateRowActive: {
    backgroundColor: palette.surface,
  },
  candidateLabelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: spacing.sm,
  },
  candidateNameRow: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  candidateName: {
    flex: 1,
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
    color: palette.ink,
  },
  candidateNameActive: {
    fontFamily: fonts.semibold,
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
  footerMeta: {
    fontFamily: fonts.regular,
    fontSize: 11,
    lineHeight: 16,
    color: palette.faint,
    textAlign: "right",
  },
});
