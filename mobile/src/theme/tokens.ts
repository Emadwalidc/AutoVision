import type { TextStyle } from "react-native";

/**
 * Taste-Skill direction: True-black flat (seed 740244229).
 * Pure OLED black canvas, lightness-stepped dark surfaces with zero shadows or borders,
 * high-contrast inverted pill primary actions, Inter typography in sentence case,
 * and calibrated lime (#C6F24E) strictly reserved for data marks.
 */
export const palette = {
  ground: "#000000",
  surface: "#111111",
  raised: "#2A2A2A",
  border: "#1A1A1A",
  rule: "#161616",
  control: "#444444",

  ink: "#FFFFFF",
  muted: "#B4B4B4",
  faint: "#949494",

  primaryBg: "#FFFFFF",
  primaryInk: "#000000",

  accent: "#C6F24E",

  positiveBg: "#222A10",
  positiveInk: "#C6F24E",
  negativeBg: "#2C1A20",
  negativeInk: "#E5566F",
  warningBg: "#2B2414",
  warningInk: "#E5A93B",
  neutralBg: "#222222",
  neutralInk: "#FFFFFF",
} as const;

export const radius = {
  card: 20,
  tile: 14,
  pill: 999,
  chip: 8,
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 14,
  lg: 20,
  xl: 28,
} as const;

export const fonts = {
  regular: "Inter_400Regular",
  medium: "Inter_500Medium",
  semibold: "Inter_600SemiBold",
} as const;

export const typography = {
  title: {
    fontFamily: fonts.semibold,
    fontSize: 22,
    lineHeight: 28,
    color: palette.ink,
  } satisfies TextStyle,
  heroModel: {
    fontFamily: fonts.semibold,
    fontSize: 26,
    lineHeight: 32,
    letterSpacing: -0.3,
    color: palette.ink,
  } satisfies TextStyle,
  section: {
    fontFamily: fonts.semibold,
    fontSize: 16,
    lineHeight: 22,
    color: palette.ink,
  } satisfies TextStyle,
  kpi: {
    fontFamily: fonts.semibold,
    fontSize: 20,
    lineHeight: 26,
    color: palette.ink,
  } satisfies TextStyle,
  body: {
    fontFamily: fonts.regular,
    fontSize: 14,
    lineHeight: 20,
    color: palette.ink,
  } satisfies TextStyle,
  bodyMedium: {
    fontFamily: fonts.medium,
    fontSize: 14,
    lineHeight: 20,
    color: palette.ink,
  } satisfies TextStyle,
  bodySemibold: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    lineHeight: 20,
    color: palette.ink,
  } satisfies TextStyle,
  small: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  } satisfies TextStyle,
  smallMedium: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  } satisfies TextStyle,
  tabular: {
    fontVariant: ["tabular-nums"],
  } satisfies TextStyle,
} as const;
