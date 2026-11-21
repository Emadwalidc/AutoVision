import { Feather } from "@expo/vector-icons";
import React, { useEffect, useState } from "react";
import {
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import {
  checkInferenceHealth,
  type InferenceMode,
} from "../services/inferenceService";
import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import type { HealthResponse } from "../types/api";
import {
  DEFAULT_SERVER_URL,
  formatCompactCount,
  formatModelVersionLabel,
  normalizeServerUrl,
} from "../utils/formatters";

interface ConnectionSheetProps {
  visible: boolean;
  serverUrl: string;
  inferenceMode: InferenceMode;
  onDeviceAvailable: boolean;
  onChangeInferenceMode: (mode: InferenceMode) => void;
  onSave: (cleanUrl: string) => Promise<void>;
  onHealthChecked?: (health: HealthResponse | null) => void;
  onClose: () => void;
}

/**
 * Inference mode & AutoVision health sheet opened via top-bar button
 * or "Change address" on offline server errors.
 */
export function ConnectionSheet({
  visible,
  serverUrl,
  inferenceMode,
  onDeviceAvailable,
  onChangeInferenceMode,
  onSave,
  onHealthChecked,
  onClose,
}: ConnectionSheetProps): React.ReactElement {
  const [draft, setDraft] = useState(serverUrl);
  const [testing, setTesting] = useState(false);
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<{
    kind: "ok" | "err";
    text: string;
  } | null>(null);

  useEffect(() => {
    if (!visible) {
      return;
    }
    let active = true;
    setDraft(serverUrl);
    setStatus(null);
    void checkInferenceHealth(inferenceMode, serverUrl)
      .then((h) => {
        if (!active) {
          return;
        }
        setHealthInfo(h);
        onHealthChecked?.(h);
      })
      .catch(() => {
        if (!active) {
          return;
        }
        setHealthInfo(null);
        onHealthChecked?.(null);
      });
    return () => {
      active = false;
    };
  }, [visible, inferenceMode, serverUrl, onHealthChecked]);

  const handleSave = async (): Promise<void> => {
    const clean = normalizeServerUrl(draft);
    if (!clean) {
      setStatus({ kind: "err", text: "Enter a valid local address." });
      return;
    }

    setTesting(true);
    setStatus(null);
    await onSave(clean);

    try {
      const h = await checkInferenceHealth("server", clean);
      setHealthInfo(h);
      onHealthChecked?.(h);
      setStatus({ kind: "ok", text: "Connected and saved." });
    } catch {
      setHealthInfo(null);
      onHealthChecked?.(null);
      setStatus({
        kind: "err",
        text: "Saved, but still could not reach that address.",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleReset = async (): Promise<void> => {
    setDraft(DEFAULT_SERVER_URL);
    await onSave(DEFAULT_SERVER_URL);
    try {
      const h = await checkInferenceHealth("server", DEFAULT_SERVER_URL);
      setHealthInfo(h);
      onHealthChecked?.(h);
    } catch {
      setHealthInfo(null);
      onHealthChecked?.(null);
    }
    setStatus({ kind: "ok", text: "Reset to default address." });
  };

  const isOfflineMode = onDeviceAvailable && inferenceMode === "on-device";

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          <View style={styles.header}>
            <Text style={typography.section}>Engine</Text>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Close connection sheet"
              onPress={onClose}
              style={styles.closeBtn}
            >
              <Feather name="x" size={18} color={palette.muted} />
            </Pressable>
          </View>

          {onDeviceAvailable ? (
            <View style={styles.modeRow}>
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ selected: isOfflineMode }}
                onPress={() => onChangeInferenceMode("on-device")}
                style={[
                  styles.modePill,
                  isOfflineMode && styles.modePillActive,
                ]}
              >
                <Text
                  style={[
                    styles.modeText,
                    isOfflineMode && styles.modeTextActive,
                  ]}
                >
                  On-device (Offline)
                </Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityState={{ selected: !isOfflineMode }}
                onPress={() => onChangeInferenceMode("server")}
                style={[
                  styles.modePill,
                  !isOfflineMode && styles.modePillActive,
                ]}
              >
                <Text
                  style={[
                    styles.modeText,
                    !isOfflineMode && styles.modeTextActive,
                  ]}
                >
                  Remote server
                </Text>
              </Pressable>
            </View>
          ) : null}

          <Text style={styles.description}>
            {isOfflineMode
              ? "Runs YOLOv8n-seg, ViT-B/16, and Color Expert directly on your phone — no laptop or internet required."
              : "Local address of your computer running AutoVision on the same Wi-Fi."}
          </Text>

          {!isOfflineMode ? (
            <TextInput
              value={draft}
              onChangeText={setDraft}
              placeholder={DEFAULT_SERVER_URL}
              placeholderTextColor={palette.faint}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              style={[styles.input, typography.tabular]}
            />
          ) : null}

          {healthInfo ? (
            <View style={styles.healthCard}>
              <View style={styles.healthTop}>
                <Text style={styles.healthTitle}>
                  {formatModelVersionLabel(healthInfo.model_version)}
                </Text>
                <View style={styles.healthBadge}>
                  <Text style={[styles.healthBadgeText, typography.tabular]}>
                    {healthInfo.device.toUpperCase()}
                  </Text>
                </View>
              </View>
              <Text style={[styles.healthMeta, typography.tabular]}>
                {[
                  `${healthInfo.num_makes} makes`,
                  `${healthInfo.num_prototype_classes} models`,
                  `${healthInfo.num_colors} colors`,
                  isOfflineMode
                    ? "Offline"
                    : `${formatCompactCount(healthInfo.gallery_size)} gallery`,
                ].join(" \u00B7 ")}
              </Text>
            </View>
          ) : null}

          {!isOfflineMode && status ? (
            <View
              style={[
                styles.statusPill,
                status.kind === "ok" ? styles.statusOk : styles.statusErr,
              ]}
            >
              <Text
                style={[
                  styles.statusText,
                  status.kind === "ok"
                    ? styles.statusOkText
                    : styles.statusErrText,
                ]}
              >
                {status.text}
              </Text>
            </View>
          ) : null}

          {isOfflineMode ? (
            <Pressable
              accessibilityRole="button"
              onPress={onClose}
              style={({ pressed }) => [
                styles.donePill,
                pressed && styles.btnMuted,
              ]}
            >
              <Text style={styles.primaryBtnText}>Done</Text>
            </Pressable>
          ) : (
            <View style={styles.actions}>
              <Pressable
                accessibilityRole="button"
                disabled={testing}
                onPress={() => {
                  void handleSave();
                }}
                style={({ pressed }) => [
                  styles.primaryBtn,
                  (pressed || testing) && styles.btnMuted,
                ]}
              >
                <Text style={styles.primaryBtnText}>
                  {testing ? "Checking..." : "Save & test"}
                </Text>
              </Pressable>

              <Pressable
                accessibilityRole="button"
                disabled={testing}
                onPress={() => {
                  void handleReset();
                }}
                style={styles.secondaryBtn}
              >
                <Text style={styles.secondaryBtnText}>Reset</Text>
              </Pressable>
            </View>
          )}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "#000000CC",
    justifyContent: "center",
    padding: spacing.lg,
  },
  sheet: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    gap: spacing.md,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  closeBtn: {
    width: 32,
    height: 32,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
  },
  description: {
    fontFamily: fonts.regular,
    fontSize: 13,
    lineHeight: 18,
    color: palette.muted,
  },
  input: {
    backgroundColor: palette.raised,
    borderRadius: radius.tile,
    paddingHorizontal: spacing.md,
    paddingVertical: 12,
    fontFamily: fonts.medium,
    fontSize: 14,
    color: palette.ink,
  },
  statusPill: {
    borderRadius: radius.tile,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  statusOk: {
    backgroundColor: palette.positiveBg,
  },
  statusErr: {
    backgroundColor: palette.negativeBg,
  },
  statusText: {
    fontFamily: fonts.medium,
    fontSize: 13,
    lineHeight: 18,
  },
  statusOkText: {
    color: palette.positiveInk,
  },
  statusErrText: {
    color: palette.negativeInk,
  },
  actions: {
    flexDirection: "row",
    gap: 10,
  },
  primaryBtn: {
    flex: 1,
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.md,
  },
  primaryBtnText: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    color: palette.primaryInk,
  },
  secondaryBtn: {
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.raised,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
  },
  secondaryBtnText: {
    fontFamily: fonts.medium,
    fontSize: 14,
    color: palette.ink,
  },
  btnMuted: {
    opacity: 0.7,
  },
  modeRow: {
    flexDirection: "row",
    gap: 8,
    backgroundColor: palette.ground,
    borderRadius: radius.pill,
    padding: 4,
  },
  modePill: {
    flex: 1,
    height: 36,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 10,
  },
  modePillActive: {
    backgroundColor: palette.primaryBg,
  },
  modeText: {
    fontFamily: fonts.medium,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
  modeTextActive: {
    fontFamily: fonts.semibold,
    color: palette.primaryInk,
  },
  donePill: {
    height: 44,
    borderRadius: radius.pill,
    backgroundColor: palette.primaryBg,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
  },
  healthCard: {
    backgroundColor: palette.raised,
    borderRadius: radius.tile,
    paddingHorizontal: spacing.md,
    paddingVertical: 10,
    gap: 4,
  },
  healthTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  healthTitle: {
    fontFamily: fonts.semibold,
    fontSize: 13,
    lineHeight: 18,
    color: palette.ink,
  },
  healthBadge: {
    backgroundColor: palette.positiveBg,
    borderRadius: radius.pill,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  healthBadgeText: {
    fontFamily: fonts.semibold,
    fontSize: 11,
    lineHeight: 15,
    color: palette.positiveInk,
  },
  healthMeta: {
    fontFamily: fonts.regular,
    fontSize: 12,
    lineHeight: 16,
    color: palette.muted,
  },
});
