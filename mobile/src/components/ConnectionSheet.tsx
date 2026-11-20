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

import { checkHealth } from "../services/apiClient";
import { fonts, palette, radius, spacing, typography } from "../theme/tokens";
import { DEFAULT_SERVER_URL, normalizeServerUrl } from "../utils/formatters";

interface ConnectionSheetProps {
  visible: boolean;
  serverUrl: string;
  onSave: (cleanUrl: string) => Promise<void>;
  onClose: () => void;
}

/**
 * Unobtrusive modal sheet opened only when the user taps "Change address"
 * on an offline error or long-presses the app wordmark.
 */
export function ConnectionSheet({
  visible,
  serverUrl,
  onSave,
  onClose,
}: ConnectionSheetProps): React.ReactElement {
  const [draft, setDraft] = useState(serverUrl);
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState<{
    kind: "ok" | "err";
    text: string;
  } | null>(null);

  useEffect(() => {
    if (visible) {
      setDraft(serverUrl);
      setStatus(null);
    }
  }, [visible, serverUrl]);

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
      await checkHealth(clean);
      setStatus({ kind: "ok", text: "Connected and saved." });
    } catch {
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
    setStatus({ kind: "ok", text: "Reset to default address." });
  };

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
            <Text style={typography.section}>Connection</Text>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Close connection sheet"
              onPress={onClose}
              style={styles.closeBtn}
            >
              <Feather name="x" size={18} color={palette.muted} />
            </Pressable>
          </View>

          <Text style={styles.description}>
            Local address of your computer running AutoVision on the same Wi-Fi.
          </Text>

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

          {status ? (
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
});
