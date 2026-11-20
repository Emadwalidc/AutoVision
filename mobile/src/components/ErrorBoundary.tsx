import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { fonts, palette, radius, spacing, typography } from "../theme/tokens";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      errorMessage: "",
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error.message || "Something went wrong while rendering.",
    };
  }

  handleReset = (): void => {
    this.setState({ hasError: false, errorMessage: "" });
  };

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <View style={styles.card}>
            <View style={styles.tag}>
              <Text style={styles.tagText}>Unexpected error</Text>
            </View>
            <Text style={typography.title}>Could not open view</Text>
            <Text style={styles.message}>{this.state.errorMessage}</Text>
            <Pressable
              accessibilityRole="button"
              onPress={this.handleReset}
              style={({ pressed }) => [
                styles.button,
                pressed && styles.buttonPressed,
              ]}
            >
              <Text style={styles.buttonText}>Reload view</Text>
            </Pressable>
          </View>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: palette.ground,
    justifyContent: "center",
    padding: spacing.lg,
  },
  card: {
    backgroundColor: palette.surface,
    borderRadius: radius.card,
    padding: spacing.lg,
    gap: spacing.md,
  },
  tag: {
    alignSelf: "flex-start",
    backgroundColor: palette.negativeBg,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  tagText: {
    fontFamily: fonts.semibold,
    fontSize: 12,
    lineHeight: 16,
    color: palette.negativeInk,
  },
  message: {
    ...typography.body,
    color: palette.muted,
  },
  button: {
    alignSelf: "flex-start",
    backgroundColor: palette.primaryBg,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.lg,
    height: 44,
    justifyContent: "center",
  },
  buttonPressed: {
    opacity: 0.84,
  },
  buttonText: {
    fontFamily: fonts.semibold,
    fontSize: 14,
    color: palette.primaryInk,
  },
});
