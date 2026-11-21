const fs = require("fs");
const path = require("path");
const {
  withAndroidManifest,
  withAppBuildGradle,
  withDangerousMod,
  withMainApplication,
} = require("@expo/config-plugins");

const PACKAGE_NAME = "com.autovision.mobile";
const ORT_DEP = 'implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.0")';
const EXIF_DEP = 'implementation("androidx.exifinterface:exifinterface:1.3.7")';

function withOnDeviceBuildGradle(config) {
  return withAppBuildGradle(config, (mod) => {
    let contents = mod.modResults.contents;

    if (!contents.includes("onnxruntime-android")) {
      contents = contents.replace(
        /^dependencies\s*\{/m,
        `dependencies {\n    ${ORT_DEP}\n    ${EXIF_DEP}`,
      );
    }

    if (!contents.includes("noCompress")) {
      contents = contents.replace(
        /^android\s*\{/m,
        `android {\n    androidResources {\n        noCompress += ["onnx", "bin", "json"]\n    }\n    externalNativeBuild {\n        cmake {\n            version "3.31.4"\n        }\n    }`,
      );
    }

    if (!contents.includes("CMAKE_OBJECT_PATH_MAX=160")) {
      contents = contents.replace(
        /defaultConfig\s*\{/,
        `defaultConfig {\n        externalNativeBuild {\n            cmake {\n                arguments "-DCMAKE_OBJECT_PATH_MAX=160"\n            }\n        }`,
      );
    }

    mod.modResults.contents = contents;
    return mod;
  });
}

function withCleartextManifest(config) {
  return withAndroidManifest(config, (mod) => {
    const app = mod.modResults.manifest.application?.[0];
    if (app && app.$) {
      app.$["android:usesCleartextTraffic"] = "true";
    }
    return mod;
  });
}

function withOnDeviceMainApplication(config) {
  return withMainApplication(config, (mod) => {
    let contents = mod.modResults.contents;
    if (!contents.includes("AutoVisionOnDevicePackage")) {
      if (contents.includes("PackageList(this).packages.apply {")) {
        contents = contents.replace(
          /PackageList\(this\)\.packages\.apply\s*\{/,
          "PackageList(this).packages.apply {\n              add(AutoVisionOnDevicePackage())",
        );
      } else if (contents.includes("PackageList(this).packages")) {
        contents = contents.replace(
          /(val packages[^=]*=\s*PackageList\(this\)\.packages)/,
          "$1\n            packages.add(AutoVisionOnDevicePackage())",
        );
      } else {
        throw new Error(
          "Could not locate PackageList(this).packages in MainApplication.kt to register AutoVisionOnDevicePackage().",
        );
      }
    }
    mod.modResults.contents = contents;
    return mod;
  });
}

function withOnDeviceAssetsAndSources(config) {
  return withDangerousMod(config, [
    "android",
    async (mod) => {
      const projectRoot = mod.modRequest.projectRoot;
      const androidRoot = mod.modRequest.platformProjectRoot;

      const srcKtDir = path.join(projectRoot, "native", "android");
      const dstKtDir = path.join(
        androidRoot,
        "app",
        "src",
        "main",
        "java",
        ...PACKAGE_NAME.split("."),
      );
      fs.mkdirSync(dstKtDir, { recursive: true });
      for (const fileName of [
        "AutoVisionOnDeviceModule.kt",
        "AutoVisionOnDevicePackage.kt",
      ]) {
        const srcPath = path.join(srcKtDir, fileName);
        if (!fs.existsSync(srcPath)) {
          throw new Error(`Missing native Kotlin source: ${srcPath}`);
        }
        fs.copyFileSync(srcPath, path.join(dstKtDir, fileName));
      }

      const srcAssetsDir = path.join(projectRoot, "ondevice_assets");
      const dstAssetsDir = path.join(
        androidRoot,
        "app",
        "src",
        "main",
        "assets",
        "autovision",
      );
      fs.mkdirSync(dstAssetsDir, { recursive: true });
      for (const assetName of [
        "autovision_v2_core.onnx",
        "autovision_color_v2.onnx",
        "yolov8n_seg.onnx",
        "prototypes_fp32.bin",
        "autovision_meta.json",
      ]) {
        const srcAsset = path.join(srcAssetsDir, assetName);
        if (!fs.existsSync(srcAsset)) {
          throw new Error(`Missing on-device asset: ${srcAsset}`);
        }
        fs.copyFileSync(srcAsset, path.join(dstAssetsDir, assetName));
      }

      const proguardPath = path.join(androidRoot, "app", "proguard-rules.pro");
      if (fs.existsSync(proguardPath)) {
        const keepRule =
          "-keep class com.autovision.mobile.** { *; }\n-keep class ai.onnxruntime.** { *; }\n";
        const currentPg = fs.readFileSync(proguardPath, "utf8");
        if (!currentPg.includes("com.autovision.mobile.**")) {
          fs.writeFileSync(proguardPath, `${currentPg.trimEnd()}\n${keepRule}`, "utf8");
        }
      }

      return mod;
    },
  ]);
}

module.exports = function withAutoVisionOnDevice(config) {
  config = withOnDeviceBuildGradle(config);
  config = withCleartextManifest(config);
  config = withOnDeviceMainApplication(config);
  config = withOnDeviceAssetsAndSources(config);
  return config;
};
