package com.autovision.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class AutoVisionOnDeviceModule(
    private val reactContext: ReactApplicationContext,
) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "AutoVisionOnDevice"

    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val loadLock = Any()

    @Volatile
    private var engine: LoadedEngine? = null

    private data class LoadedEngine(
        val env: OrtEnvironment,
        val coreSession: OrtSession,
        val colorSession: OrtSession,
        val yoloSession: OrtSession,
        val meta: EngineMeta,
        val prototypes: FloatArray,
    )

    private data class EngineMeta(
        val modelVersion: String,
        val yearMin: Int,
        val yearMax: Int,
        val modelTemperature: Float,
        val cosfaceScale: Float,
        val numPrototypes: Int,
        val embedDim: Int,
        val numUniquePrototypeClasses: Int,
        val makeClasses: List<String>,
        val modelClasses: List<String>,
        val bodyClasses: List<String>,
        val colorClasses: List<String>,
        val colorHex: Map<String, String>,
        val colorGroups: Map<String, String>,
        val makeDisplayNames: Map<String, String>,
        val modelDisplayNames: Map<String, String>,
        val protoClasses: List<String>,
        val protoMakes: List<String>,
        val protoBodies: List<String>,
        val protoSplitTypes: List<String>,
        val modelMakeIndices: IntArray,
    )

    private data class DetectedBox(
        val x1: Int,
        val y1: Int,
        val x2: Int,
        val y2: Int,
        val confidence: Float,
        val areaRatio: Float,
        val detectedByYolo: Boolean,
        val maskCoeffs: FloatArray?,
    )

    private data class ProtoCandidate(
        val canonicalClass: String,
        val displayName: String,
        val make: String,
        val bodyType: String,
        val similarity: Float,
        val isFewShot: Boolean,
    )

    private data class ScoredLabel(
        val label: String,
        val displayName: String,
        val confidence: Float,
    )

    override fun invalidate() {
        super.invalidate()
        synchronized(loadLock) {
            val current = engine
            engine = null
            if (current != null) {
                runCatching { current.coreSession.close() }
                runCatching { current.colorSession.close() }
                runCatching { current.yoloSession.close() }
            }
        }
        executor.shutdown()
    }

    @ReactMethod
    fun getHealth(promise: Promise) {
        executor.execute {
            try {
                val loaded = ensureEngineLoaded()
                val m = loaded.meta
                val resp = JSONObject().apply {
                    put("status", "ok")
                    put("model_version", m.modelVersion)
                    put("device", "On-device")
                    put("ready", true)
                    put("num_makes", m.makeClasses.size)
                    put("num_models", m.modelClasses.size)
                    put("num_prototype_classes", m.numUniquePrototypeClasses)
                    put("num_bodies", m.bodyClasses.size)
                    put("num_colors", m.colorClasses.size)
                    put("gallery_size", 0)
                    put("yolo_available", true)
                }
                promise.resolve(resp.toString())
            } catch (t: Throwable) {
                promise.reject("ONDEVICE_HEALTH_ERROR", t.message ?: "Failed to initialize on-device engine", t)
            }
        }
    }

    @ReactMethod
    fun analyzeImage(imageUri: String, useTta: Boolean, promise: Promise) {
        executor.execute {
            var srcBitmap: Bitmap? = null
            try {
                val tStart = System.nanoTime()
                val loaded = ensureEngineLoaded()
                srcBitmap = loadOrientedBitmap(imageUri)
                val imgW = srcBitmap.width
                val imgH = srcBitmap.height
                val tDecode = System.nanoTime()

                val detOut = detectVehicles(loaded, srcBitmap)
                val boxes = detOut.boxes
                val tDetect = System.nanoTime()

                var classifyNs = 0L
                var retrieveNs = 0L
                val vehiclesArr = JSONArray()

                for ((idx, box) in boxes.withIndex()) {
                    var cropBmp: Bitmap? = null
                    var resized224: Bitmap? = null
                    try {
                        val tc0 = System.nanoTime()
                        val cw = max(1, box.x2 - box.x1)
                        val ch = max(1, box.y2 - box.y1)
                        cropBmp = Bitmap.createBitmap(srcBitmap, box.x1, box.y1, cw, ch)
                        resized224 = Bitmap.createScaledBitmap(cropBmp, INPUT_SIZE, INPUT_SIZE, true)

                        val cropPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
                        resized224.getPixels(cropPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

                        val mask224 = buildVehicleMask224(
                            box = box,
                            protoMasks = detOut.protoMasks,
                            scale = detOut.scale,
                            padX = detOut.padX,
                            padY = detOut.padY,
                        )

                        val coreOut = runCoreClassifier(loaded, cropPixels, useTta)
                        val colorProbs = runColorExpert(loaded, cropPixels, mask224, useTta)
                        val tc1 = System.nanoTime()
                        classifyNs += (tc1 - tc0)

                        val tr0 = System.nanoTime()
                        val vehicleJson = buildVehicleJson(
                            idx = idx,
                            box = box,
                            imgW = imgW,
                            imgH = imgH,
                            coreOut = coreOut,
                            colorProbs = colorProbs,
                            loaded = loaded,
                        )
                        val tr1 = System.nanoTime()
                        retrieveNs += (tr1 - tr0)

                        vehiclesArr.put(vehicleJson)
                    } finally {
                        resized224?.recycle()
                        cropBmp?.recycle()
                    }
                }

                val tEnd = System.nanoTime()
                val timings = JSONObject().apply {
                    put("decode_ms", round2((tDecode - tStart) / 1e6))
                    put("detect_ms", round2((tDetect - tDecode) / 1e6))
                    put("classify_ms", round2(classifyNs / 1e6))
                    put("retrieve_ms", round2(retrieveNs / 1e6))
                    put("total_ms", round2((tEnd - tStart) / 1e6))
                }

                val response = JSONObject().apply {
                    put("model_version", loaded.meta.modelVersion)
                    put("device", "On-device")
                    put("image_size", JSONObject().apply {
                        put("width", imgW)
                        put("height", imgH)
                    })
                    put("num_vehicles", vehiclesArr.length())
                    put("primary_vehicle_index", 0)
                    put("vehicles", vehiclesArr)
                    put("timings_ms", timings)
                }
                promise.resolve(response.toString())
            } catch (t: Throwable) {
                promise.reject("ONDEVICE_ERROR", t.message ?: "On-device vehicle analysis failed", t)
            } finally {
                srcBitmap?.recycle()
            }
        }
    }

    private fun ensureEngineLoaded(): LoadedEngine {
        val current = engine
        if (current != null) {
            return current
        }
        synchronized(loadLock) {
            val recheck = engine
            if (recheck != null) {
                return recheck
            }

            val modelsDir = File(reactContext.filesDir, "autovision_models")
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }

            val assetFiles = listOf(
                CORE_MODEL_FILE,
                COLOR_MODEL_FILE,
                YOLO_MODEL_FILE,
                PROTOTYPES_FILE,
                META_FILE,
            )
            for (name in assetFiles) {
                stageAssetIfNeeded(name, File(modelsDir, name))
            }

            val metaJson = JSONObject(File(modelsDir, META_FILE).readText(Charsets.UTF_8))
            val meta = parseMeta(metaJson)

            val protoBytes = File(modelsDir, PROTOTYPES_FILE).readBytes()
            val expectedFloats = meta.numPrototypes * meta.embedDim
            val protos = FloatArray(expectedFloats)
            ByteBuffer.wrap(protoBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(protos)

            val env = OrtEnvironment.getEnvironment()
            val threads = min(4, max(1, Runtime.getRuntime().availableProcessors()))
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            val coreSession = env.createSession(File(modelsDir, CORE_MODEL_FILE).absolutePath, opts)
            val colorSession = env.createSession(File(modelsDir, COLOR_MODEL_FILE).absolutePath, opts)
            val yoloSession = env.createSession(File(modelsDir, YOLO_MODEL_FILE).absolutePath, opts)

            val created = LoadedEngine(
                env = env,
                coreSession = coreSession,
                colorSession = colorSession,
                yoloSession = yoloSession,
                meta = meta,
                prototypes = protos,
            )
            engine = created
            return created
        }
    }

    private fun stageAssetIfNeeded(assetName: String, dst: File) {
        val assetPath = "$ASSET_SUBDIR/$assetName"
        val expectedLength = runCatching {
            reactContext.assets.openFd(assetPath).use { it.length }
        }.getOrDefault(-1L)

        if (dst.isFile) {
            if (expectedLength > 0 && dst.length() == expectedLength) {
                return
            }
            if (expectedLength <= 0 && dst.length() > 0) {
                return
            }
        }

        val tmp = File(dst.parentFile, "$assetName.tmp")
        reactContext.assets.open(assetPath).use { input ->
            FileOutputStream(tmp).use { output ->
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    output.write(buf, 0, n)
                }
                output.flush()
            }
        }
        if (dst.exists()) {
            dst.delete()
        }
        if (!tmp.renameTo(dst)) {
            throw IllegalStateException("Failed to stage asset $assetPath to ${dst.absolutePath}")
        }
    }

    private fun parseMeta(obj: JSONObject): EngineMeta {
        val makeClasses = obj.getJSONArray("make_classes").toStringList()
        val modelClasses = obj.getJSONArray("model_classes").toStringList()
        val makeToIdx = HashMap<String, Int>(makeClasses.size * 2)
        for ((i, m) in makeClasses.withIndex()) {
            makeToIdx[m] = i
        }
        val modelMakeIndices = IntArray(modelClasses.size) { idx ->
            val cls = modelClasses[idx]
            val mk = cls.substringBefore("__", "")
            makeToIdx[mk] ?: -1
        }
        return EngineMeta(
            modelVersion = obj.optString("model_version", "autovision-v2-ondevice"),
            yearMin = obj.optInt("year_min", 1950),
            yearMax = obj.optInt("year_max", 2026),
            modelTemperature = obj.optDouble("model_temperature", 2.15).toFloat(),
            cosfaceScale = obj.optDouble("cosface_scale", 30.0).toFloat(),
            numPrototypes = obj.getInt("num_prototypes"),
            embedDim = obj.getInt("embed_dim"),
            numUniquePrototypeClasses = obj.getInt("num_unique_prototype_classes"),
            makeClasses = makeClasses,
            modelClasses = modelClasses,
            bodyClasses = obj.getJSONArray("body_classes").toStringList(),
            colorClasses = obj.getJSONArray("color_classes").toStringList(),
            colorHex = obj.getJSONObject("color_hex").toStringMap(),
            colorGroups = obj.getJSONObject("color_groups").toStringMap(),
            makeDisplayNames = obj.getJSONObject("make_display_names").toStringMap(),
            modelDisplayNames = obj.getJSONObject("model_display_names").toStringMap(),
            protoClasses = obj.getJSONArray("proto_classes").toStringList(),
            protoMakes = obj.getJSONArray("proto_makes").toStringList(),
            protoBodies = obj.getJSONArray("proto_bodies").toStringList(),
            protoSplitTypes = obj.getJSONArray("proto_split_types").toStringList(),
            modelMakeIndices = modelMakeIndices,
        )
    }

    private fun openImageStream(uriString: String): InputStream {
        val parsed = Uri.parse(uriString)
        return when (parsed.scheme?.lowercase()) {
            ContentResolver.SCHEME_CONTENT -> reactContext.contentResolver.openInputStream(parsed)
                ?: throw IllegalArgumentException("Could not open content URI: $uriString")
            ContentResolver.SCHEME_FILE -> FileInputStream(File(requireNotNull(parsed.path) { "Empty file path" }))
            null -> FileInputStream(File(uriString))
            else -> reactContext.contentResolver.openInputStream(parsed)
                ?: throw IllegalArgumentException("Unsupported image URI: $uriString")
        }
    }

    private fun loadOrientedBitmap(imageUri: String): Bitmap {
        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        openImageStream(imageUri).use { BitmapFactory.decodeStream(it, null, boundsOpts) }
        val rawW = boundsOpts.outWidth
        val rawH = boundsOpts.outHeight
        require(rawW > 0 && rawH > 0) { "Failed to decode image dimensions" }

        var sampleSize = 1
        while (max(rawW / sampleSize, rawH / sampleSize) > MAX_IMAGE_DIM) {
            sampleSize *= 2
        }

        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = openImageStream(imageUri).use { BitmapFactory.decodeStream(it, null, decodeOpts) }
            ?: throw IllegalArgumentException("Failed to decode image bitmap")

        val orientation = runCatching {
            openImageStream(imageUri).use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            }
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)

        val matrix = Matrix()
        var needsTransform = true
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.setRotate(180f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
            else -> needsTransform = false
        }

        val oriented = if (needsTransform) {
            val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
            if (rotated !== decoded) {
                decoded.recycle()
            }
            rotated
        } else {
            decoded
        }

        val maxSide = max(oriented.width, oriented.height)
        if (maxSide > MAX_IMAGE_DIM) {
            val scale = MAX_IMAGE_DIM.toFloat() / maxSide.toFloat()
            val nw = max(1, (oriented.width * scale).roundToInt())
            val nh = max(1, (oriented.height * scale).roundToInt())
            val scaled = Bitmap.createScaledBitmap(oriented, nw, nh, true)
            if (scaled !== oriented) {
                oriented.recycle()
            }
            return scaled
        }
        return oriented
    }

    private data class DetectionOutput(
        val boxes: List<DetectedBox>,
        val protoMasks: Array<Array<FloatArray>>?,
        val scale: Float,
        val padX: Int,
        val padY: Int,
    )

    private fun detectVehicles(loaded: LoadedEngine, src: Bitmap): DetectionOutput {
        val imgW = src.width
        val imgH = src.height
        val imgArea = max(1f, (imgW * imgH).toFloat())

        val scale = min(YOLO_SIZE.toFloat() / max(imgW, 1), YOLO_SIZE.toFloat() / max(imgH, 1))
        val nw = max(1, (imgW * scale).roundToInt())
        val nh = max(1, (imgH * scale).roundToInt())
        val padX = (YOLO_SIZE - nw) / 2
        val padY = (YOLO_SIZE - nh) / 2

        val letterbox = Bitmap.createBitmap(YOLO_SIZE, YOLO_SIZE, Bitmap.Config.ARGB_8888)
        val nchw = FloatArray(3 * YOLO_SIZE * YOLO_SIZE)
        try {
            val canvas = Canvas(letterbox)
            canvas.drawColor(Color.rgb(114, 114, 114))
            val paint = Paint(Paint.FILTER_BITMAP_FLAG)
            canvas.drawBitmap(src, null, Rect(padX, padY, padX + nw, padY + nh), paint)

            val pixels = IntArray(YOLO_SIZE * YOLO_SIZE)
            letterbox.getPixels(pixels, 0, YOLO_SIZE, 0, 0, YOLO_SIZE, YOLO_SIZE)
            val plane = YOLO_SIZE * YOLO_SIZE
            for (i in pixels.indices) {
                val px = pixels[i]
                nchw[i] = ((px ushr 16) and 0xFF) / 255f
                nchw[plane + i] = ((px ushr 8) and 0xFF) / 255f
                nchw[2 * plane + i] = (px and 0xFF) / 255f
            }
        } finally {
            letterbox.recycle()
        }

        val rawCandidates = ArrayList<DetectedBox>()
        var protoMasks: Array<Array<FloatArray>>? = null

        OnnxTensor.createTensor(
            loaded.env,
            FloatBuffer.wrap(nchw),
            longArrayOf(1L, 3L, YOLO_SIZE.toLong(), YOLO_SIZE.toLong()),
        ).use { inputTensor ->
            loaded.yoloSession.run(mapOf(YOLO_INPUT_NAME to inputTensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out0 = (result[0].value as Array<Array<FloatArray>>)[0] // [116, 8400]
                @Suppress("UNCHECKED_CAST")
                val out1 = (result[1].value as Array<Array<Array<FloatArray>>>)[0] // [32, 160, 160]
                protoMasks = out1

                val numAnchors = out0[0].size
                for (a in 0 until numAnchors) {
                    val sCar = out0[4 + 2][a]
                    val sBus = out0[4 + 5][a]
                    val sTruck = out0[4 + 7][a]
                    val bestScore = max(sCar, max(sBus, sTruck))
                    if (bestScore < YOLO_CONF) continue

                    val cx = out0[0][a]
                    val cy = out0[1][a]
                    val bw640 = out0[2][a]
                    val bh640 = out0[3][a]

                    val bx1 = ((cx - 0.5f * bw640) - padX) / scale
                    val by1 = ((cy - 0.5f * bh640) - padY) / scale
                    val bx2 = ((cx + 0.5f * bw640) - padX) / scale
                    val by2 = ((cy + 0.5f * bh640) - padY) / scale

                    val clX1 = bx1.coerceIn(0f, imgW.toFloat())
                    val clY1 = by1.coerceIn(0f, imgH.toFloat())
                    val clX2 = bx2.coerceIn(0f, imgW.toFloat())
                    val clY2 = by2.coerceIn(0f, imgH.toFloat())
                    val rw = max(0f, clX2 - clX1)
                    val rh = max(0f, clY2 - clY1)
                    val ratio = ((rw * rh) / imgArea).coerceIn(0f, 1f)
                    if (ratio < YOLO_MIN_AREA_RATIO) continue

                    val padW = YOLO_PAD * rw
                    val padH = YOLO_PAD * rh
                    val x1 = max(0, floor(clX1 - padW).toInt())
                    val y1 = max(0, floor(clY1 - padH).toInt())
                    val x2 = min(imgW, max(x1 + 1, ceil(clX2 + padW).toInt()))
                    val y2 = min(imgH, max(y1 + 1, ceil(clY2 + padH).toInt()))

                    val coeffs = FloatArray(MASK_DIM) { k -> out0[84 + k][a] }
                    rawCandidates.add(
                        DetectedBox(
                            x1 = x1,
                            y1 = y1,
                            x2 = x2,
                            y2 = y2,
                            confidence = bestScore.coerceIn(0f, 1f),
                            areaRatio = ratio,
                            detectedByYolo = true,
                            maskCoeffs = coeffs,
                        ),
                    )
                }
            }
        }

        if (rawCandidates.isNotEmpty()) {
            rawCandidates.sortByDescending { it.confidence }
            val kept = ArrayList<DetectedBox>()
            for (cand in rawCandidates) {
                if (kept.none { computeIoU(it, cand) >= YOLO_NMS_IOU }) {
                    kept.add(cand)
                }
            }

            kept.sortWith(
                compareBy<DetectedBox> { b ->
                    val cx = 0.5f * (b.x1 + b.x2) / max(1, imgW)
                    val cy = 0.5f * (b.y1 + b.y2) / max(1, imgH)
                    val dist = hypot((cx - 0.5f).toDouble(), (cy - 0.5f).toDouble()).toFloat() / 0.70710678f
                    val centerProx = max(0f, 1f - dist)
                    -(b.areaRatio * (0.70f + 0.30f * centerProx))
                }.thenByDescending { it.confidence },
            )

            return DetectionOutput(
                boxes = kept.take(MAX_VEHICLES),
                protoMasks = protoMasks,
                scale = scale,
                padX = padX,
                padY = padY,
            )
        }

        val mw = (imgW * 0.075f).toInt()
        val mh = (imgH * 0.075f).toInt()
        val x1 = min(mw, max(imgW - 1, 0))
        val y1 = min(mh, max(imgH - 1, 0))
        val x2 = max(x1 + 1, imgW - mw)
        val y2 = max(y1 + 1, imgH - mh)
        val fbRatio = (((x2 - x1) * (y2 - y1)).toFloat() / imgArea).coerceIn(0f, 1f)
        return DetectionOutput(
            boxes = listOf(
                DetectedBox(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    confidence = 0f,
                    areaRatio = fbRatio,
                    detectedByYolo = false,
                    maskCoeffs = null,
                ),
            ),
            protoMasks = null,
            scale = scale,
            padX = padX,
            padY = padY,
        )
    }

    private fun computeIoU(a: DetectedBox, b: DetectedBox): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val iw = max(0, ix2 - ix1)
        val ih = max(0, iy2 - iy1)
        val inter = (iw * ih).toFloat()
        if (inter <= 0f) return 0f
        val areaA = ((a.x2 - a.x1) * (a.y2 - a.y1)).toFloat()
        val areaB = ((b.x2 - b.x1) * (b.y2 - b.y1)).toFloat()
        return inter / max(1e-6f, areaA + areaB - inter)
    }

    private fun buildVehicleMask224(
        box: DetectedBox,
        protoMasks: Array<Array<FloatArray>>?,
        scale: Float,
        padX: Int,
        padY: Int,
    ): BooleanArray {
        val total = INPUT_SIZE * INPUT_SIZE
        val coeffs = box.maskCoeffs
        if (box.detectedByYolo && coeffs != null && protoMasks != null) {
            val mask = BooleanArray(total)
            val protoH = protoMasks[0].size
            val protoW = protoMasks[0][0].size
            val strideRatio = protoW.toFloat() / YOLO_SIZE.toFloat() // 160/640 = 0.25
            val boxW = max(1, box.x2 - box.x1).toFloat()
            val boxH = max(1, box.y2 - box.y1).toFloat()

            var activeCount = 0
            for (y224 in 0 until INPUT_SIZE) {
                val oy = box.y1 + ((y224 + 0.5f) / INPUT_SIZE) * boxH
                val ly = oy * scale + padY
                val py = (ly * strideRatio - 0.5f).coerceIn(0f, (protoH - 1).toFloat())
                val y0 = floor(py).toInt()
                val y1 = min(protoH - 1, y0 + 1)
                val wy1 = py - y0
                val wy0 = 1f - wy1

                for (x224 in 0 until INPUT_SIZE) {
                    val ox = box.x1 + ((x224 + 0.5f) / INPUT_SIZE) * boxW
                    val lx = ox * scale + padX
                    val px = (lx * strideRatio - 0.5f).coerceIn(0f, (protoW - 1).toFloat())
                    val x0 = floor(px).toInt()
                    val x1 = min(protoW - 1, x0 + 1)
                    val wx1 = px - x0
                    val wx0 = 1f - wx1

                    var z = 0f
                    for (k in 0 until MASK_DIM) {
                        val pk = protoMasks[k]
                        val pVal = (pk[y0][x0] * wx0 + pk[y0][x1] * wx1) * wy0 +
                            (pk[y1][x0] * wx0 + pk[y1][x1] * wx1) * wy1
                        z += coeffs[k] * pVal
                    }
                    if (z >= 0f) {
                        mask[y224 * INPUT_SIZE + x224] = true
                        activeCount++
                    }
                }
            }

            if (activeCount >= 64) {
                val eroded = erodeMask224(mask, erodePx = 2)
                var erodedCount = 0
                for (b in eroded) {
                    if (b) erodedCount++
                }
                if (erodedCount >= 32) {
                    return eroded
                }
            }
        }

        // Centered interior car box fallback [12%..88%]
        val fb = BooleanArray(total)
        val yLo = (0.12f * INPUT_SIZE).toInt()
        val yHi = (0.88f * INPUT_SIZE).toInt()
        val xLo = (0.12f * INPUT_SIZE).toInt()
        val xHi = (0.88f * INPUT_SIZE).toInt()
        for (y in yLo until yHi) {
            for (x in xLo until xHi) {
                fb[y * INPUT_SIZE + x] = true
            }
        }
        return fb
    }

    private fun erodeMask224(src: BooleanArray, erodePx: Int): BooleanArray {
        var curr = src.copyOf()
        for (iter in 0 until erodePx) {
            val next = BooleanArray(src.size)
            for (y in 1 until INPUT_SIZE - 1) {
                val row = y * INPUT_SIZE
                for (x in 1 until INPUT_SIZE - 1) {
                    val idx = row + x
                    next[idx] = curr[idx] &&
                        curr[idx - INPUT_SIZE] &&
                        curr[idx + INPUT_SIZE] &&
                        curr[idx - 1] &&
                        curr[idx + 1]
                }
            }
            curr = next
        }
        return curr
    }

    private data class CoreOutput(
        val embedding: FloatArray,
        val makeProbs: FloatArray,
        val modelLogits: FloatArray,
        val bodyProbs: FloatArray,
        val yearLogits: FloatArray,
    )

    private fun pixelsToNchwNormalized(
        pixels: IntArray,
        mean: FloatArray,
        std: FloatArray,
        flipHorizontal: Boolean,
        mask224: BooleanArray?,
    ): FloatArray {
        val plane = INPUT_SIZE * INPUT_SIZE
        val out = FloatArray(3 * plane)
        for (y in 0 until INPUT_SIZE) {
            val rowOff = y * INPUT_SIZE
            for (x in 0 until INPUT_SIZE) {
                val srcX = if (flipHorizontal) (INPUT_SIZE - 1 - x) else x
                val srcIdx = rowOff + srcX
                val dstIdx = rowOff + x

                val rByte: Int
                val gByte: Int
                val bByte: Int
                if (mask224 != null && !mask224[srcIdx]) {
                    rByte = 128
                    gByte = 128
                    bByte = 128
                } else {
                    val px = pixels[srcIdx]
                    rByte = (px ushr 16) and 0xFF
                    gByte = (px ushr 8) and 0xFF
                    bByte = px and 0xFF
                }

                out[dstIdx] = (rByte / 255f - mean[0]) / std[0]
                out[plane + dstIdx] = (gByte / 255f - mean[1]) / std[1]
                out[2 * plane + dstIdx] = (bByte / 255f - mean[2]) / std[2]
            }
        }
        return out
    }

    private fun runCoreSingle(loaded: LoadedEngine, inputNchw: FloatArray): CoreOutput {
        OnnxTensor.createTensor(
            loaded.env,
            FloatBuffer.wrap(inputNchw),
            longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
        ).use { tensor ->
            loaded.coreSession.run(mapOf(CORE_INPUT_NAME to tensor)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val emb = (res[0].value as Array<FloatArray>)[0].copyOf()
                @Suppress("UNCHECKED_CAST")
                val mkL = (res[1].value as Array<FloatArray>)[0].copyOf()
                @Suppress("UNCHECKED_CAST")
                val mdL = (res[2].value as Array<FloatArray>)[0].copyOf()
                @Suppress("UNCHECKED_CAST")
                val bdL = (res[3].value as Array<FloatArray>)[0].copyOf()
                @Suppress("UNCHECKED_CAST")
                val yrL = (res[4].value as Array<FloatArray>)[0].copyOf()
                return CoreOutput(emb, mkL, mdL, bdL, yrL)
            }
        }
    }

    private fun runCoreClassifier(
        loaded: LoadedEngine,
        pixels: IntArray,
        useTta: Boolean,
    ): CoreOutput {
        val x0 = pixelsToNchwNormalized(pixels, CLIP_MEAN, CLIP_STD, flipHorizontal = false, mask224 = null)
        val out0 = runCoreSingle(loaded, x0)
        if (!useTta) {
            l2NormalizeInPlace(out0.embedding)
            return CoreOutput(
                embedding = out0.embedding,
                makeProbs = softmax(out0.makeProbs),
                modelLogits = out0.modelLogits,
                bodyProbs = softmax(out0.bodyProbs),
                yearLogits = out0.yearLogits,
            )
        }

        val x1 = pixelsToNchwNormalized(pixels, CLIP_MEAN, CLIP_STD, flipHorizontal = true, mask224 = null)
        val out1 = runCoreSingle(loaded, x1)

        val emb = FloatArray(out0.embedding.size) { i -> out0.embedding[i] + out1.embedding[i] }
        l2NormalizeInPlace(emb)
        val mkL = FloatArray(out0.makeProbs.size) { i -> 0.5f * (out0.makeProbs[i] + out1.makeProbs[i]) }
        val mdL = FloatArray(out0.modelLogits.size) { i -> 0.5f * (out0.modelLogits[i] + out1.modelLogits[i]) }
        val bdL = FloatArray(out0.bodyProbs.size) { i -> 0.5f * (out0.bodyProbs[i] + out1.bodyProbs[i]) }
        val yrL = FloatArray(out0.yearLogits.size) { i -> 0.5f * (out0.yearLogits[i] + out1.yearLogits[i]) }

        return CoreOutput(
            embedding = emb,
            makeProbs = softmax(mkL),
            modelLogits = mdL,
            bodyProbs = softmax(bdL),
            yearLogits = yrL,
        )
    }

    private fun runColorExpert(
        loaded: LoadedEngine,
        pixels: IntArray,
        mask224: BooleanArray,
        useTta: Boolean,
    ): FloatArray {
        val labFeats = extractMaskedLabFeatures(pixels, mask224)
        val x0 = pixelsToNchwNormalized(pixels, IMAGENET_MEAN, IMAGENET_STD, flipHorizontal = false, mask224 = mask224)

        val logits0 = runColorSingle(loaded, x0, labFeats)
        if (!useTta) {
            return softmax(logits0)
        }
        val x1 = pixelsToNchwNormalized(pixels, IMAGENET_MEAN, IMAGENET_STD, flipHorizontal = true, mask224 = mask224)
        val logits1 = runColorSingle(loaded, x1, labFeats)
        val avg = FloatArray(logits0.size) { i -> 0.5f * (logits0[i] + logits1[i]) }
        return softmax(avg)
    }

    private fun runColorSingle(
        loaded: LoadedEngine,
        xNchw: FloatArray,
        labFeats: FloatArray,
    ): FloatArray {
        OnnxTensor.createTensor(
            loaded.env,
            FloatBuffer.wrap(xNchw),
            longArrayOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
        ).use { imgTensor ->
            OnnxTensor.createTensor(
                loaded.env,
                FloatBuffer.wrap(labFeats),
                longArrayOf(1L, 16L),
            ).use { labTensor ->
                loaded.colorSession.run(
                    mapOf("image" to imgTensor, "lab_feats" to labTensor),
                ).use { res ->
                    @Suppress("UNCHECKED_CAST")
                    return (res[0].value as Array<FloatArray>)[0].copyOf()
                }
            }
        }
    }

    private fun extractMaskedLabFeatures(
        pixels: IntArray,
        mask224: BooleanArray,
    ): FloatArray {
        var count = 0
        for (b in mask224) {
            if (b) count++
        }
        val useAll = count == 0
        val effectiveCount = if (useAll) pixels.size else count
        val stride = if (effectiveCount > 4096) max(1, effectiveCount / 4096) else 1

        val sampled = ArrayList<Int>(min(effectiveCount, 4096) + 16)
        var seen = 0
        for (i in pixels.indices) {
            if (!useAll && !mask224[i]) continue
            if (seen % stride == 0) {
                sampled.add(pixels[i])
            }
            seen++
        }

        val n = max(1, sampled.size)
        val lArr = FloatArray(n)
        val aArr = FloatArray(n)
        val bArr = FloatArray(n)
        val cArr = FloatArray(n)

        var lSum = 0.0
        var darkAchrom = 0
        var brightAchrom = 0
        var warmBrown = 0

        val eps = 216f / 24389f
        val kappa = 24389f / 27f

        for (i in 0 until n) {
            val px = sampled[i]
            val r01 = ((px ushr 16) and 0xFF) / 255f
            val g01 = ((px ushr 8) and 0xFF) / 255f
            val b01 = (px and 0xFF) / 255f

            val lr = if (r01 <= 0.04045f) r01 / 12.92f else ((r01 + 0.055f) / 1.055f).pow(2.4f)
            val lg = if (g01 <= 0.04045f) g01 / 12.92f else ((g01 + 0.055f) / 1.055f).pow(2.4f)
            val lb = if (b01 <= 0.04045f) b01 / 12.92f else ((b01 + 0.055f) / 1.055f).pow(2.4f)

            val x = (0.4124564f * lr + 0.3575761f * lg + 0.1804375f * lb) / 0.95047f
            val y = 0.2126729f * lr + 0.7151522f * lg + 0.0721750f * lb
            val z = (0.0193339f * lr + 0.1191920f * lg + 0.9503041f * lb) / 1.08883f

            val fx = if (x > eps) cbrt(max(0f, x)) else (kappa * x + 16f) / 116f
            val fy = if (y > eps) cbrt(max(0f, y)) else (kappa * y + 16f) / 116f
            val fz = if (z > eps) cbrt(max(0f, z)) else (kappa * z + 16f) / 116f

            val lStar = 116f * fy - 16f
            val aStar = 500f * (fx - fy)
            val bStar = 200f * (fy - fz)
            val chroma = sqrt(aStar * aStar + bStar * bStar)

            lArr[i] = lStar
            aArr[i] = aStar
            bArr[i] = bStar
            cArr[i] = chroma
            lSum += lStar

            if (lStar < 28f && chroma < 10f) darkAchrom++
            if (lStar > 76f && chroma < 9f) brightAchrom++
            if (lStar in 18f..62f && aStar > 4f && bStar > 6f && chroma > 12f) warmBrown++
        }

        val lMean = (lSum / n).toFloat()
        var varSum = 0.0
        for (v in lArr) {
            val d = v - lMean
            varSum += d * d
        }
        val lStd = sqrt(varSum / n).toFloat()

        lArr.sort()
        aArr.sort()
        bArr.sort()
        cArr.sort()

        return floatArrayOf(
            sortedPercentile(lArr, 15f) / 100f,
            sortedPercentile(lArr, 50f) / 100f,
            sortedPercentile(lArr, 85f) / 100f,
            lStd / 50f,
            sortedPercentile(aArr, 25f) / 60f,
            sortedPercentile(aArr, 50f) / 60f,
            sortedPercentile(aArr, 75f) / 60f,
            sortedPercentile(bArr, 25f) / 60f,
            sortedPercentile(bArr, 50f) / 60f,
            sortedPercentile(bArr, 75f) / 60f,
            sortedPercentile(cArr, 25f) / 60f,
            sortedPercentile(cArr, 50f) / 60f,
            sortedPercentile(cArr, 75f) / 60f,
            darkAchrom.toFloat() / n.toFloat(),
            brightAchrom.toFloat() / n.toFloat(),
            warmBrown.toFloat() / n.toFloat(),
        )
    }

    private fun sortedPercentile(sorted: FloatArray, pct: Float): Float {
        if (sorted.isEmpty()) return 0f
        if (sorted.size == 1) return sorted[0]
        val pos = (pct / 100f) * (sorted.size - 1)
        val lo = floor(pos).toInt()
        val hi = ceil(pos).toInt()
        val frac = pos - lo
        return sorted[lo] * (1f - frac) + sorted[hi] * frac
    }

    private fun buildVehicleJson(
        idx: Int,
        box: DetectedBox,
        imgW: Int,
        imgH: Int,
        coreOut: CoreOutput,
        colorProbs: FloatArray,
        loaded: LoadedEngine,
    ): JSONObject {
        val m = loaded.meta
        var mkTop3 = topKPredictions(
            probs = coreOut.makeProbs,
            labels = m.makeClasses,
            k = 3,
            useDisplay = true,
            makeDisplayNames = m.makeDisplayNames,
            modelDisplayNames = m.modelDisplayNames,
        )

        // Gentle make-consistency tie-breaker over calibrated model probabilities
        val temp = max(m.modelTemperature, 1e-3f)
        val adjustedModelProbs = if (mkTop3.isNotEmpty() && mkTop3[0].confidence >= 0.85f) {
            val scale = max(m.cosfaceScale, 1e-6f)
            val effScale = scale / temp
            val adjLogits = FloatArray(coreOut.modelLogits.size) { c ->
                val cosVal = coreOut.modelLogits[c] / scale
                val mkIdx = m.modelMakeIndices[c]
                val prior = if (mkIdx >= 0) coreOut.makeProbs[mkIdx] else 0f
                (cosVal + 0.025f * prior) * effScale
            }
            softmax(adjLogits)
        } else {
            val scaled = FloatArray(coreOut.modelLogits.size) { c -> coreOut.modelLogits[c] / temp }
            softmax(scaled)
        }

        var mdTop3 = topKPredictions(
            probs = adjustedModelProbs,
            labels = m.modelClasses,
            k = 3,
            useDisplay = true,
            makeDisplayNames = m.makeDisplayNames,
            modelDisplayNames = m.modelDisplayNames,
        )

        val protoResult = matchPrototypesDetailed(coreOut.embedding, loaded)
        val fewShot = protoResult.top1
        val bestOpen = protoResult.bestOpenCandidate
        if (bestOpen != null &&
            bestOpen.similarity >= 0.56f &&
            bestOpen.similarity >= protoResult.bestClosedSim + 0.035f
        ) {
            val promoConf = bestOpen.similarity.coerceIn(0.50f, 0.95f)
            val promoted = ScoredLabel(
                label = bestOpen.canonicalClass,
                displayName = bestOpen.displayName,
                confidence = promoConf,
            )
            mdTop3 = (listOf(promoted) + mdTop3.filter { it.label != bestOpen.canonicalClass }).take(3)

            if (bestOpen.make !in m.makeClasses || bestOpen.similarity >= protoResult.bestClosedSim + 0.05f) {
                val mkPromoted = ScoredLabel(
                    label = bestOpen.make,
                    displayName = formatCanonicalDisplayName(
                        bestOpen.make,
                        m.makeDisplayNames,
                        m.modelDisplayNames,
                    ),
                    confidence = promoConf,
                )
                mkTop3 = (listOf(mkPromoted) + mkTop3.filter { it.label != bestOpen.make }).take(3)
            }
        }

        val bdTop3 = topKPredictions(
            probs = coreOut.bodyProbs,
            labels = m.bodyClasses,
            k = 3,
            useDisplay = false,
            makeDisplayNames = m.makeDisplayNames,
            modelDisplayNames = m.modelDisplayNames,
        )
        val clTop3 = topKPredictions(
            probs = colorProbs,
            labels = m.colorClasses,
            k = 3,
            useDisplay = false,
            makeDisplayNames = m.makeDisplayNames,
            modelDisplayNames = m.modelDisplayNames,
        )

        // DLDL Year expected value and std over [yearMin, yearMax]
        val yrProbs = softmax(coreOut.yearLogits)
        var yrExpected = 0.0
        for (i in yrProbs.indices) {
            yrExpected += yrProbs[i] * (m.yearMin + i)
        }
        var yrVar = 0.0
        for (i in yrProbs.indices) {
            val d = (m.yearMin + i) - yrExpected
            yrVar += yrProbs[i] * d * d
        }
        val yrStd = sqrt(max(0.0, yrVar))
        val yrRound = yrExpected.roundToInt().coerceIn(m.yearMin, m.yearMax)
        val yrLow = max(m.yearMin, min(yrRound, floor(yrExpected - yrStd).toInt()))
        val yrHigh = min(m.yearMax, max(yrRound, ceil(yrExpected + yrStd).toInt()))

        val topColor = clTop3[0].label
        val attrs = JSONObject().apply {
            put("make", JSONObject().apply {
                put("label", mkTop3[0].label)
                put("display_name", mkTop3[0].displayName)
                put("confidence", round4(mkTop3[0].confidence))
                put("top3", topKToJsonArray(mkTop3))
            })
            put("model", JSONObject().apply {
                put("canonical_class", mdTop3[0].label)
                put("display_name", mdTop3[0].displayName)
                put("confidence", round4(mdTop3[0].confidence))
                put("top3", topKToJsonArray(mdTop3))
                if (fewShot != null) {
                    put("few_shot_match", JSONObject().apply {
                        put("canonical_class", fewShot.canonicalClass)
                        put("display_name", fewShot.displayName)
                        put("make", fewShot.make)
                        put("body_type", fewShot.bodyType)
                        put("similarity", round4(fewShot.similarity))
                        put("is_few_shot", fewShot.isFewShot)
                    })
                } else {
                    put("few_shot_match", JSONObject.NULL)
                }
            })
            put("year", JSONObject().apply {
                put("predicted", round1(yrExpected))
                put("predicted_rounded", yrRound)
                put("range_low", yrLow)
                put("range_high", yrHigh)
                put("std", round2(yrStd))
            })
            put("body_type", JSONObject().apply {
                put("label", bdTop3[0].label)
                put("display_name", bdTop3[0].displayName)
                put("confidence", round4(bdTop3[0].confidence))
                put("top3", topKToJsonArray(bdTop3))
            })
            put("color", JSONObject().apply {
                put("label", topColor)
                put("display_name", topColor)
                put("group", m.colorGroups[topColor] ?: topColor)
                put("hex", m.colorHex[topColor] ?: "#6B7280")
                put("confidence", round4(clTop3[0].confidence))
                put("top3", topKToJsonArray(clTop3))
            })
        }

        return JSONObject().apply {
            put("vehicle_index", idx)
            put("detected_by_yolo", box.detectedByYolo)
            put("detection_confidence", round4(box.confidence))
            put("area_ratio", round4(box.areaRatio))
            put("bbox_xyxy", JSONObject().apply {
                put("x1", box.x1.toDouble())
                put("y1", box.y1.toDouble())
                put("x2", box.x2.toDouble())
                put("y2", box.y2.toDouble())
            })
            put("bbox_norm", JSONObject().apply {
                put("x1", round4((box.x1.toFloat() / max(1, imgW)).coerceIn(0f, 1f)))
                put("y1", round4((box.y1.toFloat() / max(1, imgH)).coerceIn(0f, 1f)))
                put("x2", round4((box.x2.toFloat() / max(1, imgW)).coerceIn(0f, 1f)))
                put("y2", round4((box.y2.toFloat() / max(1, imgH)).coerceIn(0f, 1f)))
            })
            put("attributes", attrs)
            put("similar_vehicles", JSONArray())
        }
    }

    private data class ProtoMatchSummary(
        val top1: ProtoCandidate?,
        val bestClosedSim: Float,
        val bestOpenCandidate: ProtoCandidate?,
    )

    private fun matchPrototypesDetailed(emb: FloatArray, loaded: LoadedEngine): ProtoMatchSummary {
        val m = loaded.meta
        val n = m.numPrototypes
        val dim = m.embedDim
        if (n <= 0 || dim != emb.size) {
            return ProtoMatchSummary(null, -1f, null)
        }

        val q = emb.copyOf()
        l2NormalizeInPlace(q)

        var bestAllIdx = -1
        var bestAllSim = -Float.MAX_VALUE
        var bestClosedSim = -Float.MAX_VALUE
        var bestOpenIdx = -1
        var bestOpenSim = -Float.MAX_VALUE

        val protos = loaded.prototypes
        for (i in 0 until n) {
            val base = i * dim
            var dot = 0f
            for (d in 0 until dim) {
                dot += protos[base + d] * q[d]
            }
            if (dot > bestAllSim) {
                bestAllSim = dot
                bestAllIdx = i
            }
            if (m.protoSplitTypes[i] == "train") {
                if (dot > bestClosedSim) {
                    bestClosedSim = dot
                }
            } else {
                if (dot > bestOpenSim) {
                    bestOpenSim = dot
                    bestOpenIdx = i
                }
            }
        }

        val top1 = if (bestAllIdx >= 0) toProtoCandidate(bestAllIdx, bestAllSim, m) else null
        val openCand = if (bestOpenIdx >= 0) toProtoCandidate(bestOpenIdx, bestOpenSim, m) else null
        return ProtoMatchSummary(
            top1 = top1,
            bestClosedSim = if (bestClosedSim == -Float.MAX_VALUE) -1f else bestClosedSim,
            bestOpenCandidate = openCand,
        )
    }

    private fun toProtoCandidate(idx: Int, sim: Float, m: EngineMeta): ProtoCandidate {
        val cls = m.protoClasses[idx]
        return ProtoCandidate(
            canonicalClass = cls,
            displayName = formatCanonicalDisplayName(cls, m.makeDisplayNames, m.modelDisplayNames),
            make = m.protoMakes[idx],
            bodyType = m.protoBodies[idx],
            similarity = sim.coerceIn(-1f, 1f),
            isFewShot = m.protoSplitTypes[idx] == "few_shot" || m.protoSplitTypes[idx] == "eu_proto",
        )
    }

    private fun topKPredictions(
        probs: FloatArray,
        labels: List<String>,
        k: Int,
        useDisplay: Boolean,
        makeDisplayNames: Map<String, String>,
        modelDisplayNames: Map<String, String>,
    ): List<ScoredLabel> {
        val kEff = min(k, min(labels.size, probs.size))
        val indices = probs.indices.sortedByDescending { probs[it] }.take(kEff)
        return indices.map { idx ->
            val raw = labels[idx]
            val disp = if (useDisplay) {
                formatCanonicalDisplayName(raw, makeDisplayNames, modelDisplayNames)
            } else {
                raw
            }
            ScoredLabel(raw, disp, probs[idx].coerceIn(0f, 1f))
        }
    }

    private fun topKToJsonArray(items: List<ScoredLabel>): JSONArray {
        val arr = JSONArray()
        for (item in items) {
            arr.put(JSONObject().apply {
                put("label", item.label)
                put("display_name", item.displayName)
                put("confidence", round4(item.confidence))
            })
        }
        return arr
    }

    private fun formatCanonicalDisplayName(
        raw: String,
        makeDisplayNames: Map<String, String>,
        modelDisplayNames: Map<String, String>,
    ): String {
        val cleaned = raw.trim()
        if (cleaned.isEmpty()) return ""
        if ("__" in cleaned) {
            val mkSlug = cleaned.substringBefore("__")
            val mkD = makeDisplayNames[mkSlug]
            val mdD = modelDisplayNames[cleaned]
            if (!mkD.isNullOrEmpty() && !mdD.isNullOrEmpty()) return "$mkD $mdD".trim()
            if (!mdD.isNullOrEmpty()) return "${formatPart(mkSlug)} $mdD".trim()
            return formatDisplayName(cleaned)
        }
        return makeDisplayNames[cleaned] ?: formatDisplayName(cleaned)
    }

    private fun formatDisplayName(raw: String): String {
        val cleaned = raw.trim()
        if (cleaned.isEmpty()) return ""
        if ("__" in cleaned) {
            val brand = cleaned.substringBefore("__")
            val model = cleaned.substringAfter("__")
            return "${formatPart(brand)} ${formatPart(model)}".trim()
        }
        return formatPart(cleaned)
    }

    private fun formatPart(rawPart: String): String {
        val tokens = ArrayList<String>()
        for (word in rawPart.replace('_', ' ').split(Regex("\\s+"))) {
            if (word.isEmpty()) continue
            val lowerW = word.lowercase()
            val special = SPECIAL_DISPLAY_TOKENS[lowerW]
            if (special != null) {
                tokens.add(special)
                continue
            }
            val subparts = word.split('-').map { sub ->
                val lowSub = sub.lowercase()
                val subSpecial = SPECIAL_DISPLAY_TOKENS[lowSub]
                when {
                    subSpecial != null -> subSpecial
                    sub.any { it.isDigit() } && sub.any { it.isLetter() } -> sub.uppercase()
                    else -> sub.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
                }
            }
            tokens.add(subparts.joinToString("-"))
        }
        return tokens.joinToString(" ")
    }

    private fun l2NormalizeInPlace(v: FloatArray) {
        var normSq = 0f
        for (x in v) normSq += x * x
        val norm = max(1e-12f, sqrt(normSq))
        for (i in v.indices) v[i] /= norm
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxVal = -Float.MAX_VALUE
        for (v in logits) {
            if (v > maxVal) maxVal = v
        }
        val exps = FloatArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            val e = exp((logits[i] - maxVal).toDouble())
            exps[i] = e.toFloat()
            sum += e
        }
        val denom = max(1e-12, sum).toFloat()
        for (i in exps.indices) exps[i] /= denom
        return exps
    }

    private fun round1(v: Double): Double = (v * 10.0).roundToInt() / 10.0
    private fun round2(v: Double): Double = (v * 100.0).roundToInt() / 100.0
    private fun round4(v: Float): Double = (v * 10000.0).roundToInt() / 10000.0

    private fun JSONArray.toStringList(): List<String> =
        List(length()) { i -> getString(i) }

    private fun JSONObject.toStringMap(): Map<String, String> {
        val map = HashMap<String, String>(length() * 2)
        val it = keys()
        while (it.hasNext()) {
            val k = it.next()
            map[k] = optString(k, "")
        }
        return map
    }

    companion object {
        private const val ASSET_SUBDIR = "autovision"
        private const val CORE_MODEL_FILE = "autovision_v2_core.onnx"
        private const val COLOR_MODEL_FILE = "autovision_color_v2.onnx"
        private const val YOLO_MODEL_FILE = "yolov8n_seg.onnx"
        private const val PROTOTYPES_FILE = "prototypes_fp32.bin"
        private const val META_FILE = "autovision_meta.json"

        private const val CORE_INPUT_NAME = "image"
        private const val YOLO_INPUT_NAME = "images"

        private const val MAX_IMAGE_DIM = 2048
        private const val INPUT_SIZE = 224
        private const val YOLO_SIZE = 640
        private const val MASK_DIM = 32
        private const val YOLO_CONF = 0.25f
        private const val YOLO_MIN_AREA_RATIO = 0.04f
        private const val YOLO_PAD = 0.05f
        private const val YOLO_NMS_IOU = 0.45f
        private const val MAX_VEHICLES = 4

        private val CLIP_MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val CLIP_STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        private val SPECIAL_DISPLAY_TOKENS = mapOf(
            "bmw" to "BMW",
            "brz" to "BRZ",
            "cj2a" to "CJ-2A",
            "cj3b" to "CJ-3B",
            "cla" to "CLA",
            "clk" to "CLK",
            "cls" to "CLS",
            "cr-v" to "CR-V",
            "crv" to "CR-V",
            "fj40" to "FJ40",
            "fjcruiser" to "FJ Cruiser",
            "frs" to "FR-S",
            "gl" to "GL",
            "gla" to "GLA",
            "glk" to "GLK",
            "gt" to "GT",
            "gt6" to "GT6",
            "gti" to "GTI",
            "gto" to "GTO",
            "gtr" to "GT-R",
            "gtx" to "GTX",
            "iq" to "iQ",
            "mg" to "MG",
            "mgb" to "MGB",
            "mr2" to "MR2",
            "nv200" to "NV200",
            "pv444" to "PV444",
            "r32" to "R32",
            "rx7" to "RX-7",
            "rx8" to "RX-8",
            "sl" to "SL",
            "slk" to "SLK",
            "slr" to "SLR",
            "suv" to "SUV",
            "svx" to "SVX",
            "sx4" to "SX4",
            "tc" to "tC",
            "tr" to "TR",
            "wrx" to "WRX",
            "xa" to "xA",
            "xb" to "xB",
            "xc60" to "XC60",
            "xc70" to "XC70",
            "xc90" to "XC90",
            "xd" to "xD",
            "xl7" to "XL7",
        )
    }
}
