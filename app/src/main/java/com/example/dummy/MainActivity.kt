package com.example.dummy

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import android.hardware.camera2.CaptureRequest
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.switchmaterial.SwitchMaterial
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import com.google.android.material.button.MaterialButton
import android.content.Intent
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.util.HashMap // Import HashMap for scores

class MainActivity : BaseDrawerActivity() {

    private val TAG = "CameraAI_API"
    private val REQUEST_CODE_PERMISSIONS = 10
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

    // URL for the settings recommendation API (Port 5001)
    private val SETTINGS_API_URL = "http://192.168.188.224:5001/recommend_settings"

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var switchMode: SwitchMaterial
    private lateinit var buttonRecommend: MaterialButton // Button to trigger recommendation
    // private lateinit var buttonApplyAI: Button // Removed - AI settings applied automatically
    private lateinit var buttonCapture: Button
    private lateinit var imageManual: ImageView
    private lateinit var imageAI: ImageView
    private lateinit var comparisonLayout: LinearLayout
    private lateinit var manualControlsLayout: LinearLayout // Layout containing manual controls
    private lateinit var seekBarISO: SeekBar
    private lateinit var textViewISOValue: TextView
    private lateinit var seekBarShutter: SeekBar
    private lateinit var textViewShutterValue: TextView
    private lateinit var buttonCompare: Button

    // ScaleGestureDetector for zoom
    private lateinit var scaleGestureDetector: ScaleGestureDetector

    // Camera Components
    private var imageCapture: ImageCapture? = null
    // private var imageAnalyzer: ImageAnalysis? = null // Removed if not used for other analysis
    private var camera: androidx.camera.core.Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null // Added provider reference
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService // Executor for API calls
    private val httpClient = OkHttpClient()

    // State variables
    private var isAiModeEnabled = false // Track AI mode state
    private var currentISO: Int = 400 // Default ISO
    private var currentShutterSpeedNs: Long = 8000000 // Default Shutter Speed (1/125s in ns)

    // Handler for debouncing settings application
    private val settingsApplyHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var settingsApplyRunnable: Runnable? = null

    // Store URIs when images are saved
    private var manualCaptureUri: Uri? = null
    private var aiCaptureUri: Uri? = null

    // TODO: Store actual scores when calculated
    private var manualScoresMap: HashMap<String, Double>? = null
    private var aiScoresMap: HashMap<String, Double>? = null


    override val contentLayoutId: Int
        get() = R.layout.content_main

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize UI components
        previewView = findViewById(R.id.previewView)
        switchMode = findViewById(R.id.switchMode)
        buttonRecommend = findViewById(R.id.buttonRecommend)
        // buttonApplyAI = findViewById(R.id.buttonApplyAI) // Removed
        buttonCapture = findViewById(R.id.buttonCapture)
        imageManual = findViewById(R.id.imageManual)
        imageAI = findViewById(R.id.imageAI)
        comparisonLayout = findViewById(R.id.comparisonLayout)
        manualControlsLayout = findViewById(R.id.manualControlsLayout)
        seekBarISO = findViewById(R.id.seekBarISO)
        textViewISOValue = findViewById(R.id.textViewISOValue)
        seekBarShutter = findViewById(R.id.seekBarShutter)
        textViewShutterValue = findViewById(R.id.textViewShutterValue)
        buttonCompare = findViewById(R.id.buttonCompare)

        // Initialize executors
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()

        // Setup pinch-to-zoom and listeners
        setupPinchToZoom()
        setupUIListeners() // Call this AFTER initializing all buttons/views

        // Check permissions and start camera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Set initial UI state based on switch
        isAiModeEnabled = switchMode.isChecked
        updateManualControlsVisibility() // Show/hide controls based on initial state
        updateUIFromState() // Set initial text/slider values

        // Touch listener for zoom
        previewView.setOnTouchListener { _, event ->
            val handled = scaleGestureDetector.onTouchEvent(event)
            handled || previewView.onTouchEvent(event) // Pass unhandled events to previewView
        }

        // Set the checked item for this activity in the drawer
        setCheckedNavigationItem(R.id.nav_capture)
    }

    // Consolidated UI Listeners
    private fun setupUIListeners() {

        // Recommend button listener (Only active in AI mode)
        buttonRecommend.setOnClickListener {
            if (!isAiModeEnabled) {
                Toast.makeText(this, "Enable AI Mode to get recommendations", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (previewView.bitmap == null) {
                Toast.makeText(this, "Preview not available", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            Log.d(TAG, "Recommend button clicked in AI mode.")
            val bitmap = previewView.bitmap // Get bitmap directly from PreviewView

            if (bitmap != null) {
                // Convert Bitmap to JPEG ByteArray in background thread
                analysisExecutor.execute {
                    try {
                        val outputStream = ByteArrayOutputStream()
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                        val imageBytes = outputStream.toByteArray()
                        Log.d(TAG, "Preview frame captured for recommendation, size: ${imageBytes.size} bytes")
                        requestRecommendedSettings(imageBytes) // Call the correct API function
                    } catch (e: Exception) {
                        Log.e(TAG, "Error converting preview bitmap to bytes: ${e.message}", e)
                        runOnUiThread { Toast.makeText(this, "Error capturing preview", Toast.LENGTH_SHORT).show() }
                    }
                }
            } else {
                Log.e(TAG, "Could not get bitmap from previewView for recommendation.")
                Toast.makeText(this, "Could not capture preview", Toast.LENGTH_SHORT).show()
            }
        }

        // Mode Switch Listener
        switchMode.setOnCheckedChangeListener { _, isChecked ->
            isAiModeEnabled = isChecked
            val mode = if (isChecked) "AI Mode ON" else "Manual Mode ON"
            Toast.makeText(this, mode, Toast.LENGTH_SHORT).show()
            updateManualControlsVisibility() // Show/hide controls

            if (isChecked) { // Switched TO AI
                comparisonLayout.visibility = View.GONE
                resetCamera2SettingsToAuto() // Reset camera hardware to auto modes
                Log.d(TAG, "Switched to AI Mode")
                // Optionally trigger an initial recommendation?
                // buttonRecommend.performClick()
            } else { // Switched TO Manual
                comparisonLayout.visibility = View.GONE
                // Apply the current values shown on the sliders
                applyManualSettingsFromUI()
                Log.d(TAG, "Switched to Manual Mode")
            }
        }

        // Capture button listener
        buttonCapture.setOnClickListener {
            takePhoto()
        }

        // Compare button listener
        buttonCompare.setOnClickListener {
            openComparisonActivity()
        }

        // --- SeekBar Listeners (Apply settings only in Manual Mode) ---
        seekBarISO.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    // Update text view immediately
                    textViewISOValue.text = getISOValueFromProgress(progress).toString()
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                if (!isAiModeEnabled) { // Only apply if in manual mode
                    // Debounce the application of settings
                    settingsApplyRunnable?.let { settingsApplyHandler.removeCallbacks(it) }
                    settingsApplyRunnable = Runnable {
                        Log.d(TAG, "Applying debounced manual settings from ISO SeekBar")
                        applyManualSettingsFromUI()
                    }
                    settingsApplyHandler.postDelayed(settingsApplyRunnable!!, 300) // 300ms delay
                }
            }
        })

        seekBarShutter.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    // Update text view immediately
                    textViewShutterValue.text = formatShutterSpeed(getShutterSpeedNsFromProgress(progress))
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                if (!isAiModeEnabled) { // Only apply if in manual mode
                    // Debounce the application of settings
                    settingsApplyRunnable?.let { settingsApplyHandler.removeCallbacks(it) }
                    settingsApplyRunnable = Runnable {
                         Log.d(TAG, "Applying debounced manual settings from Shutter SeekBar")
                        applyManualSettingsFromUI()
                    }
                    settingsApplyHandler.postDelayed(settingsApplyRunnable!!, 300) // 300ms delay
                }
            }
        })
    }

    // Helper to show/hide manual controls based on isAiModeEnabled
    private fun updateManualControlsVisibility() {
        if (isAiModeEnabled) {
            manualControlsLayout.visibility = View.VISIBLE // Keep controls visible in AI mode to show recommended values
            buttonRecommend.visibility = View.VISIBLE
            // Disable SeekBars in AI mode
            seekBarISO.isEnabled = true
            seekBarShutter.isEnabled = true
        } else {
            manualControlsLayout.visibility = View.VISIBLE
            buttonRecommend.visibility = View.GONE
            // Enable SeekBars in Manual mode
            seekBarISO.isEnabled = true
            seekBarShutter.isEnabled = true
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                // Consider setting target rotation based on device orientation if needed
                // .setTargetRotation(previewView.display.rotation)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider?.unbindAll()
                camera = cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture // Add imageAnalyzer here if needed later
                )
                Log.d(TAG, "Camera bound successfully.")

                // Apply initial settings based on mode AFTER camera is bound
                if (isAiModeEnabled) {
                    resetCamera2SettingsToAuto()
                } else {
                    applyManualSettingsFromUI() // Apply initial slider values
                }

            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
                Toast.makeText(this, "Failed to start camera.", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // --- Removed old API/Analysis functions: analyzeCurrentFrameViaAPI, sendImageToAPI, applyAISettings ---

    // Apply Manual Settings read from UI SeekBars
    @SuppressLint("UnsafeOptInUsageError")
    private fun applyManualSettingsFromUI() {
        if (camera == null) {
            Log.w(TAG, "Cannot apply manual settings, camera is null.")
            return
        }
        if (isAiModeEnabled) {
            Log.w(TAG, "Attempted to apply manual settings while in AI mode. Ignoring.")
            return // Don't apply manual settings if in AI mode
        }

        val iso = getISOValueFromProgress(seekBarISO.progress)
        val shutterNs = getShutterSpeedNsFromProgress(seekBarShutter.progress)

        Log.d(TAG, "Applying Manual Settings from UI - ISO: $iso, Shutter Speed: ${shutterNs}ns")

        // Apply using Camera2 Interop
        applyCamera2Settings(iso, shutterNs) { success ->
            if (success) {
                // Update internal state only if application was successful
                currentISO = iso
                currentShutterSpeedNs = shutterNs
                Log.i(TAG, "Applied manual settings from UI.")
                // UI TextViews should already be updated by onProgressChanged
            } else {
                Log.e(TAG, "Failed to apply manual settings from UI.")
                // Optionally revert UI or show error
            }
        }
    }

    // Apply Recommended AI Settings
    @SuppressLint("UnsafeOptInUsageError")
    private fun applyRecommendedSettings(iso: Int, shutterSpeedSec: Float) {
        if (camera == null) {
            Log.e(TAG, "Cannot apply recommended settings, camera is null.")
            return
        }
        // Only apply if still in AI mode when the response arrives
        if (!isAiModeEnabled) {
             Log.w(TAG, "Received AI recommendation but no longer in AI mode. Ignoring.")
             return
        }

        val shutterSpeedNs = (shutterSpeedSec * 1_000_000_000L).toLong()
        Log.d(TAG, "Applying Recommended Settings - ISO: $iso, Shutter Speed: ${shutterSpeedNs}ns")

        // Apply using Camera2 Interop
        applyCamera2Settings(iso, shutterSpeedNs) { success ->
            if (success) {
                // Update internal state and UI if application was successful
                currentISO = iso
                currentShutterSpeedNs = shutterSpeedNs
                updateUIFromState() // Update SeekBars and TextViews to reflect applied AI settings
                Log.i(TAG, "Successfully applied recommended settings.")
            } else {
                Log.e(TAG, "Failed to apply recommended settings.")
                Toast.makeText(this, "Failed to apply AI settings", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Central function to apply Camera2 settings (ISO, Shutter) and handle result
    @SuppressLint("UnsafeOptInUsageError")
    private fun applyCamera2Settings(iso: Int, shutterNanos: Long, callback: (Boolean) -> Unit) {
        camera?.let { cam ->
            val camera2Control = Camera2CameraControl.from(cam.cameraControl)
            val captureOptionsBuilder = CaptureRequestOptions.Builder()
                // Disable auto exposure when setting manual ISO/Shutter
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, iso.coerceIn(100, 8000)) // Coerce ISO to a reasonable range
                .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterNanos.coerceAtLeast(10000)) // Coerce shutter time (min 10us)

            // Apply the settings
            val future = camera2Control.addCaptureRequestOptions(captureOptionsBuilder.build())
            future.addListener({
                try {
                    future.get() // Check for errors
                    Log.d(TAG, "Applied Camera2 Settings: ISO=$iso, Shutter=${shutterNanos}ns")
                    callback(true) // Indicate success
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to apply Camera2 settings: ${e.message}", e)
                    callback(false) // Indicate failure
                }
            }, ContextCompat.getMainExecutor(this))

        } ?: run {
            Log.e(TAG, "Cannot apply Camera2 settings, camera is null")
            callback(false) // Indicate failure
        }
    }

    // Function to reset Camera2 settings to Auto
    @SuppressLint("UnsafeOptInUsageError")
    private fun resetCamera2SettingsToAuto() {
        camera?.let { cam ->
            val camera2Control = Camera2CameraControl.from(cam.cameraControl)
            val optionsBuilder = CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON) // Enable Auto Exposure
                // Clear specific manual settings if previously set
                .clearCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY)
                .clearCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME)
                // Optionally re-enable AWB and AF if they were disabled
                // .setCaptureRequestOption(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_ON)
                // .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)

            val captureOptions = optionsBuilder.build()
            // Assign the returned future to a variable
            val resetFuture = camera2Control.addCaptureRequestOptions(captureOptions)
            resetFuture.addListener({ // Use the variable in the listener
                try {
                    resetFuture.get() // Check for errors using the correct variable
                    Log.d(TAG, "Camera2 Settings reset to Auto")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to reset Camera2 settings to Auto: ${e.message}", e)
                }
            }, ContextCompat.getMainExecutor(this))
        }
    }

    // Function to capture and save the final photo
    private fun takePhoto() {
        Log.d(TAG, "takePhoto called")
        val imageCapture = imageCapture ?: run {
            Log.e(TAG, "ImageCapture use case is null.")
            Toast.makeText(this, "Error: Camera not ready.", Toast.LENGTH_SHORT).show()
            return
        }

        val suffix = if (isAiModeEnabled) "AI" else "Manual" // Use state variable
        val name = SimpleDateFormat("ddMMyy_HHmmss", Locale.US)
            .format(System.currentTimeMillis()) + "_$suffix.jpg"

        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/CameraAI_Captures")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            .build()

        Log.d(TAG, "Attempting to save image: $name")
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    Toast.makeText(baseContext, "Photo capture failed: ${exc.message}", Toast.LENGTH_SHORT).show()
                    runOnUiThread { comparisonLayout.visibility = View.GONE }
                }

                // --- Updated onImageSaved ---
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = output.savedUri
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)

                    savedUri?.let { uri ->
                        Log.d(TAG, "Displaying saved image: $uri")
                        runOnUiThread { // Ensure UI updates are on main thread
                            if (isAiModeEnabled) {
                                Log.d(TAG, "Setting AI image view and storing URI")
                                imageAI.setImageURI(uri)
                                aiCaptureUri = uri // Store AI URI
                                imageAI.tag = uri // Also store in tag (optional redundancy)
                            } else {
                                Log.d(TAG, "Setting Manual image view and storing URI")
                                imageManual.setImageURI(uri)
                                manualCaptureUri = uri // Store Manual URI
                                imageManual.tag = uri // Also store in tag (optional redundancy)
                            }
                            // Show comparison layout only if BOTH images have been captured
                            if (manualCaptureUri != null && aiCaptureUri != null) {
                                comparisonLayout.visibility = View.VISIBLE
                                Log.d(TAG, "Comparison layout visibility set to VISIBLE")
                            } else {
                                comparisonLayout.visibility = View.GONE // Keep hidden if only one is captured
                            }
                        }
                    } ?: run {
                        Log.e(TAG, "Saved URI is null, cannot display image.")
                        Toast.makeText(baseContext, "Failed to get image URI", Toast.LENGTH_SHORT).show()
                        runOnUiThread { comparisonLayout.visibility = View.GONE }
                    }
                }
                // --- End Updated onImageSaved ---
            }
        )
    }

    // --- Image Conversion (Simplified - Assumes JPEG or uses Bitmap fallback) ---
    @SuppressLint("UnsafeOptInUsageError")
    private fun imageProxyToJpegByteArray(image: ImageProxy): ByteArray? {
        return try {
            when (image.format) {
                ImageFormat.JPEG -> {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    bytes
                }
                ImageFormat.YUV_420_888 -> {
                    // Convert YUV to Bitmap first, then compress Bitmap to JPEG
                    val bitmap = imageProxyToBitmap(image) // Use helper
                    if (bitmap != null) {
                        val out = ByteArrayOutputStream()
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                        out.toByteArray()
                    } else {
                        Log.e(TAG, "YUV to Bitmap conversion failed.")
                        null
                    }
                }
                else -> {
                    Log.e(TAG, "Unsupported image format for JPEG conversion: ${image.format}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to JPEG bytes: ${e.message}", e)
            null
        } finally {
             image.close() // Ensure ImageProxy is closed after use
        }
    }

    // Helper to convert ImageProxy (any format) to Bitmap, handling rotation
    @SuppressLint("UnsafeOptInUsageError")
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            // Use a library function if available, or implement YUV->RGB conversion
            // For simplicity, let's assume a basic conversion or use a placeholder
            // A more robust solution might involve RenderScript or a dedicated library

            // Example using a simple (potentially slow) conversion for YUV
            if (imageProxy.format == ImageFormat.YUV_420_888) {
                 val yBuffer = imageProxy.planes[0].buffer // Y
                 val uBuffer = imageProxy.planes[1].buffer // U
                 val vBuffer = imageProxy.planes[2].buffer // V

                 val ySize = yBuffer.remaining()
                 val uSize = uBuffer.remaining()
                 val vSize = vBuffer.remaining()

                 val nv21 = ByteArray(ySize + uSize + vSize)

                 // Check pixel strides and row strides if necessary for specific devices
                 // This basic implementation assumes planar layout without significant gaps
                 yBuffer.get(nv21, 0, ySize)
                 vBuffer.get(nv21, ySize, vSize) // Order depends on NV12 vs NV21 - CameraX usually provides NV21-like access
                 uBuffer.get(nv21, ySize + vSize, uSize)


                 val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
                 val out = ByteArrayOutputStream()
                 yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, out)
                 val imageBytes = out.toByteArray()
                 val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                 // Apply rotation if necessary
                 if (bitmap != null && imageProxy.imageInfo.rotationDegrees != 0) {
                     val matrix = Matrix()
                     matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                     Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                 } else {
                     bitmap
                 }

            } else {
                 // Fallback for other formats (like JPEG already handled in imageProxyToJpegByteArray)
                 Log.w(TAG, "Bitmap conversion might be inefficient for format: ${imageProxy.format}")
                 val buffer = imageProxy.planes[0].buffer
                 val bytes = ByteArray(buffer.remaining())
                 buffer.get(bytes)
                 BitmapFactory.decodeByteArray(bytes, 0, bytes.size) // Decode directly if possible
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap: ${e.message}", e)
            null
        }
        // DO NOT close the imageProxy here if imageProxyToJpegByteArray calls this
    }
    // --- End Image Conversion ---


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause called")
    }

    override fun onStop() {
        super.onStop()
        Log.d(TAG, "onStop called")
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called")
        // Remove any pending settings application callbacks
        settingsApplyRunnable?.let { settingsApplyHandler.removeCallbacks(it) }
        cameraExecutor.shutdown()
        analysisExecutor.shutdown()
        // Clean up OkHttp resources
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
        httpClient.cache?.close() // Close cache if used
    }

    // Setup Pinch to Zoom
    private fun setupPinchToZoom() {
        val listener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                camera?.let { cam ->
                    val zoomState = cam.cameraInfo.zoomState.value ?: return false
                    val currentZoomRatio = zoomState.zoomRatio
                    val delta = detector.scaleFactor
                    val newZoomRatio = (currentZoomRatio * delta).coerceIn(
                        zoomState.minZoomRatio,
                        zoomState.maxZoomRatio
                    )
                    cam.cameraControl.setZoomRatio(newZoomRatio)
                    // Log.d(TAG, "Zoom Ratio: $newZoomRatio") // Optional logging
                    return true
                }
                return false
            }
        }
        scaleGestureDetector = ScaleGestureDetector(this, listener)
    }

    // Navigate to Comparison Activity
    private fun openComparisonActivity() {
        Log.d(TAG, "Attempting to open ComparisonActivity")

        if (manualCaptureUri == null || aiCaptureUri == null) {
            Log.e(TAG, "Cannot open comparison: One or both image URIs are missing.")
            Toast.makeText(this, "Capture both Manual and AI images first", Toast.LENGTH_SHORT).show()
            return
        }

        // --- Calculate and assign scores ---
        manualScoresMap = calculateBlurScores(manualCaptureUri!!) // Calculate scores for manual image
        aiScoresMap = calculateBlurScores(aiCaptureUri!!)         // Calculate scores for AI image
        // --- End Score Calculation ---

        Log.d(TAG, "Starting ComparisonActivity with Manual URI: $manualCaptureUri, AI URI: $aiCaptureUri")
        Log.d(TAG, "Manual Scores: $manualScoresMap") // Log scores
        Log.d(TAG, "AI Scores: $aiScoresMap")         // Log scores

        val intent = Intent(this, ComparisonActivity::class.java).apply {
            putExtra("manual_image_uri", manualCaptureUri)
            putExtra("ai_image_uri", aiCaptureUri)

            // Pass the populated (or null if calculation failed) maps
            putExtra("manual_scores", manualScoresMap)
            putExtra("ai_scores", aiScoresMap)

            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(intent)
    }

    // Save Bitmap to Cache for Sharing (Removed - Using MediaStore directly now)
    /*
    private fun saveBitmapToCache(bitmap: Bitmap, prefix: String): Uri? {
        // ... implementation removed ...
    }
    */

    // Request Recommended Settings from API (Port 5001)
    private fun requestRecommendedSettings(imageBytes: ByteArray) {
        Log.d(TAG, "Requesting recommended settings from: $SETTINGS_API_URL")
        runOnUiThread { Toast.makeText(this, "Getting AI recommendation...", Toast.LENGTH_SHORT).show() }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "camera_preview.jpg", imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .build()
        val request = Request.Builder().url(SETTINGS_API_URL).post(requestBody).build()

        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "Settings API call failed: ${e.message}", e)
                runOnUiThread { Toast.makeText(this@MainActivity, "Recommendation failed: ${e.message}", Toast.LENGTH_SHORT).show() }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body?.string()
                response.close() // Close body immediately

                if (!response.isSuccessful || responseBody == null) {
                    Log.e(TAG, "Settings API error. Code: ${response.code}, Body: $responseBody")
                    runOnUiThread { Toast.makeText(this@MainActivity, "Recommendation failed (Server Error ${response.code})", Toast.LENGTH_SHORT).show() }
                    return
                }

                Log.d(TAG, "Settings API response: $responseBody")
                try {
                    val jsonObject = JSONObject(responseBody)
                    val recommendedIso = jsonObject.optInt("iso", -1)
                    val recommendedShutterSpeedSec = jsonObject.optDouble("shutter_speed", -1.0)

                    if (recommendedIso > 0 && recommendedShutterSpeedSec > 0) {
                        Log.i(TAG, "Recommended Settings - ISO: $recommendedIso, Shutter: ${recommendedShutterSpeedSec}s")
                        runOnUiThread {
                            applyRecommendedSettings(recommendedIso, recommendedShutterSpeedSec.toFloat())
                            Toast.makeText(this@MainActivity, "AI Settings Applied", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        Log.w(TAG, "Received invalid settings from API: ISO=$recommendedIso, Shutter=$recommendedShutterSpeedSec")
                        runOnUiThread { Toast.makeText(this@MainActivity, "Invalid recommendation received", Toast.LENGTH_SHORT).show() }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing settings JSON response: ${e.message}", e)
                    runOnUiThread { Toast.makeText(this@MainActivity, "Error processing recommendation", Toast.LENGTH_SHORT).show() }
                }
            }
        })
    }

    // Helper function to update UI elements based on current state variables
    private fun updateUIFromState() {
        // Map currentISO to SeekBar progress
        val isoProgress = mapIsoToProgress(currentISO)
        seekBarISO.progress = isoProgress
        textViewISOValue.text = currentISO.toString()

        // Map currentShutterSpeedNs to SeekBar progress
        val shutterProgress = mapShutterNsToProgress(currentShutterSpeedNs)
        seekBarShutter.progress = shutterProgress
        textViewShutterValue.text = formatShutterSpeed(currentShutterSpeedNs)

        Log.d(TAG, "UI updated to reflect ISO: $currentISO, Shutter: ${formatShutterSpeed(currentShutterSpeedNs)}")
    }

    // Format shutter speed for display
    private fun formatShutterSpeed(shutterNs: Long): String {
        val shutterSec = shutterNs / 1_000_000_000.0
        return when {
            shutterSec >= 0.3 -> "%.1f s".format(shutterSec) // Show decimal for slower speeds
            shutterSec > 0 -> "1/%d s".format((1.0 / shutterSec).roundToInt().coerceAtLeast(1))
            else -> "N/A" // Handle zero or negative case
        }
    }


    // --- UPDATED MAPPING FUNCTIONS ---
    // Assumes seekBarISO has min="100", max="2500"
    // Assumes seekBarShutter has min="1", max="1000"

    private fun getISOValueFromProgress(progress: Int): Int {
        // Direct mapping assuming XML min/max are correct
        val actualMin = 100
        val actualMax = 2500
        return progress.coerceIn(actualMin, actualMax)
    }

    private fun getShutterSpeedNsFromProgress(progress: Int): Long {
        // Progress corresponds directly to the denominator (1 to 1000)
        val actualMinDenom = 1
        val actualMaxDenom = 1000

        val denominator = progress.coerceIn(actualMinDenom, actualMaxDenom)

        // Calculate nanoseconds: 1 / denominator * 1e9
        return (1_000_000_000.0 / denominator).toLong().coerceAtLeast(10000) // Ensure minimum exposure time
    }

    // Inverse mapping: ISO value to progress (needed for updateUIFromState)
    private fun mapIsoToProgress(iso: Int): Int {
        // Direct mapping assuming XML min="100", max="2500"
        val actualMin = 100
        val actualMax = 2500
        return iso.coerceIn(actualMin, actualMax)
    }

    // Inverse mapping: Shutter speed (ns) to progress (needed for updateUIFromState)
    private fun mapShutterNsToProgress(shutterNs: Long): Int {
        // Calculate denominator from nanoseconds
        val actualMinDenom = 1
        val actualMaxDenom = 1000

        if (shutterNs <= 0) return actualMinDenom // Handle invalid input, map to fastest speed progress

        val denominatorDouble = 1_000_000_000.0 / shutterNs.coerceAtLeast(1).toDouble() // Ensure shutterNs > 0

        val denominator = denominatorDouble.roundToInt().coerceIn(actualMinDenom, actualMaxDenom)

        return denominator
    }
    // --- END OF UPDATED MAPPING FUNCTIONS ---

    // Function to calculate scores (replace with your actual implementation)
    private fun calculateBlurScores(imageUri: Uri): HashMap<String, Double>? {
        return try {
            // 1. Load Bitmap from URI
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)

            // 2. Calculate scores using your algorithms
            val lapScore = calculateLaplacian(bitmap)
            val tenScore = calculateTenengrad(bitmap)
            val pbmScore = calculatePBM(bitmap)
            val compScore = calculateComposite(lapScore, tenScore, pbmScore) // Example

            // 3. Create and return map
            hashMapOf(
                "laplacian" to lapScore,
                "tenengrad" to tenScore,
                "pbm" to pbmScore,
                "composite" to compScore
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating scores for $imageUri: ${e.message}", e)
            null // Return null if calculation fails
        }
    }

    // Dummy calculation functions (replace with real ones)
    private fun calculateLaplacian(bitmap: Bitmap): Double = Math.random() * 1000
    private fun calculateTenengrad(bitmap: Bitmap): Double = Math.random() * 1000
    private fun calculatePBM(bitmap: Bitmap): Double = Math.random() * 0.1
    private fun calculateComposite(lap: Double, ten: Double, pbm: Double): Double = Math.random() * 10

} // End of MainActivity class