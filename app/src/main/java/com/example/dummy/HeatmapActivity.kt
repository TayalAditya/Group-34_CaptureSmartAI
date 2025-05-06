package com.example.dummy

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.FileNotFoundException
import java.io.IOException
import java.util.concurrent.TimeUnit

class HeatmapActivity : BaseDrawerActivity() { // Inherit from BaseDrawerActivity

    private val TAG = "HeatmapActivity"
    // Use the same base URL as MainActivity, just change the endpoint
    private val HEATMAP_API_URL = "http://IP/generate_heatmap" // Make sure IP is correct

    private lateinit var imageViewHeatmap: ImageView
    private lateinit var progressBarHeatmap: ProgressBar
    private lateinit var textViewStatus: TextView
    private var imageUri: Uri? = null

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS) // Increase timeouts for potentially larger image uploads/processing
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    override val contentLayoutId: Int
        get() = R.layout.content_heatmap // Use the content layout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // setContentView(R.layout.activity_heatmap) // BaseDrawerActivity handles this

        imageViewHeatmap = findViewById(R.id.imageViewHeatmap)
        progressBarHeatmap = findViewById(R.id.progressBarHeatmap)
        textViewStatus = findViewById(R.id.textViewHeatmapStatus)

        // Get the URI String passed from ComparisonActivity or potentially the drawer
        val uriString = intent.getStringExtra("manual_image_uri") // Key used by ComparisonActivity

        if (uriString != null) {
            try {
                imageUri = Uri.parse(uriString)
                Log.d(TAG, "Received Image URI: $imageUri")

                // Load the original bitmap (potentially unrotated)
                val originalBitmap = loadBitmapFromUri(imageUri)

                if (originalBitmap != null) {
                    // --- Get Orientation and Rotate ---
                    val orientation = getOrientationFromUri(imageUri)
                    val rotatedBitmap = rotateBitmap(originalBitmap, orientation)
                    Log.d(TAG, "Original Orientation: $orientation. Bitmap rotated.")
                    // --- End Rotation Logic ---

                    // Convert the *rotated* bitmap to JPEG byte array
                    val outputStream = ByteArrayOutputStream()
                    // Use the rotatedBitmap here
                    rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                    val imageBytes = outputStream.toByteArray()

                    if (imageBytes.isEmpty()) {
                        showError("Failed to convert rotated bitmap to bytes.")
                        return
                    }

                    Log.d(TAG, "Rotated bitmap loaded and converted to ${imageBytes.size} bytes. Requesting heatmap...")
                    requestHeatmapGeneration(imageBytes) // Send the bytes of the rotated bitmap

                } else {
                    Log.e(TAG, "Failed to load bitmap from URI.")
                    showError("Failed to load image.")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error processing URI or generating heatmap: ${e.message}", e)
                showError("Error displaying heatmap.")
            }
        } else {
            Log.e(TAG, "Image URI not provided in Intent.")
            showError("No image specified for heatmap.")
        }

        setCheckedNavigationItem(R.id.nav_heatmap)
    }

    private fun loadBitmapFromUri(uri: Uri?): Bitmap? {
        if (uri == null) return null
        return try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: FileNotFoundException) {
            Log.e(TAG, "File not found for URI: $uri", e)
            null
        } catch (e: IOException) {
            Log.e(TAG, "IOException loading bitmap from URI: $uri", e)
            null
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException: Permission denied for URI: $uri", e)
            null // May need explicit permission grant if URI is from another app/provider
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OutOfMemoryError loading bitmap from URI: $uri", e)
            // Consider downsampling here if this happens often
            null
        }
    }

    // --- NEW: Function to get EXIF orientation ---
    private fun getOrientationFromUri(uri: Uri?): Int {
        if (uri == null) return ExifInterface.ORIENTATION_UNDEFINED
        try {
            // Open a new InputStream specifically for ExifInterface
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val exifInterface = ExifInterface(inputStream)
                return exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            }
        } catch (e: IOException) {
            // Log error but default to normal orientation
            Log.e(TAG, "Could not get EXIF orientation from URI: $uri", e)
        } catch (e: SecurityException) {
             Log.e(TAG, "SecurityException getting EXIF orientation from URI: $uri", e)
        }
        return ExifInterface.ORIENTATION_NORMAL // Default if reading fails
    }

    // --- NEW: Function to rotate bitmap based on orientation ---
    private fun rotateBitmap(source: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            // Handle mirrored orientations if necessary (less common for camera photos)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1.0f, 1.0f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1.0f, -1.0f)
            // Combinations like TRANSPOSE/TRANSVERSE might need combined operations
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1.0f, 1.0f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1.0f, 1.0f)
            }
            ExifInterface.ORIENTATION_NORMAL, ExifInterface.ORIENTATION_UNDEFINED -> return source // No rotation needed
            else -> return source // Unknown orientation, return original
        }

        return try {
            // Create a new bitmap by applying the rotation matrix to the source bitmap
            val rotatedBitmap = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            // Recycle the original bitmap if it's different and no longer needed directly
            // Be careful if source is used elsewhere. Here it seems safe.
            if (rotatedBitmap != source) {
               // source.recycle() // Careful with recycling if source is needed elsewhere
            }
            rotatedBitmap
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OutOfMemoryError rotating bitmap", e)
            // Return the original bitmap if we run out of memory trying to rotate
            source
        }
    }

    private fun requestHeatmapGeneration(imageBytes: ByteArray) {
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "manual_image.jpg", imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull()))
            .build()

        val request = Request.Builder().url(HEATMAP_API_URL).post(requestBody).build()

        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "Heatmap API call failed: ${e.message}", e)
                runOnUiThread { showError("Heatmap generation failed: ${e.message}") }
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    val errorBody = response.body?.string() ?: "Unknown server error"
                    Log.e(TAG, "Heatmap API error. Code: ${response.code}, Body: $errorBody")
                    runOnUiThread { showError("Heatmap failed (Server Error ${response.code})") }
                    response.close() // Close body even on error
                    return
                }

                // Get image bytes directly from response body
                val imageBytesResponse = response.body?.bytes()
                response.close() // Close body immediately

                if (imageBytesResponse == null || imageBytesResponse.isEmpty()) {
                    Log.e(TAG, "Received empty response body from heatmap API.")
                    runOnUiThread { showError("Received empty heatmap response.") }
                    return
                }

                // Decode bytes to Bitmap
                val heatmapBitmap = BitmapFactory.decodeByteArray(imageBytesResponse, 0, imageBytesResponse.size)

                if (heatmapBitmap == null) {
                    Log.e(TAG, "Failed to decode heatmap image bytes.")
                    runOnUiThread { showError("Failed to decode heatmap result.") }
                    return
                }

                // Display the bitmap on the UI thread
                runOnUiThread {
                    Log.d(TAG, "Heatmap generated successfully. Displaying image.")
                    progressBarHeatmap.visibility = View.GONE
                    textViewStatus.visibility = View.GONE
                    imageViewHeatmap.setImageBitmap(heatmapBitmap)
                    Toast.makeText(this@HeatmapActivity, "Heatmap Generated", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    private fun showError(message: String) {
        progressBarHeatmap.visibility = View.GONE
        textViewStatus.text = message
        textViewStatus.visibility = View.VISIBLE
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        imageViewHeatmap.setImageResource(R.drawable.ic_placeholder) // Show placeholder on error
    }

     override fun onDestroy() {
        super.onDestroy()
        // Cancel any ongoing OkHttp calls for this activity
        // Iterate through queued and running calls and cancel if tagged (if you add tagging)
        // Or cancel all calls associated with the client if this client is only used here.
        // For simplicity, we rely on activity destruction, but explicit cancellation is better.
    }
}

