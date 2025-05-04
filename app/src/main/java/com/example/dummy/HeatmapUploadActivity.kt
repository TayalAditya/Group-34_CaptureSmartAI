package com.example.dummy

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix // <-- Add this import
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.exifinterface.media.ExifInterface // <-- Add this import
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.FileNotFoundException
import java.io.IOException
import java.util.concurrent.TimeUnit

// Inherit from BaseDrawerActivity
class HeatmapUploadActivity : BaseDrawerActivity() {

    private val TAG = "HeatmapUploadActivity"
    // Endpoint for heatmap generation
    private val HEATMAP_API_URL = "http://192.168.188.224:5001/generate_heatmap" // Ensure IP is correct

    // Views from content_heatmap_upload.xml
    private lateinit var buttonUploadHeatmap: Button
    private lateinit var imageViewHeatmapResult: ImageView
    private lateinit var progressBarHeatmapUpload: ProgressBar
    private lateinit var textViewHeatmapUploadStatus: TextView

    // OkHttp Client
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS) // Adjust timeouts as needed
        .readTimeout(60, TimeUnit.SECONDS)    // Heatmap generation might take longer
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    // Activity Result Launcher for picking an image
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult() // Use StartActivityForResult for ACTION_PICK
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                Log.d(TAG, "Image selected: $uri")
                // Optionally show a preview of the selected image before sending
                // imageViewHeatmapResult.setImageURI(uri)
                imageViewHeatmapResult.visibility = View.VISIBLE
                textViewHeatmapUploadStatus.visibility = View.GONE // Hide previous status
                generateHeatmap(uri) // Start heatmap generation
            } ?: run {
                Log.e(TAG, "Failed to get image URI from result.")
                showError("Failed to select image.")
            }
        } else {
            Log.d(TAG, "Image selection cancelled or failed.")
            // Optionally show a toast or message
        }
    }


    // Implement the abstract property from BaseDrawerActivity
    override val contentLayoutId: Int
        get() = R.layout.content_heatmap_upload // Use the heatmap upload layout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Base class handles setContentView

        // Initialize views using IDs from content_heatmap_upload.xml
        imageViewHeatmapResult = findViewById(R.id.imageViewHeatmapResult)
        buttonUploadHeatmap = findViewById(R.id.buttonUploadHeatmap)
        progressBarHeatmapUpload = findViewById(R.id.progressBarHeatmapUpload)
        textViewHeatmapUploadStatus = findViewById(R.id.textViewHeatmapUploadStatus)

        // Set listener for the upload button
        buttonUploadHeatmap.setOnClickListener {
            Log.d(TAG, "Upload Heatmap button clicked")
            openImagePicker() // Launch image picker
        }

        // Set the checked item for this activity in the drawer
        setCheckedNavigationItem(R.id.nav_heatmap)
    }

    private fun openImagePicker() {
        // Use ACTION_PICK for selecting existing images
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        // intent.type = "image/*" // Can also use this type specifier
        try {
            pickImageLauncher.launch(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch image picker: ${e.message}", e)
            Toast.makeText(this, "Cannot open image picker.", Toast.LENGTH_SHORT).show()
        }
    }

    // Function to convert URI to ByteArray and request heatmap
    private fun generateHeatmap(imageUri: Uri) {
        Log.d(TAG, "Starting heatmap generation for URI: $imageUri")
        progressBarHeatmapUpload.visibility = View.VISIBLE
        textViewHeatmapUploadStatus.text = "Processing image..."
        textViewHeatmapUploadStatus.visibility = View.VISIBLE
        imageViewHeatmapResult.setImageResource(R.drawable.ic_placeholder) // Reset view

        try {
            // Load the original bitmap
            val originalBitmap = loadBitmapFromUri(imageUri)

            if (originalBitmap == null) {
                showError("Failed to load image.")
                return
            }

            // --- Get Orientation and Rotate ---
            val orientation = getOrientationFromUri(imageUri)
            val rotatedBitmap = rotateBitmap(originalBitmap, orientation)
            Log.d(TAG, "Original Orientation: $orientation. Bitmap rotated for upload.")
            // --- End Rotation Logic ---


            // Convert the *rotated* Bitmap to JPEG ByteArray
            val outputStream = ByteArrayOutputStream()
            rotatedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream) // Adjust quality if needed
            val imageBytes = outputStream.toByteArray()
            Log.d(TAG, "Rotated image converted to ByteArray, size: ${imageBytes.size} bytes")

            if (imageBytes.isEmpty()) {
                 showError("Failed to convert rotated image to bytes.")
                 // Clean up rotated bitmap if it's different from original and rotation failed to compress
                 if (rotatedBitmap != originalBitmap) {
                    // rotatedBitmap.recycle() // Consider recycling if needed, be cautious
                 }
                 return
            }

            // Send the *rotated* image bytes to the server
            requestHeatmapFromServer(imageBytes)

            // Clean up bitmaps if they are no longer needed and different
            // Be careful with recycling if bitmaps might be used elsewhere or if errors occur
            // if (rotatedBitmap != originalBitmap) {
            //    originalBitmap.recycle() // Recycle original if rotated one was created and sent
            // }
            // rotatedBitmap.recycle() // Recycle the one that was compressed

        } catch (e: IOException) {
            Log.e(TAG, "Error processing image URI: ${e.message}", e)
            showError("Error loading image: ${e.message}")
        } catch (e: Exception) {
             Log.e(TAG, "Unexpected error during image processing: ${e.message}", e)
             showError("Error processing image")
        }
    }

    // --- Function to load bitmap (remains mostly the same) ---
     private fun loadBitmapFromUri(uri: Uri): Bitmap? {
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
             null
         } catch (e: OutOfMemoryError) {
             Log.e(TAG, "OutOfMemoryError loading bitmap from URI: $uri", e)
             null
         }
     }

    // --- NEW: Function to get EXIF orientation ---
    private fun getOrientationFromUri(uri: Uri?): Int {
        if (uri == null) return ExifInterface.ORIENTATION_UNDEFINED
        try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val exifInterface = ExifInterface(inputStream)
                return exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            }
        } catch (e: IOException) {
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
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1.0f, 1.0f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1.0f, -1.0f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1.0f, 1.0f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1.0f, 1.0f)
            }
            ExifInterface.ORIENTATION_NORMAL, ExifInterface.ORIENTATION_UNDEFINED -> return source
            else -> return source
        }

        return try {
            val rotatedBitmap = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            // Avoid recycling source here immediately, let the caller manage if needed
            rotatedBitmap
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OutOfMemoryError rotating bitmap", e)
            source // Return original if rotation fails
        }
    }


    // Function to send image bytes and receive heatmap image
    private fun requestHeatmapFromServer(imageBytes: ByteArray) {
        Log.d(TAG, "Sending image to heatmap server: $HEATMAP_API_URL")
        textViewHeatmapUploadStatus.text = "Generating heatmap..." // Update status

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", // Matches the key expected by the Flask API
                "uploaded_image.jpg",
                imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url(HEATMAP_API_URL)
            .post(requestBody)
            .build()

        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "Heatmap API call failed: ${e.message}", e)
                showError("Heatmap generation failed: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                Log.d(TAG, "Heatmap API call successful: ${response.code}")

                if (!response.isSuccessful) {
                    val errorBody = response.body?.string() ?: "Unknown server error"
                    Log.e(TAG, "Heatmap API error. Code: ${response.code}, Body: $errorBody")
                    showError("Heatmap failed (Server Error ${response.code})")
                    response.close()
                    return
                }

                // Get image bytes directly from response body
                val imageBytesResponse = response.body?.bytes()
                response.close() // Close body immediately

                if (imageBytesResponse == null || imageBytesResponse.isEmpty()) {
                    Log.e(TAG, "Received empty response body from heatmap API.")
                    showError("Received empty heatmap response.")
                    return
                }

                // Decode bytes to Bitmap
                val heatmapBitmap = BitmapFactory.decodeByteArray(imageBytesResponse, 0, imageBytesResponse.size)

                if (heatmapBitmap == null) {
                    Log.e(TAG, "Failed to decode heatmap image bytes.")
                    showError("Failed to decode heatmap result.")
                    return
                }

                // Display the bitmap on the UI thread
                runOnUiThread {
                    Log.d(TAG, "Heatmap generated successfully. Displaying image.")
                    progressBarHeatmapUpload.visibility = View.GONE
                    textViewHeatmapUploadStatus.visibility = View.GONE // Hide status text
                    imageViewHeatmapResult.setImageBitmap(heatmapBitmap) // Display the heatmap!
                    Toast.makeText(this@HeatmapUploadActivity, "Heatmap Generated", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    // Helper function to show errors and update UI
    private fun showError(message: String) {
        runOnUiThread { // Ensure UI updates are on the main thread
            progressBarHeatmapUpload.visibility = View.GONE
            textViewHeatmapUploadStatus.text = message
            textViewHeatmapUploadStatus.visibility = View.VISIBLE
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            imageViewHeatmapResult.setImageResource(R.drawable.ic_placeholder) // Show placeholder on error
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cancel any ongoing OkHttp calls to prevent leaks
        httpClient.dispatcher.cancelAll()
    }
}