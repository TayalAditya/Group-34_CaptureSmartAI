package com.example.dummy

import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.cardview.widget.CardView
import androidx.core.graphics.drawable.toBitmap
import okhttp3.* // Import OkHttp classes
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

// Inherit from BaseDrawerActivity
class UnblurActivity : BaseDrawerActivity() {

    private val TAG = "UnblurActivity"
    // Use the same server URL or a different one if needed for single image analysis
    private val SERVER_URL = "http://192.168.188.224:5000/analyze" // Or your specific endpoint

    // Views from content_unblur.xml
    private lateinit var buttonSelectImage: Button
    private lateinit var imageViewSelected: ImageView
    private lateinit var progressBarUnblur: ProgressBar
    private lateinit var cardScores: CardView
    private lateinit var textViewLaplacian: TextView
    private lateinit var textViewTenengrad: TextView
    private lateinit var textViewPBM: TextView
    private lateinit var textViewComposite: TextView // Assuming this shows the main blur score

    // OkHttp Client
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS) // Increase timeout if needed
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // Activity Result Launcher for picking an image
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            Log.d(TAG, "Image selected: $it")
            imageViewSelected.setImageURI(it)
            imageViewSelected.visibility = View.VISIBLE
            cardScores.visibility = View.GONE // Hide previous results
            analyzeImage(it) // Start analysis
        } ?: run {
            Log.d(TAG, "No image selected")
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
        }
    }

    // Implement the abstract property from BaseDrawerActivity
    override val contentLayoutId: Int
        get() = R.layout.content_unblur // Use the correct layout file

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Base class handles setContentView

        // Initialize views using IDs from content_unblur.xml
        buttonSelectImage = findViewById(R.id.buttonSelectImage)
        imageViewSelected = findViewById(R.id.imageViewSelected)
        progressBarUnblur = findViewById(R.id.progressBarUnblur)
        cardScores = findViewById(R.id.cardScores)
        textViewLaplacian = findViewById(R.id.textViewLaplacian)
        textViewTenengrad = findViewById(R.id.textViewTenengrad)
        textViewPBM = findViewById(R.id.textViewPBM)
        textViewComposite = findViewById(R.id.textViewComposite)

        // Set listener for the select image button
        buttonSelectImage.setOnClickListener {
            Log.d(TAG, "Select Image button clicked")
            pickImageLauncher.launch("image/*") // Launch image picker
        }

        // Set the checked item for this activity in the drawer
        setCheckedNavigationItem(R.id.nav_blur_severity) // Assuming this is the correct item
    }

    // Function to convert URI to ByteArray and send to server
    private fun analyzeImage(imageUri: Uri) {
        Log.d(TAG, "Starting image analysis for URI: $imageUri")
        progressBarUnblur.visibility = View.VISIBLE // Show progress bar
        cardScores.visibility = View.GONE // Hide results card

        try {
            // Get Bitmap from URI
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)

            // Convert Bitmap to JPEG ByteArray
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, outputStream) // Compress slightly
            val imageBytes = outputStream.toByteArray()
            Log.d(TAG, "Image converted to ByteArray, size: ${imageBytes.size} bytes")

            // Send image bytes to the server
            sendImageToServer(imageBytes)

        } catch (e: IOException) {
            Log.e(TAG, "Error processing image URI: ${e.message}", e)
            runOnUiThread {
                progressBarUnblur.visibility = View.GONE
                Toast.makeText(this, "Error loading image: ${e.message}", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
             Log.e(TAG, "Unexpected error during image processing: ${e.message}", e)
             runOnUiThread {
                 progressBarUnblur.visibility = View.GONE
                 Toast.makeText(this, "Error processing image", Toast.LENGTH_SHORT).show()
             }
        }
    }

    // Function to send image bytes using OkHttp
    private fun sendImageToServer(imageBytes: ByteArray) {
        Log.d(TAG, "Sending image to server: $SERVER_URL")
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image",
                "uploaded_image.jpg",
                imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, imageBytes.size)
            )
            .build()

        val request = Request.Builder()
            .url(SERVER_URL) // Use the defined server URL
            .post(requestBody)
            .build()

        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "API call failed: ${e.message}", e)
                runOnUiThread {
                    progressBarUnblur.visibility = View.GONE
                    Toast.makeText(this@UnblurActivity, "Analysis failed: ${e.message}", Toast.LENGTH_LONG).show()
                    cardScores.visibility = View.GONE
                }
            }

            override fun onResponse(call: Call, response: Response) {
                Log.d(TAG, "API call successful: ${response.code}")
                val responseBody = response.body?.string()

                runOnUiThread { progressBarUnblur.visibility = View.GONE } // Hide progress bar

                if (response.isSuccessful && responseBody != null) {
                    Log.d(TAG, "Response body: $responseBody")
                    try {
                        val jsonObject = JSONObject(responseBody)

                        // --- Extract scores using CORRECT keys from Python API ---
                        val laplacian = jsonObject.optDouble("laplacian_variance", -1.0)
                        val tenengrad = jsonObject.optDouble("tenengrad_score", -1.0) // Use 'tenengrad_score'
                        val pbm = jsonObject.optDouble("perceptual_blur_metric", -1.0) // Use 'perceptual_blur_metric'
                        val blurScore = jsonObject.optDouble("predicted_blur_score", -1.0) // Use 'predicted_blur_score'
                        // ---

                        Log.d(TAG, "Parsed Scores: Blur=$blurScore, Lap=$laplacian, Ten=$tenengrad, PBM=$pbm")

                        runOnUiThread {
                            // Display scores
                            textViewLaplacian.text = if (laplacian >= 0) "Laplacian: %.2f".format(laplacian) else "Laplacian: N/A"
                            textViewTenengrad.text = if (tenengrad >= 0) "Tenengrad: %.2f".format(tenengrad) else "Tenengrad: N/A"
                            textViewPBM.text = if (pbm >= 0) "PBM: %.4f".format(pbm) else "PBM: N/A"
                            // Display the predicted_blur_score in the composite text view
                            textViewComposite.text = if (blurScore >= 0) "Blur Severity: %.1f".format(blurScore) else "Blur Severity: N/A"

                            cardScores.visibility = View.VISIBLE // Show results card
                        }

                    } catch (e: Exception) {
                        Log.e(TAG, "Error parsing JSON response: ${e.message}", e)
                        runOnUiThread {
                            Toast.makeText(this@UnblurActivity, "Error processing results", Toast.LENGTH_SHORT).show()
                            cardScores.visibility = View.GONE
                        }
                    }
                } else {
                    Log.e(TAG, "API call failed or empty response body. Code: ${response.code}, Body: $responseBody")
                    runOnUiThread {
                        Toast.makeText(this@UnblurActivity, "Analysis failed (Server Error ${response.code})", Toast.LENGTH_LONG).show()
                        cardScores.visibility = View.GONE
                    }
                }
                // Ensure response body is closed
                response.body?.close()
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cancel any ongoing OkHttp calls to prevent leaks
        httpClient.dispatcher.cancelAll()
    }
}