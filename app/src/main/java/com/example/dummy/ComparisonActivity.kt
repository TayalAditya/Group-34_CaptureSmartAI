package com.example.dummy

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ImageView // <-- Make sure this is imported
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
// Remove this import if you are not using ShapeableImageView anywhere else
// import com.google.android.material.imageview.ShapeableImageView

class ComparisonActivity : AppCompatActivity() {

    private val TAG = "ComparisonActivity"
    // Change types back to ImageView
    private lateinit var imageViewManual: ImageView
    private lateinit var imageViewAI: ImageView
    private lateinit var buttonGenerateHeatmap: MaterialButton

    // TextViews for scores
    private lateinit var textViewManualLap: TextView
    private lateinit var textViewManualTen: TextView
    private lateinit var textViewManualPBM: TextView
    private lateinit var textViewManualComp: TextView
    private lateinit var textViewAILap: TextView
    private lateinit var textViewAITen: TextView
    private lateinit var textViewAIPBM: TextView
    private lateinit var textViewAIComp: TextView


    private var manualImageUri: Uri? = null
    private var aiImageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.content_comparison)

        // --- Find Views By ID ---
        imageViewManual = findViewById(R.id.imageViewManual) // Assigns to ImageView
        imageViewAI = findViewById(R.id.imageViewAI)         // Assigns to ImageView
        buttonGenerateHeatmap = findViewById(R.id.buttonGenerateHeatmap)

        // ... (rest of findViewById calls for TextViews) ...
        textViewManualLap = findViewById(R.id.textViewManualLap)
        textViewManualTen = findViewById(R.id.textViewManualTen)
        textViewManualPBM = findViewById(R.id.textViewManualPBM)
        textViewManualComp = findViewById(R.id.textViewManualComp)
        textViewAILap = findViewById(R.id.textViewAILap)
        textViewAITen = findViewById(R.id.textViewAITen)
        textViewAIPBM = findViewById(R.id.textViewAIPBM)
        textViewAIComp = findViewById(R.id.textViewAIComp)


        // --- Get URIs and Scores from Intent ---
        // Use getParcelableExtra for URIs if they were passed as Parcelable
        manualImageUri = intent.getParcelableExtra("manual_image_uri")
        aiImageUri = intent.getParcelableExtra("ai_image_uri")

        // If URIs were passed as Strings, parse them:
        // val manualUriString = intent.getStringExtra("manual_image_uri")
        // if (manualUriString != null) {
        //     manualImageUri = Uri.parse(manualUriString)
        // }
        // val aiUriString = intent.getStringExtra("ai_image_uri")
        // if (aiUriString != null) {
        //     aiImageUri = Uri.parse(aiUriString)
        // }


        val manualScores = intent.getSerializableExtra("manual_scores") as? Map<String, Double>
        val aiScores = intent.getSerializableExtra("ai_scores") as? Map<String, Double>


        // --- Display Images ---
        if (manualImageUri != null) {
            Log.d(TAG, "Displaying Manual Image: $manualImageUri")
            imageViewManual.setImageURI(manualImageUri)
        } else {
            Log.e(TAG, "Manual image URI is null")
            imageViewManual.setImageResource(R.drawable.ic_placeholder) // Example placeholder
        }

        if (aiImageUri != null) {
            Log.d(TAG, "Displaying AI Image: $aiImageUri")
            imageViewAI.setImageURI(aiImageUri)
        } else {
            Log.e(TAG, "AI image URI is null")
            imageViewAI.setImageResource(R.drawable.ic_placeholder) // Example placeholder
        }

        // --- Display Scores ---
        displayScores(manualScores, aiScores)


        // --- Set Listener for the Heatmap Button ---
        buttonGenerateHeatmap.setOnClickListener {
            if (manualImageUri != null) {
                Log.d(TAG, "Generate Heatmap button clicked for URI: $manualImageUri")
                val intent = Intent(this, HeatmapActivity::class.java).apply {
                    // Pass the URI as a String, HeatmapActivity expects this key
                    putExtra("manual_image_uri", manualImageUri.toString())
                }
                startActivity(intent)
            } else {
                Log.e(TAG, "Cannot generate heatmap, manual image URI is missing.")
                Toast.makeText(this, "Manual image is not available for heatmap.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ... (displayScores and format functions) ...
    private fun displayScores(manualScores: Map<String, Double>?, aiScores: Map<String, Double>?) {
        textViewManualLap.text = "Lap: ${manualScores?.get("laplacian")?.format(2) ?: "N/A"}"
        textViewManualTen.text = "Ten: ${manualScores?.get("tenengrad")?.format(2) ?: "N/A"}"
        textViewManualPBM.text = "PBM: ${manualScores?.get("pbm")?.format(4) ?: "N/A"}"
        textViewManualComp.text = "Blur Severity: ${manualScores?.get("composite")?.format(1) ?: "N/A"}"

        textViewAILap.text = "Lap: ${aiScores?.get("laplacian")?.format(2) ?: "N/A"}"
        textViewAITen.text = "Ten: ${aiScores?.get("tenengrad")?.format(2) ?: "N/A"}"
        textViewAIPBM.text = "PBM: ${aiScores?.get("pbm")?.format(4) ?: "N/A"}"
        textViewAIComp.text = "Blur Severity: ${aiScores?.get("composite")?.format(1) ?: "N/A"}"
    }

    fun Double.format(digits: Int) = "%.${digits}f".format(this)

}