package com.example.dummy

import android.content.Intent // Keep this
import android.os.Bundle
import android.util.Log // Keep this
import android.view.View // Keep this
import android.widget.Button // Keep this (or remove if using CardView click)
// import android.widget.Toast // Keep if needed for other things

// Remove import androidx.appcompat.app.AppCompatActivity if present
import com.google.android.material.card.MaterialCardView // Add if making cards clickable

// Change inheritance to BaseDrawerActivity
class StartActivity : BaseDrawerActivity() {

    private val TAG = "StartActivity"
    // SERVER_URL is no longer needed here
    // private val SERVER_URL = "..."

    // Remove buttonUploadImage if making card clickable
    // private lateinit var buttonUploadImage: Button
    private lateinit var buttonOpenCamera: Button // Keep this (or use card click)
    // Remove imageViewResult as it's moved to UnblurActivity
    // private lateinit var imageViewResult: ImageView

    // Remove OkHttp Client if not used for anything else here
    // private val httpClient = OkHttpClient()

    // Remove Activity Result Launcher
    // private val pickImageLauncher = ...

    // --- Add CardView variables if making cards clickable ---
    private lateinit var cardUpload: MaterialCardView
    private lateinit var cardCamera: MaterialCardView
    // -------------------------------------------------------

    // Implement the abstract property from BaseDrawerActivity
    override val contentLayoutId: Int
        get() = R.layout.content_start // Use the renamed layout file

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // REMOVE setContentView(R.layout.activity_start) - Base class handles layout

        // --- REMOVE START BUTTON CODE ---
        // val startButton: Button = findViewById(R.id.startButton) // REMOVE THIS LINE
        // startButton.setOnClickListener {                         // REMOVE THIS BLOCK
        //     val intent = Intent(this, MainActivity::class.java)
        //     startActivity(intent)
        // }
        // --------------------------------

        // Set the checked item for this activity in the drawer
        setCheckedNavigationItem(R.id.nav_home)

        // --- Initialize CardViews ---
        cardUpload = findViewById(R.id.cardUpload)
        cardCamera = findViewById(R.id.cardCamera)
        // ---------------------------

        // --- Set OnClickListener on the CardView ---
        cardUpload.setOnClickListener {
            Log.d(TAG, "Upload Card Clicked - Launching UnblurActivity")
            val intent = Intent(this, UnblurActivity::class.java)
            startActivity(intent)
        }
        // ------------------------------------------

        // --- Set OnClickListener on the CardView ---
        cardCamera.setOnClickListener {
            Log.d(TAG, "Camera Card Clicked - Launching MainActivity")
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
        // ------------------------------------------

        // Remove old button listeners if cards are used
        /*
        buttonUploadImage.setOnClickListener {
            // OLD LOGIC REMOVED
            // openImagePicker()
            Log.d(TAG, "Upload Button Clicked - Launching UnblurActivity")
            val intent = Intent(this, UnblurActivity::class.java)
            startActivity(intent)
        }

        buttonOpenCamera.setOnClickListener {
            Log.d(TAG, "Camera Button Clicked - Launching MainActivity")
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
        */
    }

    // Remove openImagePicker function
    // private fun openImagePicker() { ... }

    // Remove uploadImageForUnblurring function
    // private fun uploadImageForUnblurring(imageUri: Uri) { ... }
}