package com.example.dummy

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MenuItem
import android.widget.FrameLayout
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import com.google.android.material.navigation.NavigationView
import android.widget.Toast // Import Toast

abstract class BaseDrawerActivity : AppCompatActivity(), NavigationView.OnNavigationItemSelectedListener {

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var navigationView: NavigationView
    private lateinit var toolbar: Toolbar
    private lateinit var toggle: ActionBarDrawerToggle

    // Abstract property to be implemented by child activities
    // to provide their specific content layout resource ID
    abstract val contentLayoutId: Int

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Set the base layout first
        setContentView(R.layout.activity_base_drawer)

        // --- Initialize Base Views ---
        toolbar = findViewById(R.id.toolbar)
        drawerLayout = findViewById(R.id.drawer_layout)
        navigationView = findViewById(R.id.nav_view)

        // --- Inflate Child Activity's Content ---
        val contentFrame: FrameLayout = findViewById(R.id.content_frame)
        // Use LayoutInflater to add the child's layout into the FrameLayout
        LayoutInflater.from(this).inflate(contentLayoutId, contentFrame, true)

        // --- Setup Toolbar ---
        setSupportActionBar(toolbar)

        // --- Setup Navigation Drawer ---
        toggle = ActionBarDrawerToggle(
            this,
            drawerLayout,
            toolbar, // Pass the toolbar here
            R.string.navigation_drawer_open,
            R.string.navigation_drawer_close
        )
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState() // Important: This displays the hamburger icon

        // --- Handle Navigation Item Clicks ---
        navigationView.setNavigationItemSelectedListener(this) // Use 'this' as the listener
    }

    // --- Handle Navigation Item Selection ---
    override fun onNavigationItemSelected(item: MenuItem): Boolean {
        var intent: Intent? = null
        val currentActivity = this::class.java // Get the class of the current activity

        when (item.itemId) {
            R.id.nav_home -> { // Handle the Home button
                // Navigate to StartActivity (your start page)
                if (currentActivity != StartActivity::class.java) { // Check if not already on StartActivity
                    intent = Intent(this, StartActivity::class.java) // Create Intent for StartActivity
                }
            }
            R.id.nav_capture -> {
                // Assuming MainActivity is the capture screen
                if (currentActivity != MainActivity::class.java) {
                    intent = Intent(this, MainActivity::class.java)
                }
            }
            R.id.nav_blur_severity -> {
                // Navigate to UnblurActivity
                if (currentActivity != UnblurActivity::class.java) {
                     intent = Intent(this, UnblurActivity::class.java)
                }
            }
            R.id.nav_heatmap -> {
                 // Navigate to HeatmapUploadActivity
                if (currentActivity != HeatmapUploadActivity::class.java) {
                    intent = Intent(this, HeatmapUploadActivity::class.java)
                }
            }
        }

        if (intent != null) {
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            startActivity(intent)
        }

        drawerLayout.closeDrawer(GravityCompat.START)
        return true // Indicate item selection was handled
    }

    // --- Handle back press to close drawer first ---
    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (drawerLayout.isDrawerOpen(GravityCompat.START)) {
            drawerLayout.closeDrawer(GravityCompat.START)
        } else {
            super.onBackPressed()
        }
    }

    // Optional: Helper to set the checked item in the drawer
    protected fun setCheckedNavigationItem(itemId: Int) {
        navigationView.setCheckedItem(itemId)
    }
}