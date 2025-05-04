plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose) // Correctly applying the compose plugin
}

android {
    namespace = "com.example.dummy"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.dummy"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true // This is correct
        // viewBinding = true // Add this if you use ViewBinding in MainActivity
    }
    // --- Add composeOptions ---
    composeOptions {
        // Use a version compatible with Kotlin 2.0.21 and Compose BOM 2024.05.00
        // Check the official compatibility map. 1.5.14 might be appropriate.
        kotlinCompilerExtensionVersion = "1.5.14" // Adjust if needed based on compatibility map
    }
    // --- Add packagingOptions ---
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}


dependencies {
    // Core KTX - Using string, switch to libs.androidx.core.ktx if preferred
    implementation("androidx.core:core-ktx:1.12.0")
    // Lifecycle - Using string, switch to libs.androidx.lifecycle.runtime.ktx if preferred
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")

    // --- View System Dependencies ---
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // --- CameraX Dependencies ---
    implementation("androidx.camera:camera-core:1.3.0")
    implementation("androidx.camera:camera-camera2:1.3.0")
    implementation("androidx.camera:camera-lifecycle:1.3.0")
    implementation("androidx.camera:camera-view:1.3.0")
    implementation("androidx.camera:camera-video:1.3.0")

    // --- OkHttp Dependency ---
    implementation("com.squareup.okhttp3:okhttp:4.10.0")

    // --- Jetpack Compose Dependencies (Using aliases from libs.versions.toml) ---
    implementation(platform(libs.androidx.compose.bom)) // Apply the BOM
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics) // Needed for Color.kt
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.activity.compose) // Needed for Compose in Activity

    // --- Testing Dependencies (Using aliases from libs.versions.toml) ---
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom)) // BOM for test dependencies
    androidTestImplementation(libs.androidx.ui.test.junit4) // Compose tests

    // --- Debug Dependencies (Using aliases from libs.versions.toml) ---
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)

    implementation("androidx.exifinterface:exifinterface:1.3.7") // Or the latest version
}
