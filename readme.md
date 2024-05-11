# ScanSage

ScanSage is a comprehensive solution designed to enhance security measures by seamlessly integrating machine learning-based ID card detection with real-time analytics and intuitive UI on an Android application.

## Description

ScanSage comprises two main components:

ML Model Component: This component is responsible for detecting whether a person captured by a CCTV or webcam is wearing an ID card or not. Upon detection, the relevant data is uploaded to a MongoDB database.

Android Application Component: The Android application syncs data from the MongoDB database in real-time using a WebSocket instance. It provides real-time analytics through graphs and an intuitive user interface. The application includes features such as authentication through email and password, searching entries of persons wearing ID cards or not, and generating notifications whenever a person is detected without an ID card.

## Installation

To set up ScanSage, follow these steps:

- Clone the repository: git clone https://github.com/your-username/ScanSage.git
- Install the necessary dependencies for both the ML model and the Android application.
Configure the MongoDB connection settings.
Build and run the ML model component.
Install the Android application on your device or emulator.
Tech Stacks
ML Model Component:

- Language: Python
- Libraries: OpenCV, TensorFlow, PyMongo
- Android Application Component:

- Language: Kotlin
- Framework: Android SDK
- Libraries: Realm Sync, MPAndroidChart

## Features

ML Model Component:

Detects whether a person is wearing an ID card.
Uploads detection data to a MongoDB database.
Android Application Component:

Real-time sync with MongoDB using WebSocket.
Provides real-time analytics through graphs.
User authentication through email and password.
Search functionality for entries of persons wearing ID cards or not.
Generates notifications for ID card violations.

## Development Setup

To contribute to ScanSage development, follow these steps:

- Fork the repository to your GitHub account.
- Clone your fork: git clone https://github.com/your-username/ScanSage.git
- Create a new branch for your feature or bug fix: git checkout -b feature-or-fix-name
- Make your changes and commit them: git commit -m "Your commit message"
- Push your changes to your fork: git push origin feature-or-fix-name
- Create a pull request from your fork's branch to the main repository's branch.

## Usage

1. ML Model Component:

-  Train the ML model on a dataset of images containing persons with and without ID cards.
- Deploy the trained model to detect ID card violations in real-time.

2. Android Application Component:

- Log in to the application using email and password credentials.
- View real-time analytics on the dashboard.
- Search for entries based on ID card status.
- Receive notifications for ID card violations.
