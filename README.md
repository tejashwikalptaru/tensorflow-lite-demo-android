# Tensorflow Lite Android Demo app

An android application with Tensorflow lite, Mobilenet and Inception models.
Uses the mobile camera for preview and image snapshot for image classification.


**Dependency:**
> implementation 'org.tensorflow:tensorflow-lite:+'

The demo application uses the [camera2](https://developer.android.com/reference/android/hardware/camera2/package-summary) API for interaction with mobile camera and image capture, then the captured image is passed as a bitmap to the classifier.

  

###