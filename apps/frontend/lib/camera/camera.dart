import 'dart:html' as html;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'dart:js' as js;

class CameraWidget extends StatelessWidget {
  final String websocketUrl;

  CameraWidget({required this.websocketUrl}) {
    // Initialize WebSocket in JavaScript
    js.context.callMethod('startWebSocket', [websocketUrl]);

    // Register the 'webcamVideo' HTML view for displaying the webcam stream
    ui.platformViewRegistry.registerViewFactory(
      'webcamVideo',
      (int viewId) {
        final videoElement = html.VideoElement()
          ..id = 'webcamforstreaming'
          ..autoplay = true
          ..setAttribute('playsinline', 'true')
          ..style.width = '100%'
          ..style.height = '100%';

        // Access the webcam and set it as the video source
        html.window.navigator.mediaDevices?.getUserMedia({'video': true}).then((stream) {
          videoElement.srcObject = stream;
          // Start capturing frames from the video element
          js.context.callMethod('captureAndSendFrame', ['webcamforstreaming']);
        }).catchError((error) {
          print("Error accessing webcam: $error");
        });

        return videoElement;
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(viewType: 'webcamVideo');
  }
}
