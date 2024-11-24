import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:js' as js;
import 'camera/camera.dart';
void main() {
  runApp(GodsEyeDashboard());
}

class GodsEyeDashboard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Violence Detection Dashboard",
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  String liveStreamTitle = "Non-Violent";
  double confidence = 0.0; // Confidence should be a double

  @override
  void initState() {
    super.initState();

    // Define a Dart function to update the title and confidence from JS
    void updateDashboardFromJS(dynamic data) {
      try {
        Map<String, dynamic> parsedData;
        print("Received data: $data");

        if (data is String) {
          // If data is a string, attempt to parse it as JSON
          parsedData = json.decode(data) as Map<String, dynamic>;
        } else if (data is Map) {
          // If data is already a map, use it directly
          parsedData = Map<String, dynamic>.from(data);
        } else {
          throw Exception("Unsupported data type received from JavaScript");
        }

        // Debugging: Print the structure of the received data
        print("Parsed data: $parsedData");

        // Extract the confidence value and convert it to a double
        String confidenceStr = parsedData["confidence"] ?? "0.0%"; // Default to "0.0%" if not found
        double confidenceValue = 0.0;

        // Remove the '%' and convert to double
        if (confidenceStr.endsWith('%')) {
          confidenceValue = double.tryParse(confidenceStr.replaceAll('%', '')) ?? 0.0;
        }

        // Update the state with the received data
        setState(() {
          liveStreamTitle = parsedData["result"] ?? "Unknown"; // Update the title dynamically
          confidence = confidenceValue / 100; // Convert to a value between 0 and 1 for display
        });
      } catch (e) {
        print("Failed to decode JSON data: $e");
      }
    }

    // Expose the Dart function to JavaScript
    js.context['updateDashboardFromJS'] = js.allowInterop(updateDashboardFromJS);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          // Sidebar
          Container(
            width: 200,
            color: Colors.blue[600],
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  "God's Eye Dashboard",
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 24),
                SidebarItem(label: "Live Stream"),
                SidebarItem(label: "Upload Video"),
                SidebarItem(label: "History"),
                SidebarItem(label: "Settings"),
                Spacer(),
                Text(
                  "© 2024 God's Eye System",
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),

          // Main Content
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: ListView(
                children: [
                  // Header
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "Violence Detection Dashboard",
                            style: TextStyle(
                                fontSize: 20, fontWeight: FontWeight.bold),
                          ),
                          Text(
                              "Monitor live feeds or process video files for violence detection."),
                        ],
                      ),
                    ),
                  ),

                  SizedBox(height: 16),

                  // Centered dynamic title and confidence
                  Column(
                    children: [
                      SizedBox(height: 16),
                      Center(
                        child: Column(
                          children: [
                            Text(
                              liveStreamTitle,
                              style: TextStyle(
                                fontSize: 45,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              'Confidence: ${(confidence * 100).toStringAsFixed(2)}%', // Apply to double
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.w600,
                                color: confidence > 0.9
                                    ? Colors.green
                                    : Colors.red,
                              ),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(height: 16),
                    ],
                  ),

                  SizedBox(height: 16),

                  // Live Stream Section with dynamic title
                  SectionCard(
                    title: "Live Video Stream",
                    child: Container(
                      height: 1000,
                      color: Colors.grey[300],
                      child: Center(
                        child: CameraWidget(
                            websocketUrl:
                                'ws://localhost:8000/ws/video-stream/'),
                      ),
                    ),
                  ),
                    SizedBox(height: 16),

                  // Additional sections
                  SectionCard(
                    title: "Upload Video for Analysis",
                    child: VideoUploadForm(),
                  ),

                  SizedBox(height: 16),

                  SectionCard(
                    title: "Processed Video",
                    child: Container(
                      height: 200,
                      color: Colors.grey[300],
                      child: Center(
                        child: Text(
                          "Processed Video Placeholder",
                          style: TextStyle(color: Colors.grey[700]),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}


// Sidebar Item
class SidebarItem extends StatelessWidget {
  final String label;

  const SidebarItem({required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Text(
        label,
        style: TextStyle(color: Colors.white, fontSize: 16),
      ),
    );
  }
}

// Section Card
class SectionCard extends StatelessWidget {
  final String title;
  final Widget child;

  const SectionCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            child,
          ],
        ),
      ),
    );
  }
}

// Video Upload Form
class VideoUploadForm extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        TextField(
          decoration: InputDecoration(
            labelText: "Model Path",
            border: OutlineInputBorder(),
          ),
        ),
        SizedBox(height: 8),
        TextField(
          decoration: InputDecoration(
            labelText: "Confidence Threshold",
            border: OutlineInputBorder(),
          ),
          keyboardType: TextInputType.number,
        ),
        SizedBox(height: 8),
        ElevatedButton(
          onPressed: () {
            // Handle video upload and processing here
          },
          child: Text("Process Video"),
        ),
      ],
    );
  }
}
