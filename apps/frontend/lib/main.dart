import 'package:flutter/material.dart';

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

class DashboardScreen extends StatelessWidget {
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
                  style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
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
                            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                          ),
                          Text("Monitor live feeds or process video files for violence detection."),
                        ],
                      ),
                    ),
                  ),

                  SizedBox(height: 16),

                  // Live Stream Section
                  SectionCard(
                    title: "Live Video Stream",
                    child: Container(
                      height: 200,
                      color: Colors.grey[300],
                      child: Center(
                        child: Text(
                          "Live Video Stream Placeholder",
                          style: TextStyle(color: Colors.grey[700]),
                        ),
                      ),
                    ),
                  ),

                  SizedBox(height: 16),

                  // Video Upload Section
                  SectionCard(
                    title: "Upload Video for Analysis",
                    child: VideoUploadForm(),
                  ),

                  SizedBox(height: 16),

                  // Status and Result Section
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
