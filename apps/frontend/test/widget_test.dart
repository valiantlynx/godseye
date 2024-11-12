import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:godseye_flutter/main.dart';

void main() {
  testWidgets('App loads successfully', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(GodsEyeDashboard());

    // Check if the app displays the main elements correctly.
    expect(find.text("Violence Detection Dashboard"), findsOneWidget);
    expect(find.text("Live Stream"), findsOneWidget);
    expect(find.text("Upload Video"), findsOneWidget);
    expect(find.text("History"), findsOneWidget);
    expect(find.text("Settings"), findsOneWidget);
  });
}
