import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/views/pages/welcome_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: "Poppins",
        colorScheme: ColorScheme.fromSeed(
          seedColor: info,
          brightness: Brightness.light,
        ),
      ),
      home: WelcomePage(),
    );
  }
}


