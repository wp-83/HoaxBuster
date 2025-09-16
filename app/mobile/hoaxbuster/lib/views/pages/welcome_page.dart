import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:hoaxbuster/views/widget_tree.dart';
import 'package:hoaxbuster/views/widgets/hero_widget.dart';

class WelcomePage extends StatefulWidget {
  const WelcomePage({super.key});

  @override
  State<WelcomePage> createState() => _WelcomePageState();
}

class _WelcomePageState extends State<WelcomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              HeroWidget(
                imgSize: 260.0,
                fontStyle: KTextStyle.Header1,
                tag: 'hero1',
              ),

              SizedBox(height: 40),

              // Subjudul
              Text(
                "Basmi Hoax, Temukan Kepastian!",
                textAlign: TextAlign.center,
                style: KTextStyle.Header5.copyWith(
                  fontWeight: FontWeight.bold,
                  color: accent[100],
                ),
              ),

              SizedBox(height: 24),

              // Tombol
              FractionallySizedBox(
                widthFactor: 0.8,
                child: FilledButton(
                  style: FilledButton.styleFrom(
                    backgroundColor: primary[80],
                    padding: EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  onPressed: () {
                    Navigator.pushReplacement(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 2000),
                        pageBuilder: (_, __, ___) => WidgetTree(),
                      ),
                    );
                  },
                  child:  Text(
                    "Mulai Sekarang!",
                    style: KTextStyle.Header5.copyWith(
                      fontWeight: FontWeight.bold,
                      color: basic[10],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
