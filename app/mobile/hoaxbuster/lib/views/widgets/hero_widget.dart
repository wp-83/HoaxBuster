import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';

class HeroWidget extends StatelessWidget {
  final double imgSize;
  final TextStyle fontStyle;
  final String tag;

  const HeroWidget({
    super.key,
    required this.imgSize,
    required this.fontStyle,
    required this.tag,
  });

  @override
  Widget build(BuildContext context) {
    return Hero(
      tag: tag,
      // Tambahkan flightShuttleBuilder supaya transisi halus
      flightShuttleBuilder:
          (flightContext, animation, direction, fromContext, toContext) {
            // Tween untuk scaling logo agar lebih smooth
            return ScaleTransition(
              scale: animation.drive(
                Tween<double>(
                  begin: 1.3,
                  end: 1,
                ).chain(CurveTween(curve: Curves.easeOut)),
              ),
              child: Material(
                color: Colors.transparent,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      "HoaxBuster",
                      style: fontStyle.copyWith(
                        fontWeight: FontWeight.bold,
                        color: primary[80],
                      ),
                    ),
                    const SizedBox(height: 20),
                    Image.asset(
                      "assets/images/logo.png",
                      height: imgSize,
                      width: imgSize,
                    ),
                  ],
                ),
              ),
            );
          },
      child: Column(
        children: [
          Text(
            "HoaxBuster",
            style: fontStyle.copyWith(
              fontWeight: FontWeight.bold,
              color: primary[80],
            ),
          ),
          const SizedBox(height: 20),
          Image.asset(
            "assets/images/logo.png",
            height: imgSize,
            width: imgSize,
          ),
        ],
      ),
    );
  }
}
