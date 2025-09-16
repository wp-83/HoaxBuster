import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:syncfusion_flutter_gauges/gauges.dart';
import 'package:hoaxbuster/data/app_colors.dart';

class GaugageChartWidget extends StatelessWidget {
  final double hoaxValue; // nilai 0 - 100

  const GaugageChartWidget({super.key, required this.hoaxValue});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Gauge Chart
        SizedBox(
          height: 250,
          child: SfRadialGauge(
            axes: [
              RadialAxis(
                minimum: 0,
                maximum: 100,
                showTicks: false,
                showLabels: false,
                axisLineStyle: AxisLineStyle(thickness: 60, color: basic[20]),
                ranges: [
                  GaugeRange(
                    startValue: 0,
                    endValue: 20,
                    color: safe[100]!,
                    endWidth: 44,
                    startWidth: 44,
                  ),
                  GaugeRange(
                    startValue: 20,
                    endValue: 40,
                    color: safe[80]!,
                    endWidth: 44,
                    startWidth: 44,
                  ),
                  GaugeRange(
                    startValue: 40,
                    endValue: 60,
                    color: warning[100]!,
                    endWidth: 44,
                    startWidth: 44,
                  ),
                  GaugeRange(
                    startValue: 60,
                    endValue: 80,
                    color: primary[60]!,
                    endWidth: 44,
                    startWidth: 44,
                  ),
                  GaugeRange(
                    startValue: 80,
                    endValue: 100,
                    color: primary[100]!,
                    endWidth: 44,
                    startWidth: 44,
                  ),
                ],
                pointers: [
                  NeedlePointer(
                    value: hoaxValue,
                    needleEndWidth: 12,
                    needleLength: 0.8,
                    lengthUnit: GaugeSizeUnit.factor,
                    enableAnimation: true,            // aktifkan animasi
                    animationDuration: 3000,          // durasi 1.5 detik
                    animationType: AnimationType.easeOutBack, // animasi lebih halus & ada efek mantul
                    knobStyle: const KnobStyle(
                      color: Colors.brown,
                      knobRadius: 0.12,
                    ),
                  ),
                ],
                annotations: [
                  // Tambah emoji di setiap zona
                  GaugeAnnotation(
                    angle: 160,
                    positionFactor: 0.8,
                    widget: FaIcon(
                      FontAwesomeIcons.faceLaugh,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                  GaugeAnnotation(
                    angle: 212,
                    positionFactor: 0.8,
                    widget: FaIcon(
                      FontAwesomeIcons.faceSmile,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                  GaugeAnnotation(
                    angle: 270,
                    positionFactor: 0.8,
                    widget: FaIcon(
                      FontAwesomeIcons.faceMeh,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                  GaugeAnnotation(
                    angle: 328,
                    positionFactor: 0.8,
                    widget: FaIcon(
                      FontAwesomeIcons.faceFrown,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                  GaugeAnnotation(
                    angle: 380,
                    positionFactor: 0.8,
                    widget: FaIcon(
                      FontAwesomeIcons.faceSurprise,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),

        Wrap(
          alignment: WrapAlignment.center,
          spacing: 16,
          runSpacing: 8,
          children: [
            _LegendItem(color: safe, text: "Sangat Valid"),
            _LegendItem(color: Color(0xFF66BB6A), text: "Cenderung Valid"),
            _LegendItem(color: warning, text: "Meragukan"),
            _LegendItem(color: Color(0xFFFF6666), text: "Cenderung Hoax"),
            _LegendItem(color: primary, text: "Hoax / Tidak Valid"),
          ],
        ),
      ],
    );
  }
}

// Widget legend item
class _LegendItem extends StatelessWidget {
  final Color color;
  final String text;

  const _LegendItem({required this.color, required this.text});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 14,
          height: 14,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500)),
      ],
    );
  }
}
