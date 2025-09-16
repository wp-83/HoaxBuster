import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/views/widgets/gaugage_chart_widget.dart';

class ResultPage extends StatefulWidget {
  final String resultText;
  final double prediction;

  const ResultPage({
    super.key,
    required this.resultText,
    required this.prediction,
  });

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  late TextEditingController _controller;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    // ✅ isi controller dengan hasil pencarian sebelumnya
    _controller = TextEditingController(text: widget.resultText);
    _saveHistory();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final historyString = prefs.getString("activities");
    List history = historyString != null ? jsonDecode(historyString) : [];

    final newItem = {
      "title": widget.resultText,
      "percentage": (widget.prediction * 100),
      "timestamp": DateTime.now().toIso8601String(),
    };

    history.add(newItem);
    await prefs.setString("activities", jsonEncode(history));
  }

  Future<void> _checkHoax() async {
    final text = _controller.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Tolong ketik informasi dulu.")),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final response = await http.post(
        Uri.parse("https://william83.pythonanywhere.com/predict"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"information": text}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final prediction = double.parse(data["prediction"].toString());

        if (!mounted) return;
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) =>
                ResultPage(resultText: text, prediction: prediction),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: ${response.statusCode}")),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Terjadi kesalahan: $e")));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final percentage = (widget.prediction * 100).toStringAsFixed(2);

    // Tentukan warna zona
    Widget warningText;

    if (widget.prediction >= 0.8) {
      warningText = Text.rich(
        textAlign: TextAlign.justify,
        TextSpan(
          children: [
            TextSpan(
              text:
                  "Analisis menunjukkan indikasi hoax sebesar ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "$percentage%",
              style: KTextStyle.Header6.copyWith(
                color: primary[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: " dan ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "masuk zona merah",
              style: KTextStyle.Header6.copyWith(
                color: primary[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: ". Informasi ini terindikasi hoax. ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "Jangan langsung disebar ya!",
              style: KTextStyle.Header6.copyWith(
                color: primary[100],
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        style: KTextStyle.Header6,
      );
    } else if (widget.prediction >= 0.6) {
      warningText = Text.rich(
        textAlign: TextAlign.justify,
        TextSpan(
          children: [
            TextSpan(
              text:
                  "Analisis menunjukkan indikasi hoax sebesar ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "$percentage%",
              style: KTextStyle.Header6.copyWith(
                color: primary[60],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: " dan ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "masuk zona merah",
              style: KTextStyle.Header6.copyWith(
                color: primary[60],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: ". Informasi ini terindikasi hoax. ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "Jangan langsung disebar ya!",
              style: KTextStyle.Header6.copyWith(
                color: primary[60],
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        style: KTextStyle.Header6,
      );
    } else if (widget.prediction >= 0.4) {
      warningText = Text.rich(
        textAlign: TextAlign.justify,
        TextSpan(
          children: [
            TextSpan(
              text: "Analisis menunjukkan indikasi hoax sebesar ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "$percentage%",
              style: KTextStyle.Header6.copyWith(
                color: warning[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: " dan ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "masuk zona kuning",
              style: KTextStyle.Header6.copyWith(
                color: warning[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: ". Kamu harus waspada dengan informasi ini. ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "Perlu diteliti lebih lanjut.",
              style: KTextStyle.Header6.copyWith(
                color: warning[100],
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        style: KTextStyle.Header6,
      );
    } else if (widget.prediction >= 0.2) {
      warningText = Text.rich(
        textAlign: TextAlign.justify,
        TextSpan(
          children: [
            TextSpan(
              text: "Analisis menunjukkan indikasi hoax sebesar ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "$percentage%",
              style: KTextStyle.Header6.copyWith(
                color: safe[80],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: " dan ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "masuk zona hijau",
              style: KTextStyle.Header6.copyWith(
                color: safe[80],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text:
                  ". Kamu bisa percaya pada informasi ini, tapi tetap cek sumber lain agar semakin valid ya!",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
          ],
        ),
        style: KTextStyle.Header6,
      );
    } else {
      warningText = Text.rich(
        textAlign: TextAlign.justify,
        TextSpan(
          children: [
            TextSpan(
              text: "Analisis menunjukkan indikasi hoax sebesar ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "$percentage%",
              style: KTextStyle.Header6.copyWith(
                color: safe[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text: " dan ",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
            TextSpan(
              text: "masuk zona hijau",
              style: KTextStyle.Header6.copyWith(
                color: safe[100],
                fontWeight: FontWeight.bold,
              ),
            ),
            TextSpan(
              text:
                  ". Kamu bisa percaya pada informasi ini, tapi tetap cek sumber lain agar semakin valid ya!",
              style: KTextStyle.Header6.copyWith(
                color: basic,
              ),
            ),
          ],
        ),
        style: KTextStyle.Header6,
      );
    }

    return Scaffold(
      backgroundColor: basic[10],
      appBar: AppBar(
        title: const Text("Hasil Analisis"),
        backgroundColor: basic[10],
        foregroundColor: basic[100],
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // ✅ TextField dengan isi hasil sebelumnya
            TextField(
              controller: _controller,
              keyboardType: TextInputType.multiline,
              minLines: 1,
              maxLines: null,
              style: TextStyle(
                fontSize: 16,
                color: basic[100],
              ),
              decoration: InputDecoration(
                hintText: "Ketik informasi mu di sini...",
                hintStyle: TextStyle(
                  color: basic[40],
                  fontSize: 16,
                ),
                filled: true,
                fillColor: info[10],
                contentPadding: const EdgeInsets.symmetric(
                  vertical: 16,
                  horizontal: 20,
                ),
                suffixIcon: InkWell(
                  onTap: _isLoading ? null : _checkHoax,
                  borderRadius: BorderRadius.circular(30),
                  child: Container(
                    margin: const EdgeInsets.all(6),
                    decoration: BoxDecoration(
                      color: secondary[100],
                      shape: BoxShape.circle,
                    ),
                    child: _isLoading
                        ? const Padding(
                            padding: EdgeInsets.all(12),
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : Icon(
                            Icons.arrow_forward,
                            color: basic[10],
                          ),
                  ),
                ),
                suffixIconConstraints: const BoxConstraints(
                  minHeight: 48,
                  minWidth: 64,
                ),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(30),
                  borderSide: BorderSide.none,
                ),
              ),
            ),

            const SizedBox(height: 28),

            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                "Analisis Tingkat Hoax",
                style: KTextStyle.Header6.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            const SizedBox(height: 8),

            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                textAlign: TextAlign.justify,
                "Grafik berikut menunjukkan tingkat hoax berdasarkan analisis AI. "
                "Semakin tinggi nilainya, semakin besar kemungkinan informasi tersebut hoax.",
                style: KTextStyle.Header6,
              ),
            ),

            const SizedBox(height: 20),

            GaugageChartWidget(hoaxValue: widget.prediction * 100),

            const SizedBox(height: 36),

            Align(
              alignment: Alignment.centerLeft,
              child: warningText,
            ),

            SizedBox(height: 40.0),
          ],
        ),
      ),
    );
  }
}
