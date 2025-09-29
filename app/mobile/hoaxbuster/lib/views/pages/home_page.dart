import 'dart:convert';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:hoaxbuster/views/widgets/hero_widget.dart';
import 'result_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController _controller = TextEditingController();
  bool _isLoading = false;

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
      // ✅ pake endpoint yang benar
      final response = await http.post(
        Uri.parse("https://william83.pythonanywhere.com/predict"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"information": text}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        // cek struktur JSON
        print("Response: $data");

        // ✅ Ambil prediksi dari API
        final prediction = double.parse(data["prediction"].toString());

        if (!mounted) return;
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultPage(resultText: text, prediction: prediction),
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
    return Scaffold(
      body: SingleChildScrollView(
        padding: EdgeInsets.only(top: 60.0),
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              HeroWidget(
                imgSize: 216.0,
                fontStyle: KTextStyle.Header2,
                tag: 'hero1',
              ),
        
              const SizedBox(height: 60.0),
        
              Text(
                "Mau periksa informasi apa?",
                style: KTextStyle.Header6.copyWith(fontWeight: FontWeight.bold),
              ),
        
              const SizedBox(height: 16),
        
              TextField(
                controller: _controller,
                keyboardType: TextInputType.multiline,
                minLines: 1,
                maxLines: null,
                decoration: InputDecoration(
                  hintText: "Ketik informasi mu di sini...",
                  hintStyle: TextStyle(color: basic[40]),
                  filled: true,
                  fillColor: info[10],
                  contentPadding: const EdgeInsets.symmetric(
                    vertical: 16,
                    horizontal: 20,
                  ),
                  suffixIcon: InkWell(
                    onTap: _isLoading ? null : _checkHoax,
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
                          : Icon(Icons.arrow_forward, color: basic[10]),
                    ),
                  ),
                  suffixIconConstraints: const BoxConstraints(
                    minHeight: 48,
                    minWidth: 64,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),

              SizedBox(height:  20,),

              Text(
                "Untuk sementara hanya berfokus pada politik",
                style: KTextStyle.Header7.copyWith(
                  color: primary[100],
                  fontWeight: FontWeight.bold,
                ),
              ),

              SizedBox(height: 100),
            ],
          ),
        ),
      ),
    );
  }
}
