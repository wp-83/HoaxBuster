import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:hoaxbuster/views/widgets/hero_widget.dart';

class AboutusPage extends StatelessWidget {
  const AboutusPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.only(
          top: 60.0,
          left: 16,
          right: 16,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Logo / Hero
            HeroWidget(
              imgSize: 260.0,
              fontStyle: KTextStyle.Header1,
              tag: 'hero1',
            ),

            const SizedBox(height: 40),

            // Judul
            Text(
              "Basmi Hoax, Temukan Kepastian!",
              textAlign: TextAlign.center,
              style: KTextStyle.Header5.copyWith(
                fontWeight: FontWeight.bold,
                color: accent[100],
              ),
            ),

            const SizedBox(height: 24),

            // Deskripsi
            Text(
              "Aplikasi ini hadir untuk membantu masyarakat Indonesia "
              "dalam mengenali dan mewaspadai informasi hoax yang ada "
              "di sekitar mereka.",
              textAlign: TextAlign.center,
              style: KTextStyle.Header6.copyWith(
                color: accent[100],
              ),
            ),

            const SizedBox(height: 40),

            // Team Members (responsive dengan Wrap)
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 48.0,
              runSpacing: 36.0,
              children: const [
                TeamMember(
                  name: 'Lucia Sherina N. K.',
                  imageUrl: 'assets/images/developers/1.png',
                ),
                TeamMember(
                  name: 'William Fernando S.',
                  imageUrl: 'assets/images/developers/2.png',
                ),
                TeamMember(
                  name: 'William Pratama',
                  imageUrl: 'assets/images/developers/3.png',
                ),
                TeamMember(
                  name: 'Pandu Wicaksono',
                  imageUrl: 'assets/images/developers/4.png',
                ),
              ],
            ),

            const SizedBox(height: 80),
          ],
        ),
      ),
    );
  }
}

class TeamMember extends StatelessWidget {
  final String name;
  final String? imageUrl;

  const TeamMember({
    super.key,
    required this.name,
    this.imageUrl,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        CircleAvatar(
          radius: 74.0,
          backgroundColor: warning[10],
          child: ClipOval(
            child: Image.asset(
              imageUrl ?? 'assets/images/default.png',
              height: 152,
              width: 140,
              fit: BoxFit.contain,   // isi penuh lingkaran
              alignment: Alignment.topCenter, // 👉 atur posisi (topCenter, bottomCenter, dll)
            ),
          ),
        ),
        
        const SizedBox(height: 12),
        Text(
          name,
          style: KTextStyle.Header6.copyWith(
            color: accent[100],
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
