import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:hoaxbuster/data/notifiers.dart';
import 'package:hoaxbuster/views/pages/aboutus_page.dart';
import 'package:hoaxbuster/views/pages/home_page.dart';
import 'package:hoaxbuster/views/pages/recentactivity_page.dart';
import 'package:hoaxbuster/views/widgets/navbar_widget.dart';
import 'dart:async';
import 'package:intl/intl.dart';

List<Widget> pages = [HomePage(), RecentactivityPage(), AboutusPage()];

class WidgetTree extends StatefulWidget {
  const WidgetTree({super.key});

  @override
  State<WidgetTree> createState() => _WidgetTreeState();
}

class _WidgetTreeState extends State<WidgetTree> {

  late String _timeString = DateFormat('HH:mm:ss').format(DateTime.now());
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    _timeString = _formatDateTime(DateTime.now());
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _timeString = _formatDateTime(DateTime.now());
      });
    });
  }

  String _formatDateTime(DateTime dateTime) {
    return DateFormat('HH:mm:ss WIB').format(dateTime);
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(80.0),
        child: Material(
          elevation: 4, // shadow
          borderRadius: const BorderRadius.vertical(
            bottom: Radius.circular(28),
          ),
          child: AppBar(
            toolbarHeight: 80.0,
            backgroundColor: info[10],
            elevation: 0, // hapus shadow bawaan, biar pakai Material di luar
            shape: const RoundedRectangleBorder(
              borderRadius: BorderRadius.vertical(
                bottom: Radius.circular(28),
              ),
            ),

            leading: Padding(
              padding: const EdgeInsets.only(left: 20.0),
              child: Image.asset(
                'assets/images/garuda_pancasila_nobg.png',
                fit: BoxFit.contain,
                height: 360,
              ),
            ),

            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 16.0),
                child: Center(
                  child: Text(
                    _timeString,
                    style: KTextStyle.Header5.copyWith(
                      fontWeight: FontWeight.bold,
                      color: accent[100],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),

      body: ValueListenableBuilder(
        valueListenable: selectedPageNotifier,
        builder: (context, selectedPage, child) {
          return pages.elementAt(selectedPage);
        },
      ),
      bottomNavigationBar: NavbarWidget(),
    );
  }
}