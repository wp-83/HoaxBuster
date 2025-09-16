import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/notifiers.dart';

class NavbarWidget extends StatelessWidget {
  const NavbarWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder(
      valueListenable: selectedPageNotifier,
      builder: (context, selectedPage, child) {
        return Container(
          decoration: BoxDecoration(
            boxShadow: [
              BoxShadow(
                color: Colors.black26,   // warna shadow
                blurRadius: 16,          // tingkat blur
                spreadRadius: 2,         // sebaran shadow
                offset: Offset(0, -2),   // arah shadow (ke atas karena navbar di bawah)
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(28),
              topRight: Radius.circular(28),
            ),
            child: NavigationBar(
              height: 80,
              backgroundColor: info[10],
              indicatorColor: info[40],
              labelBehavior: NavigationDestinationLabelBehavior.onlyShowSelected,
              selectedIndex: selectedPage,
              onDestinationSelected: (int value) {
                selectedPageNotifier.value = value;
              },
              destinations: const [
                NavigationDestination(icon: Icon(Icons.home), label: 'Home'),
                NavigationDestination(icon: Icon(Icons.history), label: 'Recent'),
                NavigationDestination(icon: Icon(Icons.info_outline), label: 'About Us'),
              ],
            ),
          ),
        );
      },
    );
  }
}