import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:hoaxbuster/data/activity_model.dart';
import 'package:hoaxbuster/data/app_colors.dart';
import 'package:hoaxbuster/data/constants.dart';
import 'package:shared_preferences/shared_preferences.dart';

class RecentactivityPage extends StatefulWidget {
  const RecentactivityPage({super.key});

  @override
  State<RecentactivityPage> createState() => _RecentactivityPageState();
}

class _RecentactivityPageState extends State<RecentactivityPage> {
  List<Activity> activities = [];

  @override
  void initState() {
    super.initState();
    _loadActivities();
  }

  Future<void> _loadActivities() async {
    final prefs = await SharedPreferences.getInstance();
    final historyString = prefs.getString("activities");

    if (historyString != null) {
      List decoded = jsonDecode(historyString);
      setState(() {
        activities = decoded.map((e) => Activity.fromJson(e)).toList();
      });
    }
  }

  Future<void> _saveActivities() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = jsonEncode(activities.map((e) => e.toJson()).toList());
    await prefs.setString("activities", encoded);
  }

  void _deleteActivity(int index) async {
    setState(() {
      activities.removeAt(index);
    });
    await _saveActivities();
  }

  Color? _getColor(double value) {
    if (value > 80) return primary[100];
    if (value > 60) return primary[60];
    if (value > 40) return warning[80];
    if (value > 20) return safe[80];
    return safe[100];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                Image.asset(
                  'assets/images/logo.png',
                  height: 28.0,
                ),
                Container(
                  height: 32.0,
                  child: const VerticalDivider(
                    thickness: 2.0,
                  ),
                ),
                Text(
                  "HoaxBuster",
                  style: KTextStyle.Header5.copyWith(
                    color: primary[100],
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 20.0)
              ],
            ),

            const SizedBox(height: 40),

            Text(
              "RECENT ACTIVITY",
              style: KTextStyle.Header3.copyWith(
                fontWeight: FontWeight.bold,
                color: primary[80],
              ),
            ),
            const SizedBox(height: 20),

            // list aktivitas
            Expanded(
              child: activities.isEmpty
                  ? const Center(
                      child: Text(
                        "Belum ada aktivitas.",
                        style: TextStyle(color: basic),
                      ),
                    )
                  : ListView.builder(
                      itemCount: activities.length,
                      itemBuilder: (context, index) {
                        final item = activities[index];
                        final percentage = item.percentage;

                        return Container(
                          margin: const EdgeInsets.only(bottom: 16),
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: info[10],
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                color: basic.withOpacity(0.1),
                                blurRadius: 8,
                                offset: const Offset(2, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Expanded(
                                child: Text(
                                  item.title,
                                  style: KTextStyle.Header6.copyWith(
                                    fontWeight: FontWeight.w500,
                                    color: basic,
                                  ),
                                ),
                              ),

                              SizedBox(width: 16),

                              Row(
                                children: [
                                  Text(
                                    "${percentage.toStringAsFixed(2)}%",
                                    style: KTextStyle.Header6.copyWith(
                                      fontWeight: FontWeight.bold,
                                      color: _getColor(percentage),
                                    ),
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.delete,
                                        color: primary),
                                    onPressed: () => _deleteActivity(index),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
