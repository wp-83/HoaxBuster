class Activity {
  final String title;
  final double percentage;

  Activity({
    required this.title,
    required this.percentage,
  });

  Map<String, dynamic> toJson() => {
        "title": title,
        "percentage": percentage,
      };

  factory Activity.fromJson(Map<String, dynamic> json) => Activity(
        title: json["title"],
        percentage: json["percentage"],
      );
}
