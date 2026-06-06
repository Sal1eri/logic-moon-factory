from lunar_path.environment import Scenario


def default_scenarios():
    return [
        Scenario("baseline", "Base Plain With Sparse Rocks", 45, 101, 4, 0.018, 2, 1, ((8, 8), (36, 10)), 1.0, (40, 4), (5, 40), 50000, 90.0),
        Scenario("crater_field", "Dense Crater Field", 45, 202, 10, 0.018, 2, 1, ((7, 7), (37, 35)), 1.0, (40, 4), (5, 40), 60000, 90.0),
        Scenario("slope_ridges", "Highland Ridges And Slopes", 45, 303, 6, 0.014, 2, 1, ((8, 36),), 2.2, (40, 4), (5, 40), 60000, 92.0),
        Scenario("shadow_comm", "Polar Shadow And Communication Gaps", 45, 404, 6, 0.014, 3, 5, ((8, 8),), 1.35, (40, 4), (5, 40), 70000, 118.0),
        Scenario("complex_moon", "Integrated Lunar Industrial Terrain", 45, 505, 9, 0.022, 5, 4, ((6, 8), (38, 35)), 1.8, (40, 4), (5, 40), 80000, 108.0),
        Scenario("low_battery_bad_case", "Low Battery Bad Case", 45, 606, 7, 0.016, 5, 4, ((8, 8),), 1.85, (40, 4), (5, 40), 70000, 112.0),
    ]
