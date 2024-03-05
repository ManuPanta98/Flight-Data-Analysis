import unittest
from Group_16.analysis.distance_calculator import calculate_distance

class TestCalculateDistance(unittest.TestCase):
    def test_distance_new_york_to_london(self):
        # Coordinates for JFK Airport (New York) and Heathrow Airport (London)
        distance = calculate_distance(40.6413111, -73.7781391, 51.470020, -0.454295)
        # Check if the distance is within a reasonable range, say +/- 10 kilometers
        self.assertAlmostEqual(distance, 5556, delta=10)

    def test_distance_within_same_continent(self):
        # Test case within the same continent, for example, LAX to JFK
        distance = calculate_distance(33.9416, -118.4085, 40.6413111, -73.7781391)
        # Assert that it's within a range of +/- 10 kilometers
        self.assertAlmostEqual(distance, 3945, delta=10)

    def test_distance_different_continents(self):
        # Test case between two airports on different continents, for example, Sydney to Johannesburg
        distance = calculate_distance(-33.8688, 151.2093, -26.1392, 28.246)
        # Assert that it's within a range of +/- 10 kilometers
        self.assertAlmostEqual(distance, 11016, delta=10)

if __name__ == '__main__':
    unittest.main()
