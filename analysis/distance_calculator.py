from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

# Add this at the bottom of distance_calculator.py
if __name__ == "__main__":
    # Example coordinates for two locations
    lat1, lon1 = 40.7128, -74.0060  # New York
    lat2, lon2 = 51.5074, -0.1278   # London

    # Call the function and print the result
    distance = calculate_distance(lat1, lon1, lat2, lon2)
    print(f"The distance is {distance} kilometers.")
