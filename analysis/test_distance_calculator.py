# pylint: disable=wrong-import-position,import-error,redefined-outer-name
# Adjustments for local imports and linting preferences

import sys
import os
import pandas as pd
import pytest

# Adjust the path to include the directory where your scripts are located
from distance_calculator import calculate_distance
from flightanalysis import FlightAnalysis

# Initialize FlightAnalysis class instance
fa = FlightAnalysis()

# Load the route distances if they're not already calculated and saved
routes_file_path = "./downloads/updated_routes.csv"
if not os.path.exists(routes_file_path):
    fa.update_route_distances(save=True, path=routes_file_path)
else:
    fa.update_route_distances(load_from=routes_file_path)

def test_data_loading():
    """Ensure that all datasets are loaded correctly into pandas DataFrames."""
    assert isinstance(fa.airport_data, pd.DataFrame)
    assert isinstance(fa.flight_routes, pd.DataFrame)

def test_distance_calculation_accuracy():
    """Check the distance calculation accuracy against known distances.
    data was taken from https://latlongdata.com/distance-calculator/
    """
    
    source_lat, source_lon = 50.962101,	1.954760  # Calais-Dunkerque Airport
    dest_lat, dest_lon = 52.308601,	4.763890  # Amsterdam Airport Schiphol      
    # Known distance for these coordinates, replace with a calculated or known value
    expected_distance =  245
    calculated_distance = calculate_distance(source_lat, source_lon, dest_lat, dest_lon)
    assert calculated_distance == pytest.approx(expected_distance, rel=0.01)
    
def test_distance_calculation_intercontinental_accuracy():
    """Check the distance calculation accuracy against known distances in different continents."""
    
    source_lat, source_lon = 52.308601,	4.763890 # Amsterdam Airport Schiphol
    dest_lat, dest_lon = 46.485001, -84.509399 # Sault Ste Marie Airport (CANADA)
    # Known distance for these coordinates, replace with a calculated or known value
    expected_distance =  6071
    calculated_distance = calculate_distance(source_lat, source_lon, dest_lat, dest_lon)
    assert calculated_distance == pytest.approx(expected_distance, rel=0.01)

def test_zero_distance():
    """Ensure that the distance calculation returns zero when the source and destination are the same."""
    airport_lat, airport_lon = 37.466801, 15.0664  # Catania-Fontanarossa Airport
    distance = calculate_distance(airport_lat, airport_lon, airport_lat, airport_lon)
    assert distance == pytest.approx(0, abs=1e-6)
