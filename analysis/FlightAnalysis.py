import pandas as pd
from pydantic import BaseModel, Field
import datetime
import requests
import zipfile
import io
import os
from tqdm import tqdm
from distance_calculator import calculate_distance

class FlightAnalysis(BaseModel):
    class Config:
        """arbitrary_types_allowed to True enables the model to also include 
        types like pandas DataFrames"""
        arbitrary_types_allowed = True

    data_url: str = Field(default="https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false") 
    date_created: datetime.date = Field(default_factory=datetime.date.today)

    """= None because the dataset do not have data untill explicitly
    loaded or assigned within class method"""
    
    airline_data: pd.DataFrame = None
    aircraft_data: pd.DataFrame = None
    airport_data: pd.DataFrame = None
    flight_routes: pd.DataFrame = None

    def __init__(self, **data):
        super().__init__(**data)
        
        #seting up directory to store teh flight data files
        download_dir = "./downloads"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            try:
                response = requests.get(self.data_url, stream=True)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(download_dir)
            except requests.exceptions.HTTPError as err:
                os.rmdir(download_dir)
                raise ValueError(f"Download failed with status code {response.status_code}: {err}")

        try:
            self.airline_data = pd.read_csv(f"{download_dir}/airlines.csv")
            self.aircraft_data = pd.read_csv(f"{download_dir}/airplanes.csv")
            self.airport_data = pd.read_csv(f"{download_dir}/airports.csv")
            self.flight_routes = pd.read_csv(f"{download_dir}/routes.csv")
            self.flight_routes.drop(columns=["Airline", "Source airport", "Destination airport"], inplace=True)
            self.airport_data["Airport ID"] = self.airport_data["Airport ID"].astype(str)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found during data setup: {exc}")

    def __str__(self):
        return f"FlightDataAnalysis created on {self.date_created}, using data from {self.data_url}"

    def update_route_distances(self, save=False, path=None):
            # This method now needs to calculate distances using the actual coordinates.
            # Assuming your `airport_data` DataFrame has 'Latitude' and 'Longitude' columns.
            if path:
                self.flight_routes = pd.read_csv(path)
            else:
                # Modify this part to calculate real distances
                def get_distance(row):
                    source_airport = self.airport_data.loc[self.airport_data['Airport ID'] == row['Source airport ID']]
                    dest_airport = self.airport_data.loc[self.airport_data['Airport ID'] == row['Destination airport ID']]
                    if not source_airport.empty and not dest_airport.empty:
                        return calculate_distance(
                            source_airport.iloc[0]['Latitude'], source_airport.iloc[0]['Longitude'],
                            dest_airport.iloc[0]['Latitude'], dest_airport.iloc[0]['Longitude']
                        )
                    else:
                        return None

                self.flight_routes['Distance'] = self.flight_routes.apply(get_distance, axis=1)
                if save:
                    self.flight_routes.to_csv("./downloads/updated_routes.csv", index=False)

#Method 3.1
    def plot_airports_by_country(self, country_name):
            country_airports = self.airport_data[self.airport_data["Country"] == country_name]

            if country_airports.empty:
                print(f"No airports found in {country_name}. Please check the country name and try again.")
                return
            
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            country_map = world[world.name == country_name]

            if country_map.empty:
                print(f"Country {country_name} not found in the map dataset. Please check the country name and try again.")
                return

            gdf = gpd.GeoDataFrame(country_airports, geometry=gpd.points_from_xy(country_airports.Longitude, country_airports.Latitude))

            fig, ax = plt.subplots(figsize=(10, 10))
            country_map.plot(ax=ax, color='lightgrey')
            gdf.plot(ax=ax, color='red', marker='o', markersize=50, label='Airports')

            plt.title(f"Airports in {country_name}")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()

            plt.show()

        def _str_(self):
            return f"FlightDataAnalysis created on {self.date_created}, using data from {self.data_url}"


#Method 3.3

    def flights_per_country(self, country: str, internal: bool = False) -> pd.DataFrame:
            """
            Retrieve flights for a specified country. Optionally, filter for only internal flights
            within the same country.

            Args:
            country (str): The name of the country.
            internal (bool, optional): If True, return only internal flights within the same country.
                Defaults to False.

            Returns:
            pd.DataFrame: DataFrame containing the filtered flights information.
            """

        # Check if the country exists in the airport_data DataFrame
            if country not in self.airport_data['Country'].values:
                print(f"Country '{country}' not found.")
                return None

            # Get all airport IDs that are in the specified country
            country_airports_ids = self.airport_data[self.airport_data["Country"] == country]["Airport ID"].tolist()

            # Filter flight_routes based on whether we're looking for internal flights only or all flights from/to the country
            if internal:
            # For internal flights, both source and destination airports must be in the country_airports_ids list
                filtered_flights = self.flight_routes[
                (self.flight_routes["Source airport ID"].isin(country_airports_ids)) &
                (self.flight_routes["Destination airport ID"].isin(country_airports_ids))
            ]
            else:
            # For all flights departing from the country, only source airports are considered
                filtered_flights = self.flight_routes[
                self.flight_routes["Source airport ID"].isin(country_airports_ids)
            ]

            return filtered_flights


#Method 3.4

    def plot_top_airplane_models(self, countries=None, N=5):
            """Plot the n most used airplane models by number of routes.

            Args:
                countries (str or list of str): A country or a list of countries to filter the routes.
                                                Defaults to None.
                N (int): Number of airplane models to plot.
                        Defaults to 5.
            """
            # filter routes by country
            if countries is not None:
                if isinstance(countries, str):
                    countries = [countries]
                # Merge routes with airports to get country information
                filtered_routes = self.flight_routes.merge(
                    self.airport_data, left_on="Source airport ID", right_on="IATA"
                )
                filtered_routes = filtered_routes[
                    filtered_routes["Country"].isin(countries)
                ]
            else:
                filtered_routes = self.flight_routes

            # count the number of routes for each airplane model
            airplane_model_counts = filtered_routes["Equipment"].value_counts().head(N)
            plt.figure(figsize=(10, 6))
            airplane_model_counts.plot(kind="bar")
            plt.title(f"Top {N} Airplane Models by Number of Routes")
            plt.xlabel("Airplane Model")
            plt.ylabel("Number of Routes")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

#Method 3.5
    def flights_per_country_(self, country: str, internal: bool = False) -> pd.DataFrame:
            # Implementation of flights_per_country method
            pass

        def plot_flights_by_country(self, country: str, internal: bool = False):
            """
            Plots the flights for a specified country. If internal is True, plots only internal flights within the same country.
            Otherwise, plots all flights leaving the country.

            Args:
            country (str): The name of the country.
            internal (bool, optional): If True, plots only internal flights. Defaults to False.
            """
            # Utilize the existing method to filter flights
            filtered_flights = self.flights_per_country(country, internal)

            if filtered_flights is None or filtered_flights.empty:
                print("No flights found for the given criteria.")
                return

            # Plotting
            plt.figure(figsize=(10, 6))

            if internal:
                title = f"Internal Flights within {country}"
                # Count the occurrences of each route for internal flights
                flight_counts = filtered_flights.groupby(['Source airport ID', 'Destination airport ID']).size().reset_index(name='Counts')
            else:
                title = f"Flights Leaving {country}"
                # Count the occurrences of each route for all flights leaving the country
                flight_counts = filtered_flights.groupby('Destination airport ID').size().reset_index(name='Counts')
            
            # Creating a bar plot
            sns.barplot(x='Counts', y='Destination airport ID', data=flight_counts.sort_values('Counts', ascending=False))
            plt.title(title)
            plt.xlabel('Number of Flights')
            plt.ylabel('Airport ID' if internal else 'Destination Airport ID')
            plt.show()

if __name__ == "__main__":
    analysis = FlightAnalysis()  # Create an instance of the class
    analysis.update_route_distances(save=True)  # Call the method to update route distances

    # Print the first few rows of the flight_routes DataFrame to check distances
    print(analysis.flight_routes.head())