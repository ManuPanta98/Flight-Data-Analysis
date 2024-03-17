import pandas as pd
from pydantic import BaseModel, Field
import datetime
import requests
import matplotlib.pyplot as plt
import zipfile
import io
import os
from tqdm import tqdm
from distance_calculator import calculate_distance
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import geopandas as gpd
from langchain_openai import ChatOpenAI
from IPython.display import Markdown
import cartopy.io.shapereader as shpreader


class FlightAnalysis(BaseModel):
    """
    A class for analyzing flight data which includes methods for updating route distances, 
    plotting airports by country, analyzing flight distance distribution, plotting flight routes 
    from an airport, plotting the top airplane models, and plotting flights on a world map.
    
    Attributes:
        data_url (str): URL to download the flight data from.
        date_created (datetime.date): Date when the instance was created.
        airline_data (pd.DataFrame): DataFrame containing data on airlines.
        aircraft_data (pd.DataFrame): DataFrame containing data on aircraft.
        airport_data (pd.DataFrame): DataFrame containing data on airports.
        flight_routes (pd.DataFrame): DataFrame containing data on flight routes.
    
    Methods:
        update_route_distances: Calculates and optionally saves the distances between airports.
        plot_airports_by_country: Plots the airports located in the given country.
        distance_analysis: Plots a histogram of the distribution of flight distances.
        plot_flights_from_airport: Plots all flights departing from the given airport.
        plot_top_airplane_models: Plots the most used airplane models.
        plot_flights_on_map: Plots flight routes from or within a specified country on a world map.
    """

    class Config:
        arbitrary_types_allowed = True

    data_url: str = Field(
        default="https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false"
    )
    date_created: datetime.date = Field(default_factory=datetime.date.today)
    
    airline_data: pd.DataFrame = None
    aircraft_data: pd.DataFrame = None
    airport_data: pd.DataFrame = None
    flight_routes: pd.DataFrame = None

    def __init__(self, **data):
        """
        Initializes the FlightAnalysis class by downloading and extracting the flight data.
        
        Args:
            **data: Variable length keyword arguments.
        """
        super().__init__(**data)
        
        # Setting up directory to store the flight data files
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
        """String representation of the FlightDataAnalysis instance."""
        return f"FlightDataAnalysis created on {self.date_created}, using data from {self.data_url}"
    
    # Additional methods go here, with docstrings similar to the one above for __init__.

    def update_route_distances(self, save=False, path=None, load_from=None):
        """
        Calculate distances between airports using latitude and longitude coordinates and 
        update the flight_routes DataFrame with these calculated distances. 
        Optionally, save the updated DataFrame to a CSV file.
        
        Args:
            save (bool): Whether to save the DataFrame to a CSV file. Defaults to False.
            path (str): The path to save the CSV file if save is True. Defaults to None.
            load_from (str): The path to load the route distances CSV file. Defaults to None.
        """
        if load_from:
            # If a path is provided for loading data, load the data from that path.
            self.flight_routes = pd.read_csv(load_from)
        else:
            # If no path is provided, calculate the distances and update the DataFrame.
            def get_distance(row):
                # Nested function to calculate distance based on a row of the DataFrame.
                source_airport = self.airport_data.loc[self.airport_data['Airport ID'] == row['Source airport ID']]
                dest_airport = self.airport_data.loc[self.airport_data['Airport ID'] == row['Destination airport ID']]
                if not source_airport.empty and not dest_airport.empty:
                    return calculate_distance(
                        source_airport.iloc[0]['Latitude'], source_airport.iloc[0]['Longitude'],
                        dest_airport.iloc[0]['Latitude'], dest_airport.iloc[0]['Longitude']
                    )
                else:
                    return None

            # Apply the get_distance function to each row in the flight_routes DataFrame.
            self.flight_routes['Distance'] = self.flight_routes.apply(get_distance, axis=1)

        if save:
            # If save is True, write the DataFrame to the specified path or to a default path.
            save_path = path if path else "./downloads/updated_routes.csv"
            self.flight_routes.to_csv(save_path, index=False)


    #Method 1
    def plot_airports_by_country(self, country_name):
        
            country_airports = self.airport_data[self.airport_data["Country"] == country_name]
            """
            Plot the airports located in a specific country using geopandas.
        
            Args:
                country_name (str): The name of the country for which to plot the airports.
            """
            # Method implementation ...
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
                   
    def __str__(self):
        return f"FlightDataAnalysis created on {self.date_created}, using data from {self.data_url}"
    
    
    # Method 2
    def distance_analysis(self, display_plot=True, save_plot=False, filename='flight_distance_distribution.png'):
        """
        Plot a histogram of the distribution of flight distances for all flights.
        
        Args:
            display_plot (bool): Whether to display the plot. Defaults to True.
            save_plot (bool): Whether to save the plot to a file. Defaults to False.
            filename (str): The filename to save the plot if save_plot is True. 
                            Defaults to 'flight_distance_distribution.png'.
        """
        # Method implementation ...
        
        if hasattr(self, 'flight_routes') and 'Distance' in self.flight_routes.columns:
            # Plotting the distribution
            plt.figure(figsize=(10, 6))
            plt.hist(self.flight_routes['Distance'].dropna(), bins=50, color='blue', edgecolor="black")
            plt.title('Distribution of Flight Distances')
            plt.xlabel('Distance (km)')
            plt.ylabel('Number of Flights')
            
            # Display the plot if requested
            if display_plot:
                plt.show()
            
            # Save the plot to a file if requested
            if save_plot:
                plt.savefig(filename)
            else:
                print("Flight distance data is not available.")
    
    
    # Method 3

    def plot_flights_from_airport(self, airport: str, internal: bool = False):
        """
        Plot all flights departing from a specified airport on a map using Cartopy.
        If internal is True, only plot flights with destinations within the same country.
        
        Args:
            airport (str): The IATA code of the airport.
            internal (bool): Whether to plot only internal flights. Defaults to False.
        """
        # Method implementation ...
        
        airport_info = self.airport_data[self.airport_data['IATA'] == airport]
        if airport_info.empty:
            print(f"Airport '{airport}' not found.")
            return None
        
        # Extract the airport's location
        airport_coords = airport_info[['Longitude', 'Latitude']].iloc[0]
        airport_id = airport_info['Airport ID'].iloc[0]
        airport_country = airport_info['Country'].iloc[0]
        
        # Add Natural Earth data for higher resolution and country names
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.5)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ax.add_feature(cfeature.LAKES.with_scale('50m'))
        ax.add_feature(cfeature.RIVERS.with_scale('50m'))
        ax.set_global()

        ne_10m_admin_0_countries = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        country_boundaries = shpreader.Reader(ne_10m_admin_0_countries)
        country = None
        for country_feature in country_boundaries.records():
            if country_feature.attributes['NAME_LONG'] == airport_country:
                country = country_feature
                break

        # Plot the airport location
        ax.plot(airport_coords['Longitude'], airport_coords['Latitude'], 
                color='red', linewidth=2, marker='o', transform=ccrs.Geodetic(),
                markersize=10, label=f"{airport} (Departure)")

        # Filter for flights
        flights = self.flight_routes[self.flight_routes['Source airport ID'] == airport_id]
        if internal:
            country_airports = self.airport_data[self.airport_data['Country'] == airport_country]
            country_airports_ids = country_airports['Airport ID'].tolist()
            flights = flights[flights['Destination airport ID'].isin(country_airports_ids)]

        # Plot the flights
        for _, flight in flights.iterrows():
            dest_airport_info = self.airport_data[self.airport_data['Airport ID'] == flight['Destination airport ID']]
            if dest_airport_info.empty:
                continue

            dest_coords = dest_airport_info[['Longitude', 'Latitude']].iloc[0]
            ax.plot([airport_coords['Longitude'], dest_coords['Longitude']], 
                    [airport_coords['Latitude'], dest_coords['Latitude']], 
                    color='blue', linewidth=1, marker='', transform=ccrs.Geodetic())
        if internal and country is not None:
            # Zoom into the country's extent if internal flights are plotted
            bounds = country.geometry.bounds
            padding = 1.0  # degrees of padding, you might need to adjust this
            extent = [bounds[0] - padding, bounds[2] + padding, bounds[1] - padding, bounds[3] + padding]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            # Otherwise, use a global extent
            ax.set_global()

        # Customize the plot
        ax.set_title(f"Flights from {airport}", pad=20)
        plt.legend(loc='upper left')
        plt.show()

    # Method 4

    def plot_top_airplane_models(self, countries=None, N=5):
        """
        Plot the n most used airplane models by number of routes.

        Args:
            countries (str or list of str): A country or a list of countries to filter the routes.
                                            Defaults to None.
            N (int): Number of airplane models to plot. Defaults to 5.
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
        
    

    # Method 5

    def plot_flights_on_map(self, country: str, internal: bool = False, cutoff_distance: float = 1000):
        """
        Plots flight routes on a world map, differentiating between short-haul and long-haul flights.

        This method uses Cartopy to plot flight routes originating from or within a specified country, marking 
        the routes with different colors based on whether they are shorter or longer than a specified cutoff distance. 
        The method aims to provide visual insights into flight patterns, with an additional focus on environmental 
        considerations by calculating and displaying potential CO2 emissions reductions if short-haul flights were 
        replaced with rail transport.

        Parameters:
        country (str): The name of the country from which to plot flights.
            internal (bool, optional): If True, only plots internal flights within the specified country. 
                                    Defaults to False, which includes all flights departing from the country.
            cutoff_distance (float, optional): The distance in kilometers used to distinguish between short-haul 
                                            and long-haul flights. Defaults to 1000 km.

            Note:
            The emissions calculations are based on generalized CO2 emissions factors for flight and rail transport 
            and are intended to provide a simple comparative analysis. Actual emissions can vary based on numerous 
            factors, including aircraft type, occupancy, and specific rail transport efficiencies.
        """   
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        country_airports = self.airport_data[self.airport_data["Country"] == country]
        
        # Initialize the distances outside the conditional statement
        short_haul_distance = 0
        long_haul_distance = 0

        if internal:
            filtered_flights = self.flight_routes[
                (self.flight_routes["Source airport ID"].isin(country_airports["Airport ID"])) &
                (self.flight_routes["Destination airport ID"].isin(country_airports["Airport ID"]))
            ]
        else:
            filtered_flights = self.flight_routes[
                self.flight_routes["Source airport ID"].isin(country_airports["Airport ID"])
            ]
            
        fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE, linewidth=0.1)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.1)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, edgecolor='black')

        for _, flight in filtered_flights.iterrows():
            source = country_airports[country_airports["Airport ID"] == flight["Source airport ID"]].iloc[0]
            dest = self.airport_data[self.airport_data["Airport ID"] == flight["Destination airport ID"]].iloc[0]

            # Assume calculate_distance is a defined function that calculates distance between two points
            distance = calculate_distance(source["Latitude"], source["Longitude"], dest["Latitude"], dest["Longitude"])

            if distance <= cutoff_distance:
                short_haul_distance += distance
                color = 'blue'
            else:
                long_haul_distance += distance
                color = 'darkorange'

            plt.plot([source["Longitude"], dest["Longitude"]], [source["Latitude"], dest["Latitude"]],
                    color=color, linewidth=1, marker='o', transform=ccrs.Geodetic())

            ax.set_global()
            ax.set_title(f"Flights from {country}: Blue = Short-haul (<= {cutoff_distance}km), Dark Orange = Long-haul")

        # Calculate and display emissions information
        flight_emissions_per_km = 133  # grams of CO2 equivalents per passenger kilometer for flights
        rail_emissions_per_km = 7     # grams of CO2 per passenger kilometer for rails
        total_flight_emissions = (short_haul_distance * flight_emissions_per_km + long_haul_distance * flight_emissions_per_km) / 1e6  # Convert to metric tonnes
        total_rail_emissions = (short_haul_distance * rail_emissions_per_km) / 1e6  # Convert to metric tonnes
        emissions_reduction = total_flight_emissions - (total_rail_emissions + long_haul_distance * flight_emissions_per_km / 1e6)

        plt.text(0.05, 1.0, f"Total CO2 emissions (flights): {total_flight_emissions:.2f} tonnes",
        transform=ax.transAxes, backgroundcolor='white', verticalalignment='top')
        plt.text(0.05, 0.95, f"Potential CO2 emissions (replacing short-haul with rails): {(total_rail_emissions + long_haul_distance * flight_emissions_per_km / 1e6):.2f} tonnes",
        transform=ax.transAxes, backgroundcolor='white', verticalalignment='top')
        plt.text(0.05, 0.90, f"Potential reduction in CO2 emissions: {emissions_reduction:.2f} tonnes",
        transform=ax.transAxes, backgroundcolor='white', verticalalignment='top')

        plt.show()
        
    # Method 6

    def aircrafts(self):
        """
        Prints a list of unique aircraft models contained in the dataset.

        This method retrieves the unique names of aircraft models from the 'Name' column in the 'aircraft_data'
        DataFrame attribute and prints them to the console.

        Returns:
            None: This method does not return any value; it outputs directly to the console.
        """
        aircraft_models = self.aircraft_data['Name'].unique()
        print("List of Aircraft Models:")
        for model in aircraft_models:
            print(model) 
            
    # Method 7
    
    def aircraft_info(self, aircraft_name: str):
        """
        Retrieves and prints information about a specified aircraft model.

        This method checks if the given aircraft name exists within the 'Name' column of the 'aircraft_data'
        attribute. If the aircraft name is not found, it raises a ValueError with a message indicating the issue
        and providing a list of valid aircraft model names from the dataset. If the aircraft name exists, it
        constructs a prompt and sends it to the OpenAI Chat model to get information about the aircraft, which
        is then printed to the console.

        Parameters:
            aircraft_name (str): The name of the aircraft model for which information is requested.

        Raises:
            ValueError: If the specified aircraft_name is not present in the list of available aircraft models.

        Returns:
            None: This method prints the response content to the console and does not return any value.
        """
        if aircraft_name not in self.aircraft_data['Name'].values:
            available_models = self.aircraft_data['Name'].unique()
            guidance = (
                f"The specified aircraft name '{aircraft_name}' is not in the list of available aircraft models. "
                f"Please choose from the following list of models: {', '.join(available_models)}"
            )
            raise ValueError(guidance)
        
        llm = ChatOpenAI(temperature=0.1)
        prompt = f"Give me information about the aircraft model named {aircraft_name}. In the form of specification table with 2 columns: specification and value"
        response = llm.invoke(prompt)
        print(response.content)
        
    # Method 8

    def aircraft_info_markdown(self, aircraft_name: str):
        """
        Generates markdown content with information about a specified aircraft model by invoking the OpenAI Chat model.

        This method checks if the specified aircraft name exists in the dataset. If the aircraft name is not found,
        it raises a ValueError with guidance for choosing a valid aircraft model from the available data. If the
        aircraft name is found, it sends a prompt to the OpenAI Chat model to retrieve information about that aircraft.

        Parameters:
        aircraft_name (str): The name of the aircraft model to retrieve information for.

        Raises:
        ValueError: If the specified aircraft_name is not found in the list of available aircraft models.

        Returns:
        None: This method does not return. It displays the markdown content in the current Jupyter notebook cell.
        """
        if aircraft_name not in self.aircraft_data['Name'].values:
            available_models = self.aircraft_data['Name'].unique()
            guidance = (
                f"The specified aircraft name '{aircraft_name}' is not in the list of available aircraft models. "
                f"Please choose from the following list of models: {', '.join(available_models)}"
            )
            raise ValueError(guidance)
        
        llm = ChatOpenAI(temperature=0.1)
        prompt = f"Give me information about the aircraft model named {aircraft_name} in the form of specification table with 2 columns: specification and value."
        response = llm.invoke(prompt)
        display(Markdown(response.content))

    # Method 9
    
    def airport_info(self, airport_name: str):
        """
        Retrieves and prints information about the specified airport.

        This method sends a prompt to the OpenAI Chat model requesting information about the airport
        with the given name. It then prints the content of the response to the console. The Chat model's
        temperature is set to 0.1 to prioritize deterministic, informative responses.

        Parameters:
        airport_name (str): The name of the airport to retrieve information about.

        Returns:
        None: This method prints the response content to the console and does not return any value.
        """
        llm = ChatOpenAI(temperature=0.1)
        prompt = f"Give me information about the airport named {airport_name} in the form of specification table with 2 columns: specification and value."
        response = llm.invoke(prompt)
        print(response.content)