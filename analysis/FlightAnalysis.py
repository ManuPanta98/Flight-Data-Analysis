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



if __name__ == "__main__":
    analysis = FlightAnalysis()  # Create an instance of the class
    analysis.update_route_distances(save=True)  # Call the method to update route distances

    # Print the first few rows of the flight_routes DataFrame to check distances
    print(analysis.flight_routes.head())
