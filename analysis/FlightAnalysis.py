import pandas as pd
from pydantic import BaseModel, Field
import datetime
import requests
import zipfile
import io
import os
from tqdm import tqdm

class FlightDataAnalysis(BaseModel):
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