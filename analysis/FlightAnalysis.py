# Re-importing necessary libraries after the reset
from typing import Optional
import os
import requests
from pydantic import BaseModel
import pandas as pd
import zipfile
from io import BytesIO

# Corrected and updated class definition
class FlightAnalysisCorrected:
    """
    A class to analyze flight data, compliant with PEP8 standards and uses static type
    checking where applicable. It downloads a data file during initialization if the file does not
    already exist in the specified downloads directory and reads the dataset into a pandas DataFrame.
    """
    
    class Config(BaseModel):
        """
        Configuration for FlightAnalysis, used for type checking with Pydantic.
        Include any configuration parameters here.
        """
        # Correct usage of default value for a BaseModel field
        data_url: str = "https://gitlab.com/adpro1/adpro2024/-/raw/main/Files/flight_data.zip?inline=false"
    
    def __init__(self, config: Config = Config()):
        """
        Initializes the FlightAnalysis class, downloading the data file if not already present,
        and loads it into a pandas DataFrame.
        
        Parameters:
            config (Config): Configuration object containing necessary data like the data file URL.
        """
        self.config = config
        self.data_file_path = os.path.join('downloads', 'flight_data.csv')
        self.download_and_extract_data_file()
        self.data_frame = self.load_data_into_dataframe()
    
    def download_and_extract_data_file(self) -> None:
        """
        Downloads and extracts the data file to the 'downloads/' directory if it doesn't already exist.
        """
        if not os.path.exists('downloads'):
            os.makedirs('downloads')
        # Assuming the file is a zip file containing 'flight_data.csv'
        zip_file_path = os.path.join('downloads', 'flight_data.zip')
        if not os.path.exists(self.data_file_path):
            response = requests.get(self.config.data_url)
            with open(zip_file_path, 'wb') as file:
                file.write(response.content)
            # Extract the zip file
            with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall('downloads')
            print("Data file downloaded and extracted successfully.")
        else:
            print("Data file already exists.")
    
    def load_data_into_dataframe(self) -> pd.DataFrame:
        """
        Loads the data file into a pandas DataFrame and removes superfluous columns.
        
        Returns:
            pd.DataFrame: The loaded and cleaned DataFrame.
        """
        df = pd.read_csv(self.data_file_path)
        # Example of removing superfluous columns, assuming 'unnecessary_column' to be removed
        # Adjust according to actual dataset columns
        cleaned_df = df.drop(columns=['unnecessary_column'], errors='ignore')
        return cleaned_df

