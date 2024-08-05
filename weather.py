import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
import pandas as pd


def get_weather():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 34.0459, # enter your latitude
        "longitude": 118.2426, # enter your longitude
        "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = round(current.Variables(0).Value(), 1)
    current_relative_humidity_2m = round(current.Variables(1).Value(), 1)
    current_wind_speed_10m = round(current.Variables(2).Value(), 1)

    weather_info = (
        f"The current temperature is {current_temperature_2m} degrees Fahrenheit, "
        f"humidity is {current_relative_humidity_2m} percent, "
        f"and wind speed is {current_wind_speed_10m} miles per hour."
    )
    return weather_info

def get_7_day_forecast():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 33.7456,# change values to your coordinates 
        "longitude": -117.8678,
        "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles"
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        
        # Process first location
        response = responses[0]

        # Process daily data
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_weathercode = daily.Variables(2).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit="s", utc=True),
            end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq = pd.Timedelta(seconds=daily.Interval()),
            inclusive = "left"
        )}
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["weathercode"] = daily_weathercode

        daily_dataframe = pd.DataFrame(data=daily_data)

        # Weather code descriptions
        weather_descriptions = {
            0: "Clear skies", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }

        forecast_info = "Here's the seven-day forecast for Los Angeles :\n" #change to your city name 
        for index, row in daily_dataframe.iterrows():
            day_name = row['date'].strftime('%A')
            weather_desc = weather_descriptions.get(int(row['weathercode']), "Unknown weather")
            forecast_info += f"- {day_name}: {weather_desc} with a high of {round(row['temperature_2m_max'])} degrees and a low of {round(row['temperature_2m_min'])} degrees.\n"

        return forecast_info.strip()

    except Exception as e:
        print(f"Error in get_7_day_forecast: {str(e)}")
        return "I'm sorry, I'm having trouble fetching the 7-day forecast right now. Please try again later."

# Test the function
if __name__ == "__main__":
    print(get_7_day_forecast())
