import httpx
import logging
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

from execute import (
    execute_weather_search_input, execute_weather_search_output,
    execute_weather_current_input, execute_weather_current_output,
    execute_weather_forecast_input, execute_weather_forecast_output
)

logger = logging.getLogger(__name__)

class WeatherError(Exception):
    """Raised when weather API operations fail"""
    pass

class OpenWeatherClient:
    """Client for OpenWeather API interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org"
        
        if not self.api_key:
            logger.warning("OpenWeather API key not configured")
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenWeather API"""
        
        if not self.api_key:
            raise WeatherError("OpenWeather API key not configured")
        
        # Add API key to params
        params["appid"] = self.api_key
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}{endpoint}", params=params)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenWeather API error: {e.response.status_code} - {e.response.text}")
            raise WeatherError(f"OpenWeather API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error making OpenWeather request: {e}")
            raise WeatherError(f"Request failed: {str(e)}")

# Weather tool implementations
async def weather_search(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for geographic coordinates of cities using OpenWeather Geocoding API
    
    Essential first step before calling current weather or forecast APIs when you 
    only have city names or addresses. Converts location names into precise 
    coordinates for accurate weather data.
    """
    try:
        # Initialize client to ensure .env.local is loaded
        _weather_client = OpenWeatherClient()
        
        # Extract search parameters
        query = input_data.get("q", "")
        limit = input_data.get("limit", 5)
        
        if not query:
            return {
                "success": False,
                "message": "Query parameter 'q' is required",
                "data": []
            }
        
        # Call OpenWeather Geocoding API
        params = {
            "q": query,
            "limit": limit
        }
        
        raw_data = await _weather_client._make_request("/geo/1.0/direct", params)
        
        # Transform to our schema format
        locations = []
        for item in raw_data:
            location = {
                "name": item.get("name", ""),
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "country": item.get("country", ""),
                "state": item.get("state", "")
            }
            
            # Add local names if available
            if "local_names" in item:
                location["local_names"] = item["local_names"]
            
            locations.append(location)
        
        return {
            "success": True,
            "message": f"Found {len(locations)} location(s) matching '{query}'",
            "data": locations
        }
        
    except WeatherError as e:
        logger.error(f"Weather search error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": []
        }
    except Exception as e:
        logger.error(f"Unexpected error in weather_search: {e}")
        return {
            "success": False,
            "message": f"Search failed: {str(e)}",
            "data": []
        }

async def weather_current(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current weather data for any location using OpenWeather Current Weather API
    
    Access current weather data for any location on Earth including temperature, 
    humidity, pressure, wind, clouds, and precipitation. Use coordinates (lat/lon) 
    for most accurate results, or city names for convenience.
    """
    try:
        # Initialize client to ensure .env.local is loaded
        _weather_client = OpenWeatherClient()
        
        # Build API parameters
        params = {}
        
        # Location parameters (one of these must be provided)
        if input_data.get("lat") is not None and input_data.get("lon") is not None:
            params["lat"] = input_data["lat"]
            params["lon"] = input_data["lon"]
        elif input_data.get("q"):
            params["q"] = input_data["q"]
        elif input_data.get("id"):
            params["id"] = input_data["id"]
        elif input_data.get("zip"):
            params["zip"] = input_data["zip"]
        else:
            return {
                "success": False,
                "message": "Location required: provide lat/lon, q (city name), id (city ID), or zip code",
                "data": {}
            }
        
        # Optional parameters
        if input_data.get("units"):
            params["units"] = input_data["units"]
        if input_data.get("lang"):
            params["lang"] = input_data["lang"]
        if input_data.get("mode"):
            params["mode"] = input_data["mode"]
        
        # Call OpenWeather Current Weather API
        raw_data = await _weather_client._make_request("/data/2.5/weather", params)
        
        return {
            "success": True,
            "message": "Current weather data retrieved successfully",
            "data": raw_data  # OpenWeather response format matches our schema exactly
        }
        
    except WeatherError as e:
        logger.error(f"Weather current error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error in weather_current: {e}")
        return {
            "success": False,
            "message": f"Current weather request failed: {str(e)}",
            "data": {}
        }

async def weather_forecast(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get current weather and forecasts using OpenWeather One Call API 3.0
    
    Provides minute forecast for 1 hour, hourly forecast for 48 hours, daily 
    forecast for 8 days, and government weather alerts. Use coordinates (lat/lon) 
    for precise location targeting.
    """
    try:
        # Initialize client to ensure .env.local is loaded
        _weather_client = OpenWeatherClient()
        
        # Extract required coordinates
        lat = input_data.get("lat")
        lon = input_data.get("lon")
        
        if lat is None or lon is None:
            return {
                "success": False,
                "message": "Latitude and longitude coordinates are required for forecast API",
                "data": {}
            }
        
        # Build API parameters
        params = {
            "lat": lat,
            "lon": lon
        }
        
        # Optional parameters
        if input_data.get("exclude"):
            params["exclude"] = input_data["exclude"]
        if input_data.get("units"):
            params["units"] = input_data["units"]
        if input_data.get("lang"):
            params["lang"] = input_data["lang"]
        
        # Call OpenWeather One Call API 3.0
        raw_data = await _weather_client._make_request("/data/3.0/onecall", params)
        
        return {
            "success": True,
            "message": "Weather forecast data retrieved successfully",
            "data": raw_data  # OpenWeather response format matches our schema exactly
        }
        
    except WeatherError as e:
        logger.error(f"Weather forecast error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error in weather_forecast: {e}")
        return {
            "success": False,
            "message": f"Weather forecast request failed: {str(e)}",
            "data": {}
        } 