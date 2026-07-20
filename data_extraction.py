# data_extraction.py
"""
Solar Plant Copilot — Data Extraction
Pulls historical weather/solar data from Open-Meteo API for all zones.
"""

import requests
import pandas as pd
import time
import os
from config import (
    ZONES, START_DATE, END_DATE, HOURLY_VARS,
    TIMEZONE, BASE_URL, TILT, AZIMUTH, DATA_DIR
)


def fetch_zone_data(zone_name, lat, lon, start_date, end_date):
    """
    Fetch historical hourly weather/solar data for a single zone.
    Returns a pandas DataFrame or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
        "tilt": TILT,
        "azimuth": AZIMUTH,
    }

    print(f"Fetching data for {zone_name} ({lat}, {lon})...")
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        if response.status_code != 200:
            print(f"  ERROR {response.status_code}: {response.text[:300]}")
            return None

        data = response.json()
        hourly = data.get("hourly", {})

        df = pd.DataFrame(hourly)
        df["zone"] = zone_name
        df["latitude"] = lat
        df["longitude"] = lon

        print(f"  -> {len(df)} rows fetched")
        return df

    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


def extract_all_zones(
    start_date=START_DATE,
    end_date=END_DATE,
    save=True
):
    """
    Fetch data for all 4 zones and save individual + combined CSVs.
    Returns the combined DataFrame.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    all_zone_dfs = []

    for zone_name, coords in ZONES.items():
        df = fetch_zone_data(
            zone_name,
            coords["lat"],
            coords["lon"],
            start_date,
            end_date
        )

        if df is not None:
            all_zone_dfs.append(df)

            if save:
                zone_path = os.path.join(DATA_DIR, f"{zone_name}_weather.csv")
                df.to_csv(zone_path, index=False)
                print(f"  Saved -> {zone_path}")

        time.sleep(1)  # polite delay between API calls

    if not all_zone_dfs:
        print("No data fetched. Check API connectivity.")
        return None

    combined = pd.concat(all_zone_dfs, ignore_index=True)

    if save:
        combined_path = os.path.join(DATA_DIR, "all_zones_weather_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined file saved: {combined_path}")
        print(f"Total rows: {len(combined)}")
        print(f"File size: {os.path.getsize(combined_path) / (1024*1024):.2f} MB")

    return combined


if __name__ == "__main__":
    combined = extract_all_zones()
    if combined is not None:
        print("\nExtraction complete.")
        print(combined.head())