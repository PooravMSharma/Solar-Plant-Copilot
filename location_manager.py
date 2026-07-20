# location_manager.py
"""
Solar Plant Copilot — Location Manager
Handles geocoding, optimal tilt calculation, and
first-run data pipeline initialization for any location.
"""

import os
import json
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pvlib.location import Location
from config import (
    HOURLY_VARS, TIMEZONE, DATA_DIR,
    ZONE_CAPACITY_KW, SYSTEM_LOSSES,
    INVERTER_EFFICIENCY, GAMMA_PDC, U0, U1
)


# ── Geocoding ──────────────────────────────────────────────────
def geocode_location(city_name):
    """
    Convert a city/location name to coordinates.
    Returns (latitude, longitude, full_address) or None on failure.
    """
    try:
        geolocator = Nominatim(user_agent="solar_copilot")
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            return (
                round(location.latitude, 4),
                round(location.longitude, 4),
                location.address
            )
        return None
    except GeocoderTimedOut:
        return None


# ── Optimal Tilt ───────────────────────────────────────────────
def calculate_optimal_tilt(latitude):
    """
    Calculate optimal fixed-tilt panel angle for a given latitude.
    Uses standard solar engineering rule of thumb.
    Returns tilt angle in degrees.
    """
    tilt = round(abs(latitude) * 0.87, 1)
    # Clamp to reasonable range
    tilt = max(5.0, min(tilt, 45.0))
    return tilt


# ── Zone Coordinates ───────────────────────────────────────────
def generate_zone_coords(base_lat, base_lon, n_zones=4):
    """
    Generate coordinates for n zones around a base location.
    Offsets are ~500m apart to simulate a real plant spread.
    Returns dict of zone_name -> {lat, lon}.
    """
    offsets = [
        (+0.005, +0.005),
        (+0.005, -0.005),
        (-0.005, +0.005),
        (-0.005, -0.005),
    ]
    zones = {}
    for i in range(min(n_zones, len(offsets))):
        zone_name = f"Zone_{chr(65+i)}"  # Zone_A, Zone_B, etc.
        zones[zone_name] = {
            "lat": round(base_lat + offsets[i][0], 6),
            "lon": round(base_lon + offsets[i][1], 6)
        }
    return zones


# ── Location Profile ───────────────────────────────────────────
def build_location_profile(city_name, n_zones=4, capacity_kw=5000):
    """
    Build a complete location profile from a city name.
    Returns dict with all location-specific parameters.
    """
    coords = geocode_location(city_name)
    if coords is None:
        return None

    lat, lon, address = coords
    tilt = calculate_optimal_tilt(lat)
    zones = generate_zone_coords(lat, lon, n_zones)

    # Determine hemisphere for azimuth
    # Northern hemisphere -> face south (azimuth=0 in Open-Meteo convention)
    # Southern hemisphere -> face north (azimuth=180)
    azimuth = 0 if lat >= 0 else 180

    profile = {
        "city_name": city_name,
        "address": address,
        "latitude": lat,
        "longitude": lon,
        "tilt": tilt,
        "azimuth": azimuth,
        "n_zones": n_zones,
        "capacity_kw": capacity_kw,
        "zones": zones,
        "data_dir": os.path.join(DATA_DIR, _location_slug(city_name))
    }

    return profile


def _location_slug(city_name):
    """Convert city name to a safe directory name."""
    return city_name.lower().replace(" ", "_").replace(",", "")


# ── Location Cache ─────────────────────────────────────────────
def save_location_profile(profile):
    """Save location profile to disk for caching."""
    os.makedirs(profile["data_dir"], exist_ok=True)
    profile_path = os.path.join(profile["data_dir"], "location_profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Location profile saved -> {profile_path}")


def load_location_profile(city_name):
    """
    Load a previously saved location profile from disk.
    Returns profile dict or None if not found.
    """
    slug = _location_slug(city_name)
    profile_path = os.path.join(DATA_DIR, slug, "location_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            return json.load(f)
    return None


def is_location_initialized(city_name):
    """
    Check if a location has already been fully initialized
    (data fetched, processed, RAG built).
    Returns True/False.
    """
    slug = _location_slug(city_name)
    required_files = [
        os.path.join(DATA_DIR, slug, "location_profile.json"),
        os.path.join(DATA_DIR, slug, "all_zones_with_output.csv"),
        os.path.join(DATA_DIR, slug, "all_zones_window_summaries.csv"),
        os.path.join(DATA_DIR, slug, "faiss_index"),
    ]
    return all(os.path.exists(f) for f in required_files)


# ── Full Initialization Pipeline ───────────────────────────────
def initialize_location(
    city_name,
    start_date="2020-01-01",
    end_date="2024-12-31",
    n_zones=4,
    capacity_kw=5000,
    progress_callback=None
):
    """
    Full first-run initialization for a new location.
    Steps:
    1. Geocode city name
    2. Build location profile
    3. Fetch weather data
    4. Preprocess (zenith, daylight, plant output)
    5. Generate window summaries
    6. Build RAG index
    7. Save everything to location-specific directory

    progress_callback: optional function(step, total, message)
    for Streamlit progress bar updates.

    Returns (profile, summaries_df, daylight_df, hybrid_retriever,
             seasonal_lookup, embeddings) or None on failure.
    """

    def _progress(step, total, message):
        if progress_callback:
            progress_callback(step, total, message)
        print(f"[{step}/{total}] {message}")

    total_steps = 6

    # Step 1 — Build location profile
    _progress(1, total_steps, f"Geocoding {city_name}...")
    profile = build_location_profile(city_name, n_zones, capacity_kw)
    if profile is None:
        print(f"Could not geocode: {city_name}")
        return None

    os.makedirs(profile["data_dir"], exist_ok=True)
    save_location_profile(profile)
    print(f"Location: {profile['address']}")
    print(f"Coordinates: {profile['latitude']}, {profile['longitude']}")
    print(f"Optimal tilt: {profile['tilt']}°")

    # Step 2 — Fetch weather data
    _progress(2, total_steps, "Fetching weather data from Open-Meteo...")
    from data_extraction import fetch_zone_data
    import time

    all_zone_dfs = []
    for zone_name, coords in profile["zones"].items():
        df = fetch_zone_data(
            zone_name,
            coords["lat"],
            coords["lon"],
            start_date,
            end_date,
            tilt=profile["tilt"],
            azimuth=profile["azimuth"]
        )
        if df is not None:
            all_zone_dfs.append(df)
        time.sleep(1)

    if not all_zone_dfs:
        print("Failed to fetch weather data.")
        return None

    combined_df = pd.concat(all_zone_dfs, ignore_index=True)
    combined_df.to_csv(
        os.path.join(profile["data_dir"], "all_zones_weather_combined.csv"),
        index=False
    )

    # Step 3 — Preprocess
    _progress(3, total_steps, "Preprocessing data...")
    from preprocessing import add_zenith_angles, filter_daylight, simulate_plant_output
    from config import PDC0

    zenith_df = add_zenith_angles(combined_df, profile["zones"])
    daylight_df = filter_daylight(zenith_df)

    # Override PDC0 for custom capacity
    pdc0_custom = capacity_kw * 1000
    output_df = simulate_plant_output(daylight_df, pdc0=pdc0_custom)
    output_df.to_csv(
        os.path.join(profile["data_dir"], "all_zones_with_output.csv"),
        index=False
    )

    # Step 4 — Window summaries
    _progress(4, total_steps, "Generating sliding window summaries...")
    from time_series import compute_window_summaries
    from config import SEASON_MAP
    import numpy as np
    from pvlib import temperature, pvsystem

    all_summaries = []
    for zone_name in profile["zones"].keys():
        zone_full = zenith_df[zenith_df["zone"] == zone_name].copy()
        zone_full["time"] = pd.to_datetime(zone_full["time"])
        zone_full["hour"] = zone_full["time"].dt.hour

        zone_daylight = output_df[
            output_df["zone"] == zone_name
        ][["time", "ac_power_kw"]].copy()
        zone_daylight["time"] = pd.to_datetime(zone_daylight["time"])

        zone_full = zone_full.merge(zone_daylight, on="time", how="left")
        zone_full["ac_power_kw"] = zone_full["ac_power_kw"].fillna(0)

        hourly_avg = zone_full.groupby("hour")["ac_power_kw"].mean()
        summaries = compute_window_summaries(zone_full, hourly_avg, zone_name)
        all_summaries.append(summaries)

    summaries_df = pd.concat(all_summaries, ignore_index=True)
    summaries_df["month"] = pd.to_datetime(
        summaries_df["window_start"]
    ).dt.month
    summaries_df["season"] = summaries_df["month"].map(SEASON_MAP)
    summaries_df.to_csv(
        os.path.join(profile["data_dir"], "all_zones_window_summaries.csv"),
        index=False
    )

    # Step 5 — RAG index
    _progress(5, total_steps, "Building RAG index...")
    from rag import setup_rag, load_embeddings

    faiss_path = os.path.join(profile["data_dir"], "faiss_index")
    embeddings = load_embeddings()

    hybrid_retriever, seasonal_lookup, _ = setup_rag(
        summaries_df,
        embeddings=embeddings,
        load_existing=False,
        faiss_path=faiss_path
    )

    # Step 6 — Done
    _progress(6, total_steps, "Initialization complete!")

    return (
        profile,
        summaries_df,
        output_df,
        hybrid_retriever,
        seasonal_lookup,
        embeddings
    )


def load_initialized_location(city_name):
    """
    Load a previously initialized location from disk.
    Returns same tuple as initialize_location or None.
    """
    profile = load_location_profile(city_name)
    if profile is None:
        return None

    data_dir = profile["data_dir"]

    output_df = pd.read_csv(
        os.path.join(data_dir, "all_zones_with_output.csv")
    )
    summaries_df = pd.read_csv(
        os.path.join(data_dir, "all_zones_window_summaries.csv")
    )

    from config import SEASON_MAP
    summaries_df["month"] = pd.to_datetime(
        summaries_df["window_start"]
    ).dt.month
    summaries_df["season"] = summaries_df["month"].map(SEASON_MAP)

    from rag import load_embeddings, setup_rag
    faiss_path = os.path.join(data_dir, "faiss_index")
    embeddings = load_embeddings()

    hybrid_retriever, seasonal_lookup, _ = setup_rag(
        summaries_df,
        embeddings=embeddings,
        load_existing=True,
        faiss_path=faiss_path
    )

    return (
        profile,
        summaries_df,
        output_df,
        hybrid_retriever,
        seasonal_lookup,
        embeddings
    )


if __name__ == "__main__":
    # Test geocoding and profile building
    profile = build_location_profile("Mumbai, India")
    if profile:
        print("Location profile:")
        for k, v in profile.items():
            if k != "zones":
                print(f"  {k}: {v}")
        print("  zones:")
        for z, c in profile["zones"].items():
            print(f"    {z}: {c}")