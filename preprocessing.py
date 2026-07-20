# preprocessing.py
"""
Solar Plant Copilot — Preprocessing
Handles zenith angle calculation, daylight filtering,
and plant output simulation using PVWatts model.
"""

import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib import temperature, pvsystem
from config import (
    ZONES, TIMEZONE, DATA_DIR,
    PDC0, GAMMA_PDC, INVERTER_EFFICIENCY,
    SYSTEM_LOSSES, U0, U1,
    ZENITH_CSV, DAYLIGHT_CSV, OUTPUT_CSV
)


def add_zenith_angles(combined_df):
    """
    Calculate solar zenith and elevation angles for each zone.
    Uses pvlib's apparent_zenith (accounts for atmospheric refraction).
    Returns DataFrame with solar_zenith and solar_elevation columns added.
    """
    zenith_dfs = []

    for zone_name, coords in ZONES.items():
        zone_subset = combined_df[combined_df["zone"] == zone_name].copy()

        # Localize timestamps to IST (Open-Meteo returns IST but naive)
        zone_subset["time"] = pd.to_datetime(zone_subset["time"])
        zone_subset["time"] = zone_subset["time"].dt.tz_localize(TIMEZONE)

        loc = Location(
            latitude=coords["lat"],
            longitude=coords["lon"],
            tz=TIMEZONE
        )

        solpos = loc.get_solarposition(zone_subset["time"])
        zone_subset["solar_zenith"] = solpos["apparent_zenith"].values
        zone_subset["solar_elevation"] = solpos["apparent_elevation"].values

        zenith_dfs.append(zone_subset)
        print(f"{zone_name} zenith calculated — {len(zone_subset)} rows")

    return pd.concat(zenith_dfs, ignore_index=True)


def filter_daylight(df):
    """
    Filter to daylight hours only based on actual recorded irradiance.
    Uses shortwave_radiation > 0 as the filter condition.
    Returns filtered DataFrame.
    """
    daylight_df = df[df["shortwave_radiation"] > 0].copy()

    print(f"Original rows: {len(df)}")
    print(f"Daylight rows: {len(daylight_df)}")
    print(f"Dropped: {len(df) - len(daylight_df)}")
    print(f"Daylight %: {round(len(daylight_df) / len(df) * 100, 2)}")

    return daylight_df


def simulate_plant_output(df):
    """
    Simulate AC power output per zone using PVWatts model.
    Steps:
    1. Faiman cell temperature model
    2. PVWatts DC power
    3. Apply inverter efficiency + system losses -> AC power
    Returns DataFrame with cell_temperature, dc_power_kw, ac_power_kw added.
    """
    df = df.copy()

    # Step 1 — Cell temperature (Faiman model)
    df["cell_temperature"] = temperature.faiman(
        poa_global=df["global_tilted_irradiance"],
        temp_air=df["temperature_2m"],
        wind_speed=df["wind_speed_10m"],
        u0=U0,
        u1=U1
    )

    # Step 2 — DC power (PVWatts)
    df["dc_power_kw"] = pvsystem.pvwatts_dc(
        g_poa_effective=df["global_tilted_irradiance"],
        temp_cell=df["cell_temperature"],
        pdc0=PDC0,
        gamma_pdc=GAMMA_PDC
    ) / 1000

    # Step 3 — AC power (inverter + system losses)
    df["ac_power_kw"] = (
        df["dc_power_kw"] * INVERTER_EFFICIENCY * (1 - SYSTEM_LOSSES)
    ).clip(lower=0)

    print(f"Plant output simulated — {len(df)} rows")
    print(f"Peak AC output: {df['ac_power_kw'].max():.1f} kW")
    print(f"Any NaNs in ac_power_kw: {df['ac_power_kw'].isna().sum()}")

    return df


def run_preprocessing(combined_df=None, save=True):
    """
    Full preprocessing pipeline:
    1. Load combined data if not provided
    2. Add zenith angles
    3. Save zenith dataset
    4. Filter to daylight hours
    5. Simulate plant output
    6. Save final datasets
    Returns (zenith_df, daylight_df, output_df)
    """
    # Load if not provided
    if combined_df is None:
        print("Loading combined dataset...")
        combined_df = pd.read_csv(f"{DATA_DIR}/all_zones_weather_combined.csv")

    # Step 1 — Zenith angles
    print("\n── Step 1: Zenith Angle Calculation ──")
    zenith_df = add_zenith_angles(combined_df)
    if save:
        zenith_df.to_csv(ZENITH_CSV, index=False)
        print(f"Saved -> {ZENITH_CSV}")

    # Step 2 — Daylight filter
    print("\n── Step 2: Daylight Filtering ──")
    daylight_df = filter_daylight(zenith_df)
    if save:
        daylight_df.to_csv(DAYLIGHT_CSV, index=False)
        print(f"Saved -> {DAYLIGHT_CSV}")

    # Step 3 — Plant output simulation
    print("\n── Step 3: Plant Output Simulation ──")
    output_df = simulate_plant_output(daylight_df)
    if save:
        output_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved -> {OUTPUT_CSV}")

    return zenith_df, daylight_df, output_df


if __name__ == "__main__":
    zenith_df, daylight_df, output_df = run_preprocessing()
    print("\nPreprocessing complete.")
    print(output_df[["time", "zone", "ac_power_kw"]].head())