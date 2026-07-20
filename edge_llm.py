# edge_llm.py
"""
Solar Plant Copilot — Edge LLM Layer
Handles data pre-aggregation and Phi-4-Mini summarization.
This is the edge layer of the LLMSense two-stage pipeline.
"""

import ollama
import pandas as pd
from config import EDGE_LLM_MODEL, EDGE_LLM_MAX_SENTENCES


def format_window_for_summary(window_df):
    """
    Pre-aggregate a window of hourly data into structured text
    before passing to the edge LLM.
    Reduces token load significantly vs raw row-by-row input.
    Returns a structured string covering morning, peak, and afternoon.
    """
    window_df = window_df.copy()
    window_df["hour"] = pd.to_datetime(window_df["time"]).dt.hour

    morning = window_df[window_df["hour"].between(6, 11)]
    afternoon = window_df[window_df["hour"].between(12, 17)]

    if morning.empty or afternoon.empty:
        return "Insufficient data for summarization."

    peak_row = window_df.loc[window_df["ac_power_kw"].idxmax()]
    morning_gti_start = morning[morning['global_tilted_irradiance'] > 0]['global_tilted_irradiance'].iloc[0] if not morning[morning['global_tilted_irradiance'] > 0].empty else 0
    
    text = (
        f"Morning (06:00-11:00): "
        f"GTI rose from {morning_gti_start:.0f} "
        f"to {morning['global_tilted_irradiance'].max():.0f} W/m², "
        f"output from {morning[morning['ac_power_kw'] > 0]['ac_power_kw'].iloc[0] if not morning[morning['ac_power_kw'] > 0].empty else 0:.0f} "
        f"to {morning['ac_power_kw'].max():.0f} kW, "
        f"avg cloud cover {morning['cloud_cover'].mean():.0f}%, "
        f"temp {morning['temperature_2m'].iloc[0]:.1f} "
        f"to {morning['temperature_2m'].iloc[-1]:.1f}°C. "

        f"Peak: GTI {peak_row['global_tilted_irradiance']:.0f} W/m², "
        f"output {peak_row['ac_power_kw']:.0f} kW "
        f"at {pd.to_datetime(peak_row['time']).strftime('%H:%M')}. "

        f"Afternoon (12:00-17:00): "
        f"GTI declined from {afternoon['global_tilted_irradiance'].iloc[0]:.0f} "
        f"to {afternoon['global_tilted_irradiance'].iloc[-1]:.0f} W/m², "
        f"output from {afternoon['ac_power_kw'].iloc[0]:.0f} "
        f"to {afternoon['ac_power_kw'].iloc[-1]:.0f} kW, "
        f"avg cloud cover {afternoon['cloud_cover'].mean():.0f}%, "
        f"temp {afternoon['temperature_2m'].iloc[0]:.1f} "
        f"to {afternoon['temperature_2m'].iloc[-1]:.1f}°C."
    )

    return text


def get_local_summary(data_text):
    """
    Send pre-aggregated sensor data text to Phi-4-Mini for summarization.
    Uses few-shot prompting to preserve exact numbers and avoid hallucination.
    Post-processes to trim verbosity before passing to cloud LLM.
    Returns trimmed summary string.
    """
    response = ollama.chat(
        model=EDGE_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a solar plant data summarizer. "
                    "Summarize sensor data concisely using only the "
                    "information provided. Always preserve exact numbers. "
                    "Never add external information."
                )
            },
            {
                "role": "user",
                "content": (
                    "Summarize this data: Irradiance was stable at 650 W/m² "
                    "from 10am to 12pm. Output held steady at 2800 kW. "
                    "Temperature rose from 30°C to 34°C."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "10:00-12:00: Irradiance stable at 650 W/m², output "
                    "steady at 2800 kW. Temperature rose moderately from "
                    "30°C to 34°C. No anomalies observed."
                )
            },
            {
                "role": "user",
                "content": f"Summarize this data: {data_text}"
            }
        ]
    )

    raw = response["message"]["content"]

    # Post-processing: keep only first N sentences
    sentences = raw.replace("\n", " ").split(".")
    trimmed = ". ".join(
        s.strip() for s in sentences[:EDGE_LLM_MAX_SENTENCES] if s.strip()
    ) + "."

    return trimmed


def summarize_window(window_df):
    """
    End-to-end edge summarization for a window DataFrame.
    Combines format_window_for_summary + get_local_summary.
    Returns (structured_text, local_summary) tuple.
    """
    structured_text = format_window_for_summary(window_df)
    local_summary = get_local_summary(structured_text)
    return structured_text, local_summary


if __name__ == "__main__":
    print("Edge LLM module loaded.")
    print(f"Model: {EDGE_LLM_MODEL}")
    print(f"Max sentences: {EDGE_LLM_MAX_SENTENCES}")