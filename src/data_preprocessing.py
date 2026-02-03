"""
Data Preprocessing Script
Loads and preprocesses the inD dataset trajectory files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle


class InDDataPreprocessor:
    """Preprocessor for inD dataset"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.recordings = []

    def discover_recordings(self):
        for file in sorted(self.data_path.glob("*_tracks.csv")):
            recording_id = file.stem.replace("_tracks", "")
            self.recordings.append(recording_id)

        print(f"Found {len(self.recordings)} recordings: {self.recordings}")
        return self.recordings

    def load_recording(self, recording_id):
        data = {}

        tracks_file = self.data_path / f"{recording_id}_tracks.csv"
        meta_file = self.data_path / f"{recording_id}_tracksMeta.csv"
        rec_meta_file = self.data_path / f"{recording_id}_recordingMeta.csv"

        if tracks_file.exists():
            data["tracks"] = pd.read_csv(tracks_file)

        if meta_file.exists():
            data["tracksMeta"] = pd.read_csv(meta_file)

        if rec_meta_file.exists():
            data["recordingMeta"] = pd.read_csv(rec_meta_file)

        return data

    def preprocess_tracks(self, tracks_df, recording_meta):
        df = tracks_df.copy()

        fps = 25
        if recording_meta is not None and 'frameRate' in recording_meta.columns:
            fps = recording_meta['frameRate'].iloc[0]

        df["time"] = df["frame"] / fps
        df["speed"] = np.sqrt(df["xVelocity"] ** 2 + df["yVelocity"] ** 2)

        df["xAcceleration"] = df.groupby("trackId")["xVelocity"].diff() * fps
        df["yAcceleration"] = df.groupby("trackId")["yVelocity"].diff() * fps
        df["acceleration"] = np.sqrt(
            df["xAcceleration"] ** 2 + df["yAcceleration"] ** 2
        )

        df["is_stopped"] = df["speed"] < 0.5
        df["heading"] = np.arctan2(df["yVelocity"], df["xVelocity"])

        return df

    def add_derived_features(self, tracks_df, meta_df):
        meta_cols = [
            "trackId",
            "class",
            "width",
            "height",
            "initialFrame",
            "finalFrame",
            "numFrames",
        ]

        if meta_df is None:
            return tracks_df

        available_cols = [c for c in meta_cols if c in meta_df.columns]
        return tracks_df.merge(meta_df[available_cols], on="trackId", how="left")

    def filter_intersection_area(self, df):
        x_center = df["xCenter"].median()
        y_center = df["yCenter"].median()

        x_range = df["xCenter"].std() * 2
        y_range = df["yCenter"].std() * 2

        return df[
            (df["xCenter"] >= x_center - x_range)
            & (df["xCenter"] <= x_center + x_range)
            & (df["yCenter"] >= y_center - y_range)
            & (df["yCenter"] <= y_center + y_range)
        ].copy()

    def process_all_recordings(self, output_path, sample_recordings=None):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        recs = sample_recordings if sample_recordings else self.recordings
        print(f"\nProcessing {len(recs)} recordings...")

        processed = {}

        for rec_id in tqdm(recs, desc="Processing recordings"):
            try:
                data = self.load_recording(rec_id)

                if "tracks" not in data:
                    print(f"Skipping {rec_id}: No tracks file")
                    continue

                tracks = self.preprocess_tracks(
                    data["tracks"], data.get("recordingMeta")
                )

                tracks = self.add_derived_features(
                    tracks, data.get("tracksMeta")
                )

                tracks = self.filter_intersection_area(tracks)

                processed[rec_id] = {
                    "tracks": tracks,
                    "tracksMeta": data.get("tracksMeta"),
                    "recordingMeta": data.get("recordingMeta"),
                }

                with open(output_path / f"{rec_id}_processed.pkl", "wb") as f:
                    pickle.dump(processed[rec_id], f)

            except Exception as e:
                print(f"Error processing {rec_id}: {e}")

        with open(output_path / "processing_summary.pkl", "wb") as f:
            pickle.dump(
                {
                    "recordings_processed": list(processed.keys()),
                    "total_recordings": len(processed),
                    "timestamp": pd.Timestamp.now(),
                },
                f,
            )

        print("\nPreprocessing complete")
        print(f"Processed: {len(processed)} recordings")
        print(f"Saved to: {output_path}")

        return processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess inD dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw",
        help="Path to raw inD dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed",
        help="Path to save processed data",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of recordings to process",
    )

    args = parser.parse_args()

    preprocessor = InDDataPreprocessor(args.data_path)
    recordings = preprocessor.discover_recordings()

    if not recordings:
        print("No recordings found")
        return

    sample = recordings[: args.sample] if args.sample else None
    preprocessor.process_all_recordings(args.output_path, sample)


if __name__ == "__main__":
    main()
