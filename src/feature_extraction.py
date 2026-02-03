"""
Extracts traffic and pedestrian features from preprocessed inD data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

class TrafficFeatureExtractor:
    """Extract traffic, pedestrian, and bicycle features for signal optimization"""

    def __init__(self, time_window=30):
        """
        Initialize feature extractor
        Args:
            time_window: Time window in seconds for feature aggregation
        """
        self.time_window = time_window

    def extract_vehicle_features(self, tracks_df, time_bin):
        """
        Extract vehicular traffic features for a time window
        Args:
            tracks_df: Tracks DataFrame for a specific time bin
            time_bin: Time bin identifier
        Returns:
            Dictionary of vehicular features
        """
        vehicles = tracks_df[tracks_df['class'].isin(['car', 'truck_bus'])].copy()

        if len(vehicles) == 0:
            return self._get_empty_vehicle_features(time_bin)

        features = {
            'time_bin': time_bin,
            'timestamp': vehicles['frame'].min() / 25.0,  # frame -> seconds
            'total_vehicles': vehicles['trackId'].nunique(),
            'car_count': vehicles[vehicles['class'] == 'car']['trackId'].nunique(),
            'truck_count': vehicles[vehicles['class'] == 'truck_bus']['trackId'].nunique()
        }

        # Approximate stopped vehicles: speed ~ 0
        stopped = vehicles[(vehicles['xVelocity'].abs() < 0.1) & (vehicles['yVelocity'].abs() < 0.1)]
        features['total_stopped_vehicles'] = stopped['trackId'].nunique()
        features['avg_queue_length'] = 0  # lane info not available
        features['max_queue_length'] = 0

        # Speed analysis
        speeds = np.sqrt(vehicles['xVelocity']**2 + vehicles['yVelocity']**2)
        features['avg_speed'] = speeds.mean()
        features['median_speed'] = speeds.median()
        features['speed_std'] = speeds.std()
        features['percent_stopped'] = (stopped['trackId'].nunique() / features['total_vehicles'] * 100
                                       if features['total_vehicles'] > 0 else 0)

        # Lane-based features not available
        features['avg_lane_occupancy'] = 0
        features['max_lane_occupancy'] = 0
        features['num_active_lanes'] = 0

        # Traffic density (vehicles per unit area)
        area = (vehicles['xCenter'].max() - vehicles['xCenter'].min()) * \
               (vehicles['yCenter'].max() - vehicles['yCenter'].min())
        features['traffic_density'] = features['total_vehicles'] / max(area, 1)

        # Flow rate (vehicles per minute)
        time_span = (vehicles['frame'].max() - vehicles['frame'].min()) / 25.0
        features['flow_rate'] = (features['total_vehicles'] / max(time_span, 1)) * 60

        # Acceleration patterns
        accel = np.sqrt(vehicles['xAcceleration']**2 + vehicles['yAcceleration']**2)
        features['avg_acceleration'] = accel.mean()
        features['harsh_braking_count'] = len(vehicles[(vehicles['xAcceleration'] < -2.0) | (vehicles['yAcceleration'] < -2.0)])

        return features

    def extract_pedestrian_features(self, tracks_df, time_bin):
        """Extract pedestrian features for a time window"""
        pedestrians = tracks_df[tracks_df['class'] == 'pedestrian'].copy()
        features = {}

        if len(pedestrians) == 0:
            features.update({
                'pedestrian_count': 0,
                'avg_ped_waiting_time': 0,
                'max_ped_waiting_time': 0,
                'ped_crossing_demand': 0,
                'avg_ped_speed': 0,
                'stopped_pedestrians': 0
            })
            return features

        features['pedestrian_count'] = pedestrians['trackId'].nunique()
        speeds = np.sqrt(pedestrians['xVelocity']**2 + pedestrians['yVelocity']**2)
        stopped_peds = pedestrians[speeds < 0.3]

        if len(stopped_peds) > 0:
            waiting_times = stopped_peds.groupby('trackId')['frame'].count() / 25.0
            features['avg_ped_waiting_time'] = waiting_times.mean()
            features['max_ped_waiting_time'] = waiting_times.max()
            features['stopped_pedestrians'] = len(waiting_times)
        else:
            features['avg_ped_waiting_time'] = 0
            features['max_ped_waiting_time'] = 0
            features['stopped_pedestrians'] = 0

        features['ped_crossing_demand'] = features['pedestrian_count']
        features['avg_ped_speed'] = speeds.mean()

        return features

    def extract_bicycle_features(self, tracks_df, time_bin):
        """Extract bicycle features for a time window"""
        bicycles = tracks_df[tracks_df['class'] == 'bicycle'].copy()
        if len(bicycles) == 0:
            return {'bicycle_count': 0, 'avg_bicycle_speed': 0}

        speeds = np.sqrt(bicycles['xVelocity']**2 + bicycles['yVelocity']**2)
        return {
            'bicycle_count': bicycles['trackId'].nunique(),
            'avg_bicycle_speed': speeds.mean()
        }

    def extract_temporal_features(self, time_bin, frame_value):
        """Extract time-of-day features based on frame number"""
        base_time = pd.Timestamp('2020-01-01 08:00:00')
        current_time = base_time + pd.Timedelta(seconds=frame_value / 25.0)
        return {
            'hour_of_day': current_time.hour,
            'minute_of_hour': current_time.minute,
            'time_of_day_sin': np.sin(2 * np.pi * current_time.hour / 24),
            'time_of_day_cos': np.cos(2 * np.pi * current_time.hour / 24)
        }

    def _get_empty_vehicle_features(self, time_bin):
        """Return empty vehicle features if no vehicles present"""
        return {
            'time_bin': time_bin,
            'timestamp': 0,
            'total_vehicles': 0,
            'car_count': 0,
            'truck_count': 0,
            'total_stopped_vehicles': 0,
            'avg_queue_length': 0,
            'max_queue_length': 0,
            'avg_speed': 0,
            'median_speed': 0,
            'speed_std': 0,
            'percent_stopped': 0,
            'avg_lane_occupancy': 0,
            'max_lane_occupancy': 0,
            'num_active_lanes': 0,
            'traffic_density': 0,
            'flow_rate': 0,
            'avg_acceleration': 0,
            'harsh_braking_count': 0
        }

    def extract_all_features(self, tracks_df):
        """Extract all features for a recording"""
        tracks_df = tracks_df.copy()
        tracks_df['time_bin'] = (tracks_df['frame'] / (self.time_window * 25)).astype(int)

        all_features = []
        for time_bin in sorted(tracks_df['time_bin'].unique()):
            bin_data = tracks_df[tracks_df['time_bin'] == time_bin]

            vehicle_features = self.extract_vehicle_features(bin_data, time_bin)
            ped_features = self.extract_pedestrian_features(bin_data, time_bin)
            bike_features = self.extract_bicycle_features(bin_data, time_bin)
            temporal_features = self.extract_temporal_features(time_bin, bin_data['frame'].min())

            combined_features = {**vehicle_features, **ped_features, **bike_features, **temporal_features}
            all_features.append(combined_features)

        return pd.DataFrame(all_features)

    def process_all_recordings(self, processed_data_path, output_path):
        """Process all recordings and extract features"""
        processed_path = Path(processed_data_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        recording_files = sorted(processed_path.glob("*_processed.pkl"))

        print(f"\nExtracting features from {len(recording_files)} recordings...")
        print(f"   Time window: {self.time_window} seconds")

        all_features_list = []
        for rec_file in tqdm(recording_files, desc="Extracting features"):
            try:
                with open(rec_file, 'rb') as f:
                    rec_data = pickle.load(f)

                tracks_df = rec_data['tracks']
                recording_id = rec_file.stem.replace('_processed', '')

                features_df = self.extract_all_features(tracks_df)
                features_df['recording_id'] = recording_id

                features_df.to_csv(output_path / f"{recording_id}_features.csv", index=False)
                all_features_list.append(features_df)
            except Exception as e:
                print(f"\nError processing {rec_file.name}: {e}")
                continue

        if all_features_list:
            combined_features = pd.concat(all_features_list, ignore_index=True)
            combined_features.to_csv(output_path / "all_features_combined.csv", index=False)
            print(f"\nFeature extraction complete. Saved to {output_path}")
            return combined_features
        else:
            print("No features extracted!")
            return None


def main():
    parser = argparse.ArgumentParser(description='Extract traffic features from inD dataset')
    parser.add_argument('--processed_path', type=str, default='data/processed', help='Path to preprocessed data')
    parser.add_argument('--output_path', type=str, default='data/features', help='Path to save extracted features')
    parser.add_argument('--time_window', type=int, default=30, help='Time window in seconds for feature aggregation')
    args = parser.parse_args()

    extractor = TrafficFeatureExtractor(time_window=args.time_window)
    features = extractor.process_all_recordings(args.processed_path, args.output_path)

    if features is not None:
        print("\nSample features:")
        print(features.head())
        print("\nFeature statistics:")
        print(features.describe())


if __name__ == "__main__":
    main()
