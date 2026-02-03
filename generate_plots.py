"""
Additional Visualization Script
Run after main evaluation to create detailed visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def generate_all_visualizations():
    """Generate all visualization plots """
    
    print("Generating additional visualizations...")
    
    # Create results directory
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    # Load data
    try:
        features_df = pd.read_csv('data/features/all_features_combined.csv')
        labeled_df = pd.read_csv('data/labels/labeled_data_adaptive.csv')
        print(f"Loaded {len(features_df)} feature samples")
        print(f"Loaded {len(labeled_df)} labeled samples")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Traffic volume distribution
    print("\nCreating traffic volume distribution plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(features_df['total_vehicles'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Total Vehicles', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Traffic Volume Across Time Windows', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(results_path / 'traffic_volume_distribution.png')
    plt.close()
    print("   Saved: traffic_volume_distribution.png")
    
    # Speed distribution
    print("Creating speed distribution plot...")
    plt.figure(figsize=(10, 6))
    plt.hist(features_df['avg_speed'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Average Speed (m/s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Average Vehicle Speed', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(results_path / 'speed_distribution.png')
    plt.close()
    print("   Saved: speed_distribution.png")
    
    # Pedestrian analysis
    print("Creating pedestrian activity plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(features_df['pedestrian_count'], bins=20, color='purple', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Pedestrian Count', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Pedestrian Count Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(features_df['avg_ped_waiting_time'], bins=25, color='teal', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Average Waiting Time (s)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Pedestrian Waiting Time Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'pedestrian_analysis.png')
    plt.close()
    print("   Saved: pedestrian_analysis.png")
    
    # Label distribution (signal timings)
    print("Creating signal timing distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    timing_params = [
        ('cycle_time', 'Cycle Time (s)', 'darkblue'),
        ('phase_1_green', 'Phase 1 Green Time (s)', 'green'),
        ('phase_2_green', 'Phase 2 Green Time (s)', 'limegreen'),
        ('phase_3_green', 'Phase 3 Green Time (s)', 'yellowgreen'),
        ('phase_4_green', 'Phase 4 Green Time (s)', 'olive'),
        ('pedestrian_green', 'Pedestrian Green Time (s)', 'orange')
    ]
    
    for idx, (col, label, color) in enumerate(timing_params):
        row = idx // 3
        col_idx = idx % 3
        axes[row, col_idx].hist(labeled_df[col], bins=25, color=color, edgecolor='black', alpha=0.7)
        axes[row, col_idx].set_xlabel(label, fontsize=10)
        axes[row, col_idx].set_ylabel('Frequency', fontsize=10)
        axes[row, col_idx].set_title(f'{label} Distribution', fontsize=11, fontweight='bold')
        axes[row, col_idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'signal_timing_distributions.png')
    plt.close()
    print("   Saved: signal_timing_distributions.png")
    
    # Feature correlation heatmap (top features)
    print("Creating feature correlation heatmap...")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['time_bin', 'timestamp', 'recording_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Select top 12 features by variance
    variances = features_df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(12).index.tolist()
    
    corr_matrix = features_df[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix (Top 12 Features)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(results_path / 'feature_correlation_heatmap.png')
    plt.close()
    print("   Saved: feature_correlation_heatmap.png")
    
    # Traffic density vs signal timing
    print("Creating traffic density vs signal timing plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(labeled_df['traffic_density'], labeled_df['cycle_time'], 
                alpha=0.5, c=labeled_df['total_vehicles'], cmap='viridis', s=30)
    plt.colorbar(label='Total Vehicles')
    plt.xlabel('Traffic Density', fontsize=12)
    plt.ylabel('Cycle Time (s)', fontsize=12)
    plt.title('Traffic Density vs Cycle Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(results_path / 'density_vs_cycletime.png')
    plt.close()
    print("   Saved: density_vs_cycletime.png")
    
    # Stopped vehicles vs queue (if queue data available)
    print("Creating stopped vehicles analysis...")
    plt.figure(figsize=(10, 6))
    plt.scatter(labeled_df['total_stopped_vehicles'], labeled_df['cycle_time'],
                alpha=0.6, color='crimson', s=40)
    plt.xlabel('Total Stopped Vehicles', fontsize=12)
    plt.ylabel('Cycle Time (s)', fontsize=12)
    plt.title('Stopped Vehicles vs Cycle Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(results_path / 'stopped_vehicles_analysis.png')
    plt.close()
    print("   Saved: stopped_vehicles_analysis.png")
    
    # Time series of traffic parameters
    print("Creating time series plot...")
    sample_data = features_df.head(100)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(sample_data['timestamp'], sample_data['total_vehicles'], 
                linewidth=2, color='steelblue', marker='o', markersize=3)
    axes[0].set_ylabel('Total Vehicles', fontsize=11)
    axes[0].set_title('Traffic Parameters Over Time (First 100 Samples)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sample_data['timestamp'], sample_data['avg_speed'], 
                linewidth=2, color='green', marker='s', markersize=3)
    axes[1].set_ylabel('Average Speed (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sample_data['timestamp'], sample_data['pedestrian_count'], 
                linewidth=2, color='purple', marker='^', markersize=3)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Pedestrian Count', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / 'traffic_timeseries.png')
    plt.close()
    print("   Saved: traffic_timeseries.png")
    
    # Pedestrian count vs pedestrian green time
    print("Creating pedestrian demand vs green time plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(labeled_df['pedestrian_count'], labeled_df['pedestrian_green'],
                alpha=0.6, c=labeled_df['avg_ped_waiting_time'], cmap='plasma', s=50)
    plt.colorbar(label='Avg Waiting Time (s)')
    plt.xlabel('Pedestrian Count', fontsize=12)
    plt.ylabel('Pedestrian Green Time (s)', fontsize=12)
    plt.title('Pedestrian Count vs Allocated Green Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(results_path / 'pedestrian_demand_vs_greentime.png')
    plt.close()
    print("   Saved: pedestrian_demand_vs_greentime.png")
    
    # Summary statistics table
    print("Creating summary statistics...")
    summary_stats = pd.DataFrame({
        'Parameter': [
            'Total Vehicles',
            'Average Speed (m/s)',
            'Pedestrian Count',
            'Traffic Density',
            'Cycle Time (s)',
            'Phase 1 Green (s)',
            'Pedestrian Green (s)'
        ],
        'Mean': [
            features_df['total_vehicles'].mean(),
            features_df['avg_speed'].mean(),
            features_df['pedestrian_count'].mean(),
            features_df['traffic_density'].mean(),
            labeled_df['cycle_time'].mean(),
            labeled_df['phase_1_green'].mean(),
            labeled_df['pedestrian_green'].mean()
        ],
        'Std Dev': [
            features_df['total_vehicles'].std(),
            features_df['avg_speed'].std(),
            features_df['pedestrian_count'].std(),
            features_df['traffic_density'].std(),
            labeled_df['cycle_time'].std(),
            labeled_df['phase_1_green'].std(),
            labeled_df['pedestrian_green'].std()
        ],
        'Min': [
            features_df['total_vehicles'].min(),
            features_df['avg_speed'].min(),
            features_df['pedestrian_count'].min(),
            features_df['traffic_density'].min(),
            labeled_df['cycle_time'].min(),
            labeled_df['phase_1_green'].min(),
            labeled_df['pedestrian_green'].min()
        ],
        'Max': [
            features_df['total_vehicles'].max(),
            features_df['avg_speed'].max(),
            features_df['pedestrian_count'].max(),
            features_df['traffic_density'].max(),
            labeled_df['cycle_time'].max(),
            labeled_df['phase_1_green'].max(),
            labeled_df['pedestrian_green'].max()
        ]
    })
    
    summary_stats.to_csv(results_path / 'summary_statistics.csv', index=False, float_format='%.2f')
    print("   Saved: summary_statistics.csv")
    
    print("\nAll visualizations generated successfully")
    print(f"Location: {results_path}/")

if __name__ == "__main__":
    generate_all_visualizations()