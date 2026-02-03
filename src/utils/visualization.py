"""
Helper functions for traffic visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_traffic_heatmap(tracks_df, title="Traffic Density Heatmap"):
    """
    Plot traffic density heatmap
    
    Args:
        tracks_df: DataFrame with trajectory data
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create 2D histogram
    plt.hist2d(tracks_df['xCenter'], tracks_df['yCenter'], 
               bins=50, cmap='YlOrRd')
    plt.colorbar(label='Number of observations')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def plot_speed_distribution(tracks_df, by_class=True):
    """
    Plot speed distribution
    
    Args:
        tracks_df: DataFrame with trajectory data
        by_class: Whether to separate by vehicle class
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if by_class and 'class' in tracks_df.columns:
        for vehicle_class in tracks_df['class'].unique():
            class_data = tracks_df[tracks_df['class'] == vehicle_class]
            ax.hist(class_data['speed'], bins=50, alpha=0.5, 
                   label=vehicle_class, density=True)
        ax.legend()
    else:
        ax.hist(tracks_df['speed'], bins=50, color='steelblue')
    
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_queue_analysis(features_df):
    """
    Plot queue length analysis
    
    Args:
        features_df: DataFrame with extracted features
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Queue length over time
    axes[0, 0].plot(features_df['timestamp'], features_df['avg_queue_length'], 
                    linewidth=2, color='crimson')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Average Queue Length')
    axes[0, 0].set_title('Queue Length Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Queue length distribution
    axes[0, 1].hist(features_df['avg_queue_length'], bins=30, 
                    color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Average Queue Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Queue Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Stopped vehicles percentage
    axes[1, 0].plot(features_df['timestamp'], features_df['percent_stopped'], 
                    linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Percentage Stopped (%)')
    axes[1, 0].set_title('Percentage of Stopped Vehicles Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation with total vehicles
    axes[1, 1].scatter(features_df['total_vehicles'], 
                      features_df['avg_queue_length'],
                      alpha=0.5, color='green')
    axes[1, 1].set_xlabel('Total Vehicles')
    axes[1, 1].set_ylabel('Average Queue Length')
    axes[1, 1].set_title('Queue Length vs Total Vehicles')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pedestrian_analysis(features_df):
    """
    Plot pedestrian activity analysis
    
    Args:
        features_df: DataFrame with extracted features
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pedestrian count over time
    axes[0].plot(features_df['timestamp'], features_df['pedestrian_count'],
                linewidth=2, color='purple')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pedestrian Count')
    axes[0].set_title('Pedestrian Count Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Waiting time distribution
    axes[1].hist(features_df['avg_ped_waiting_time'], bins=30,
                color='teal', edgecolor='black')
    axes[1].set_xlabel('Average Waiting Time (s)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Pedestrian Waiting Time Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Pedestrian speed
    axes[2].hist(features_df['avg_ped_speed'], bins=30,
                color='coral', edgecolor='black')
    axes[2].set_xlabel('Average Speed (m/s)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Pedestrian Speed Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_signal_timings(labeled_df, n_samples=100):
    """
    Visualize signal timing predictions
    
    Args:
        labeled_df: DataFrame with labeled data
        n_samples: Number of samples to plot
    """
    df_sample = labeled_df.head(n_samples)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Cycle time over time
    axes[0].plot(df_sample.index, df_sample['cycle_time'], 
                linewidth=2, color='darkblue', label='Cycle Time')
    axes[0].fill_between(df_sample.index, 60, 120, alpha=0.2, color='gray',
                         label='Typical Range (60-120s)')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Cycle Time (s)')
    axes[0].set_title('Signal Cycle Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Green times stacked
    phases = ['phase_1_green', 'phase_2_green', 'phase_3_green', 
              'phase_4_green', 'pedestrian_green']
    
    for phase in phases:
        if phase in df_sample.columns:
            axes[1].plot(df_sample.index, df_sample[phase], 
                        linewidth=2, label=phase.replace('_', ' ').title())
    
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Green Time (s)')
    axes[1].set_title('Phase Green Times')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_correlations(features_df, n_features=15):
    """
    Plot correlation heatmap of top features
    
    Args:
        features_df: DataFrame with features
        n_features: Number of top features to include
    """
    # Select numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    # Exclude non-feature columns
    exclude_cols = ['time_bin', 'timestamp', 'recording_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Select top features by variance
    variances = features_df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(n_features).index.tolist()
    
    # Calculate correlation matrix
    corr_matrix = features_df[top_features].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title(f'Feature Correlation Heatmap (Top {n_features} Features)')
    plt.tight_layout()
    
    return plt.gcf()


def create_comparison_dashboard(original_timings, optimized_timings):
    """
    Create dashboard comparing original vs optimized signal timings
    
    Args:
        original_timings: DataFrame with baseline timings
        optimized_timings: DataFrame with ML-predicted timings
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average cycle time comparison
    orig_cycle = original_timings['cycle_time'].mean()
    opt_cycle = optimized_timings['cycle_time'].mean()
    axes[0, 0].bar(['Original', 'Optimized'], [orig_cycle, opt_cycle],
                  color=['gray', 'green'])
    axes[0, 0].set_ylabel('Cycle Time (s)')
    axes[0, 0].set_title('Average Cycle Time Comparison')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Phase distribution
    phases = ['phase_1_green', 'phase_2_green', 'phase_3_green', 'phase_4_green']
    orig_phases = [original_timings[p].mean() for p in phases]
    opt_phases = [optimized_timings[p].mean() for p in phases]
    
    x = np.arange(len(phases))
    width = 0.35
    axes[0, 1].bar(x - width/2, orig_phases, width, label='Original', color='gray')
    axes[0, 1].bar(x + width/2, opt_phases, width, label='Optimized', color='green')
    axes[0, 1].set_ylabel('Green Time (s)')
    axes[0, 1].set_title('Average Phase Green Times')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['P1', 'P2', 'P3', 'P4'])
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Pedestrian green time
    orig_ped = original_timings['pedestrian_green'].mean()
    opt_ped = optimized_timings['pedestrian_green'].mean()
    axes[1, 0].bar(['Original', 'Optimized'], [orig_ped, opt_ped],
                  color=['gray', 'green'])
    axes[1, 0].set_ylabel('Green Time (s)')
    axes[1, 0].set_title('Pedestrian Green Time Comparison')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Improvement percentage
    improvements = {
        'Cycle Time': ((orig_cycle - opt_cycle) / orig_cycle) * 100,
        'Phase 1': ((orig_phases[0] - opt_phases[0]) / orig_phases[0]) * 100,
        'Phase 2': ((orig_phases[1] - opt_phases[1]) / orig_phases[1]) * 100,
        'Ped Green': ((orig_ped - opt_ped) / orig_ped) * 100
    }
    
    axes[1, 1].barh(list(improvements.keys()), list(improvements.values()),
                   color=['red' if v < 0 else 'green' for v in improvements.values()])
    axes[1, 1].set_xlabel('Change (%)')
    axes[1, 1].set_title('Improvement Over Baseline')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig