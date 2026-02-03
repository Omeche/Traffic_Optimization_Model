# Traffic Signal Optimization Using inD Dataset

## Project Overview
Machine learning-based traffic signal control system using real-world intersection data from the inD (intersection Drone) dataset to optimize signal timings for both vehicles and pedestrians.

## Project Structure
```
traffic-signal-optimization/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # inD dataset here
│   ├── processed/              # Processed features
│   └── labels/                 # Generated signal timing labels
├── src/
│   ├── 1_data_preprocessing.py
│   ├── 2_feature_extraction.py
│   ├── 3_label_generation.py
│   ├── 4_model_training.py
│   ├── 5_evaluation.py
│   └── utils/
│       ├── visualization.py
│       └── signal_logic.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── models/
│   └── saved_models/
└── results/
    
```

## InD Dataset Structure
The inD dataset contains:
- **Recordings**: 32 recordings from 4 intersections
- **Files per recording**:
  - `XX_tracks.csv`: Vehicle/pedestrian trajectories
  - `XX_tracksMeta.csv`: Metadata for each track
  - `XX_recordingMeta.csv`: Recording information

### Key Fields in Tracks Data:
- `trackId`: Unique identifier for each road user
- `frame`: Frame number (25 fps)
- `xCenter, yCenter`: Position coordinates
- `xVelocity, yVelocity`: Velocities
- `class`: Type (car, truck_bus, pedestrian, bicycle)
- `laneId`: Lane identifier

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing
```bash
python src/1_data_preprocessing.py --data_path data/raw/inD-dataset-v1.1/data
```

### Step 2: Feature Extraction
```bash
python src/2_feature_extraction.py --time_window 30
```

### Step 3: Label Generation
```bash
python src/3_label_generation.py --method webster
```

### Step 4: Model Training
```bash
python src/4_model_training.py --models all
```

### Step 5: Evaluation
```bash
python src/5_evaluation.py --visualize
```

## Features Extracted
### Vehicular Features:
- Vehicle count per approach
- Queue length per lane
- Average speed per approach
- Lane occupancy
- Turning movement counts

### Pedestrian Features:
- Pedestrian count at crossings
- Average waiting time
- Crossing demand intensity
- Pedestrian density

## Models Implemented
1. **Decision Tree Regressor**
2. **Random Forest Regressor**
3. **Support Vector Regression (SVR)**
4. **Neural Network (MLP)**
5. **Gradient Boosting Regressor**

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Average vehicle delay reduction
- Pedestrian waiting time reduction

## Results
Results will be saved in `results/` including:
- Model performance comparison
- Feature importance plots
- Signal timing predictions vs. baseline
- Traffic flow improvements

## References
- inD Dataset: https://www.ind-dataset.com/
- Bock et al. "The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections"
@inproceedings{inDdataset,
title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
pages={1929-1934},
year={2020},
doi={10.1109/IV47402.2020.9304839}}

## Author
Traffic Signal Optimization Project
Date: January 2026