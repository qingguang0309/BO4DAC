# AI-Driven Catalyst Optimization System

An advanced AI system for optimizing catalyst formulations for CO2 capture using Bayesian optimization and machine learning techniques.

## Features

- **Bayesian Optimization**: Advanced optimization algorithm to suggest the next best catalyst formulations to test
- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Expected Improvement**: Quantifies the potential improvement of suggested formulations
- **Web Interface**: Intuitive web-based interface for easy interaction
- **Real-time Visualization**: Interactive charts showing optimization progress
- **Dynamic Configuration**: Ability to change search bounds and experimental conditions after initialization
- **Data Archiving**: Automatic archiving of old experiment records to manage data size
- **Smart Filtering**: Filters historical data based on similarity to current experimental conditions
- **Direct Data Input**: Form for users to input experimental data that matches their search bounds

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-catalyst-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://127.0.0.1:5001`

## Usage

### Step 1: Initialize the System
- Define your search space by selecting categorical options (supports, amines, additives) and numerical bounds (MW, organic content)
- Set your experimental conditions (temperature, humidity, CO2 concentration, flow rate)

### Step 2: Generate Candidates
- Click "Generate Next Candidate" to get AI-suggested catalyst formulations
- Review the suggestions along with their predicted capacity, uncertainty, and expected improvement

### Step 3: Test and Record Results
- Test the suggested formulations in your lab
- Record the actual CO2 capacity results using the "Submit Experiment" buttons
- Or use the "Input Custom Experimental Data" form to add results that match your current search bounds

### Step 4: Monitor Progress
- View optimization progress in real-time with interactive charts
- Track best capacity achieved and convergence trends

## Key Components

### Optimization Algorithm
- Uses Gaussian Process regression for modeling
- Implements Expected Improvement acquisition function
- Provides uncertainty quantification with 95% confidence intervals

### Data Management
- Automatically archives old experiments (keeps last 50)
- Filters historical data based on similarity to current conditions
- Preserves all historical data while focusing on relevant experiments

### Web Interface
- Responsive design with step-by-step workflow
- Real-time updates of optimization progress
- Clear guidance for users throughout the process

## API Endpoints

- `GET /` - Main web interface
- `POST /api/generate-candidates` - Generate new candidates
- `POST /api/record-experiment` - Record experimental results
- `POST /api/update-config` - Update search bounds and conditions
- `POST /api/archive-experiments` - Manually archive old experiments
- `POST /api/input-custom-experimental-results` - Input custom experimental data
- `GET /api/status` - Get system status
- `GET /api/export-data` - Export optimization data
- `POST /api/reset-system` - Reset the system

## Troubleshooting

- If you see "No data available for filtering" message, input at least 5 experimental results that match your current search bounds and conditions
- If the optimization isn't progressing, ensure you're submitting actual experimental results
- For best results, test suggested candidates with high expected improvement values

## Technical Details

The system implements a sophisticated Bayesian optimization pipeline:

1. **Gaussian Process Model**: Models the relationship between catalyst formulation and CO2 capacity
2. **Acquisition Function**: Uses Expected Improvement to balance exploration and exploitation
3. **Uncertainty Quantification**: Provides confidence intervals for all predictions
4. **Smart Filtering**: Dynamically focuses on relevant historical data based on experimental conditions
5. **Dynamic Reconfiguration**: Allows changing search bounds and conditions after initialization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.