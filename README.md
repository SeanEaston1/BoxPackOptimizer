# 3D Box Arrangement Optimizer

An interactive Streamlit application that optimizes the arrangement of boxes in a truck space, considering dimensions, weights, and physical constraints.

## Features

### Input Configuration
- Truck dimensions (length, width, height)
- Multiple box configurations (up to 10 boxes)
- Individual box dimensions and weights
- Maximum total weight limit
- Color scheme selection (Default, Pastel, Bold, Grayscale)

### Optimization Capabilities
- Optimal box arrangement calculation
- Weight distribution consideration
- Multiple rotation possibilities for each box
- Space utilization optimization
- Weight limit enforcement
- Stacking rules (heavier boxes placed lower)

### Visualization
- Multiple 3D views:
  - Default 3D View
  - Front View
  - Back View
  - Top View
  - Bottom View
- Color-coded box representation
- Interactive legend with box details
- Transparent visualization for better understanding

### Statistics and Analysis
- Truck volume calculation
- Utilized space measurement
- Space utilization percentage
- Individual box placement details
- Total weight calculation
- Weight distribution analysis

### Export Capabilities
- Arrangement export to JSON format
- Complete configuration backup
- Statistics export

## How to Use

1. Configure Input Parameters:
   - Set truck dimensions in the sidebar
   - Specify number of boxes (1-10)
   - Enter dimensions for each box
   - Set weight for each box
   - Define maximum weight limit
   - Choose preferred color scheme

2. Run Optimization:
   - Click "Optimize Arrangement" button
   - View real-time optimization progress

3. Analyze Results:
   - Examine multiple 3D views
   - Review space utilization statistics
   - Check weight distribution
   - Verify arrangement details

4. Export Results:
   - Download arrangement as JSON
   - Save configuration for future use

## Technical Details

The optimizer uses advanced algorithms to:
- Calculate optimal box positions
- Consider all possible rotations
- Ensure weight constraints
- Maximize space utilization
- Prevent box overlapping
- Maintain proper weight distribution

## Requirements

All dependencies are handled automatically in the Replit environment. The application uses:
- Python 3.11
- Streamlit
- NumPy
- Matplotlib
- Plotly

The application runs directly in Replit and can be started using the "Run" button.
