# SolarTimeSeriesForecasting-ITER
Repo containing some of the code scripts and notebooks I implemented during my internship at ITER. They are a collection of scripts and Jupyter notebooks designed for predicting solar energy output using machine learning models. Two Python scripts, `solar_randomForest.py` and `solar_neural.py`, and two Jupyter notebooks, `solarDataAnalysis.ipynb` and `solarDataModelling.ipynb`, are included. The models utilize historical solar energy data to forecast future energy production.

## Description of Components

1. **solar_randomForest.py**:
   - Utilizes RandomForestRegressor from scikit-learn for prediction.
   - Performs data preprocessing and feature engineering.
   - Implements GridSearchCV for hyperparameter tuning.
   - Analyzes data from specific CSV URLs provided in the script.
   - The script adds time-related features and performs data normalization and cleaning.

2. **solar_neural.py**:
   - Implements NeuralProphet, a neural network-based forecasting tool.
   - Similar to `solar_randomForest.py`, it preprocesses data and adds time-related features.
   - Includes custom functions for error evaluation and future prediction validation.
   - Trains the model and predicts future values based on historical data.

3. **solarDataAnalysis.ipynb**:
   - A Jupyter notebook dedicated to the analysis of solar energy data.
   - Includes exploratory data analysis, visualization, and preliminary data processing steps over the given datasets

4. **solarDataModelling.ipynb**:
   - Focuses on building and evaluating machine learning models.
   - This notebook includes more detailed modeling steps, comparisons of different models, and hyper-parameter tuning.

## Setup and Installation

1. **Prerequisites**:
   - Python 3.x
   - Pandas, Numpy
   - scikit-learn
   - NeuralProphet

2. **Installation**:
   - Install required libraries using pip:
     ```bash
     pip install pandas numpy scikit-learn neuralprophet
     ```

3. **Running the Scripts**:
   - Download the scripts and notebooks from the repository.
   - Run the Python scripts directly in a Python environment:
     ```bash
     python solar_randomForest.py
     python solar_neural.py
     ```
   - Open the Jupyter notebooks using Jupyter Lab or Jupyter Notebook:
     ```bash
     jupyter lab solarDataAnalysis.ipynb
     jupyter lab solarDataModelling.ipynb
     ```

## Usage

1. **Data Preparation**:
   - The scripts and notebooks are designed to work with specific CSV files, accessible via predefined URLs.
   - Ensure the data format aligns with the expected structure in the scripts.

2. **Model Training and Evaluation**:
   - `solar_randomForest.py` and `solar_neural.py` contain code for training machine learning models.
   - Models can be trained, evaluated, and compared using the provided notebooks.

3. **Predictions**:
   - Once the models are trained, use them to make predictions on future data.
   - Customize the scripts for your specific use case and data.

## Contributing

- Contributions to this repository are welcome. 
- Please adhere to standard coding practices and document any changes or improvements made.

## License

- This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- For questions or assistance, please open an issue in the repository, and a maintainer will respond accordingly.

---

This README provides a general overview and setup instructions for the repository. It is advised to refer to individual scripts and notebooks for more detailed documentation and comments on specific sections of the code.
