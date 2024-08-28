# Renewable Energy Prediction App

## Overview

The Renewable Energy Prediction App is a Streamlit application designed to predict solar power generation and consumption. The app uses machine learning models like HistGradientBoostingRegressor and RandomForestRegressor to forecast energy production and consumption based on historical data.

## Features

- **Generation Prediction**: Predicts solar power generation based on time-based features and historical power data.
- **Consumption Prediction**: Forecasts solar power consumption using factors such as temperature and employee count.
- **Visualization**: Displays graphs of actual vs. predicted values for both generation and consumption.

## Project Structure

- **main.py**: The main Streamlit application script.
- **requirements.txt**: A file listing the Python dependencies required to run the project.

## Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/Renewable-Energy-Prediction-App.git
cd Renewable-Energy-Prediction-App
```
## Install the Requirements

Install the necessary Python packages using pip:

```sh
pip install -r requirements.txt
```
## Usage

### 1. Run the Application

Start the Streamlit application by running:

```sh
streamlit run main.py
```
## 2. Select Prediction Type

Use the sidebar to select either "Generation" or "Consumption" prediction.  
Upload your CSV file with the relevant data when prompted.

## 3. Predict Solar Power Generation

- For generation prediction, the model will predict solar power based on features like hour, day, and lagged power values.
- The app will display a graph comparing the actual vs. predicted power generation and output the results in a table.

## 4. Predict Solar Power Consumption

- For consumption prediction, the model forecasts power consumption using temperature, employee count, and past consumption data.
- The app will display a graph comparing the actual vs. predicted power consumption and output the results in a table.

## Explore and Modify

You can customize the model, data preprocessing, or the Streamlit app to suit your needs:

- **Data Preprocessing**: Modify the feature engineering process or add new features.
- **Model Training**: Experiment with different models or hyperparameters to improve accuracy.
- **App Features**: Add new features or enhance the user interface in `main.py`.

## Future Work

- **Advanced Models**: Explore more sophisticated models for energy prediction, such as deep learning approaches.
- **Hyperparameter Tuning**: Experiment with hyperparameter tuning to enhance model performance.
- **Data Augmentation**: Incorporate additional data or features to improve prediction accuracy.

## Useful Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Acknowledgements

- **Streamlit**: For providing an easy-to-use framework for building interactive applications.
- **Scikit-learn**: For the machine learning algorithms used in the project.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
