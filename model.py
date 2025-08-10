import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import gradio as gr

# Load the dataset
data = pd.read_csv('D:\Other Files\Games\BHP project\Bengaluru_House_Data.csv')

# Initial data exploration



data = data.drop(['area_type', 'availability', 'balcony', 'society'], axis=1)
data
data.isna().sum()
data = data.dropna()  # Added parentheses to call the dropna method
data.isna().sum()
data.shape
data['size'].unique()