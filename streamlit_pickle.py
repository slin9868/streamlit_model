
import streamlit as st 
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import Pipeline

st.header('A Model for AHD')

st.write('Please enter the Age, Sex, and Slope information below.')
 
age = st.number_input('Age')
sex = st.number_input('Sex')
slope = st.number_input('Slope')

X = np.array([[age, sex, slope]])

with open('forestmodel.pkl', 'rb') as f:
    model = pickle.load(f)
    
pred = model.predict(X)

st.write(f'The model predicts {pred[0]}')
