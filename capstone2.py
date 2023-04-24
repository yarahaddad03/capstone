import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB

st.write("""
# Students Performance Prediction App

This app predicts undergraduate students pass/fail

Data obtained from the School of Business Admissions Office)
""")

st.sidebar.header('User Input Features')

students_data = pd.read_csv('preprocesseddata.csv')

def user_input_features():
    Age = st.sidebar.slider('Age at admission', 16, 50, 18)
    Gender = st.sidebar.selectbox('Gender', ('0', '1'))
    Country = st.sidebar.selectbox('Country', ('0', '1'))
    Scholarship = st.sidebar.selectbox('Scholarship', ('0', '1'))
    sat = st.sidebar.slider('SAT', 0, 1530, 1100)
    Governorates = st.sidebar.slider('Governorates', 0, 7, 1)
    School = st.sidebar.slider('School', 0, 100, 50)
    Hours = st.sidebar.slider('hours_diff', 0, 100, 50)
    
    data = {'Age at admission': Age,
            'Gender': Gender,
            'Country' : Country,
            'Scholarship' : Scholarship,
            'SAT' : sat,
            'Governorates' : Governorates,
            'School' : School,
            'hours_diff' : Hours}
    features = pd.DataFrame(data, index=[0])
    return features

df= user_input_features()

st.subheader('User Input parameters')
st.write(df)

#Model
nb = GaussianNB()

X = students_data.drop('Performance', axis=1)
y = students_data['Performance']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

# Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the test data using the trained model
y_pred = model.predict(X_test)

prediction = model.predict(df)
prediction_proba= model.predict_proba(df)


st.subheader('Prediction')
st.text('Fail: 0, Pass: 1')
#st.write(students_data['Performance'][prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
