from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

city_le = LabelEncoder()
condition_le = LabelEncoder()

# Define a list of city names
city_list = [
    'Shoreline', 'Seattle', 'Kent', 'Bellevue', 'Redmond',
    'Maple Valley', 'North Bend', 'Lake Forest Park', 'Sammamish',
    'Auburn', 'Des Moines', 'Bothell', 'Federal Way', 'Kirkland',
    'Issaquah', 'Woodinville', 'Normandy Park', 'Fall City', 'Renton',
    'Carnation', 'Snoqualmie', 'Duvall', 'Burien', 'Covington',
    'Inglewood-Finn Hill', 'Kenmore', 'Newcastle', 'Mercer Island',
    'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Algona', 'Skykomish',
    'Tukwila', 'Vashon', 'Yarrow Point', 'SeaTac', 'Medina',
    'Enumclaw', 'Snoqualmie Pass', 'Pacific', 'Beaux Arts Village',
    'Preston', 'Milton'
]

conditions = ['Terrible', 'Poor', 'Acceptable', 'Excellent', 'Outstanding']

# Fit the LabelEncoder with the city list and conditions
city_le.fit(city_list)
condition_le.fit(conditions)

# Load the trained Random Forest model
model = pickle.load(open('home.pkl', 'rb'))

appp = Flask(__name__)

@appp.route('/')
def home():
    return render_template('housepriceprediction.html', city_list=city_list, conditions=conditions)

@appp.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get user input from the HTML form
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    Total_SQFT = int(request.form['Total_SQFT'])
    floors = int(request.form['floors'])
    condition = str(request.form['condition'])
    city = str(request.form['city'])

    city_encoded = city_le.transform([city])
    condition_encoded = condition_le.transform([condition])

    # Create a DataFrame with the user input
    data = {'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms], 'Total_SQFT': [Total_SQFT],
            'Floors': [floors], 'Condition': condition_encoded, 'City': city_encoded}

    input_df = pd.DataFrame(data)

    # Make a prediction using the loaded model
    predicted_price = model.predict(input_df)[0]
    formatted_predicted_price = "{:.2f}".format(predicted_price)
    return render_template('housepriceprediction.html', predicted_price=formatted_predicted_price, city_list=city_list, conditions=conditions)

if __name__ == '__main__':
    appp.run(debug=True)