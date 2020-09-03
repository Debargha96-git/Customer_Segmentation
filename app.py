# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

#Load the clutsering model
filename='customer clustering.pkl'
y_kmeans=pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        Age = int(request.form['Age'])
        Annual_income = int(request.form['Annual income'])
        Spending_score  = int(request.form['Spending score'])
        data = np.array([[Age, Annual_income, Spending_score]])
        my_prediction = y_kmeans
        return render_template('result.html', prediction=my_prediction)




if __name__ == '__main__':
     app.run(host='127.0.0.1', port=8040, debug=True)
    #app.run(debug=True)