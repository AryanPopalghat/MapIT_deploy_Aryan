import numpy as np
import pickle
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load your saved model (replace 'your_model.h5' with the correct filename)
pred_model = load_model('your_model.h5')

# Load the scaler (replace 'scaler.pkl' with the correct filename)
with open('model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create a FastAPI app
app = FastAPI()

# Define a Pydantic model for input data
class SalesInfo(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_7: float
    lag_8: float
    lag_9: float
    lag_10: float

# API endpoint to predict future sales
@app.post('/predict')
def predict_sales(data: SalesInfo):
    # Convert input data to a numpy array
    features = np.array([[
        data.lag_1, data.lag_2, data.lag_3, data.lag_4, data.lag_5,
        data.lag_6, data.lag_7, data.lag_8, data.lag_9, data.lag_10
    ]])

    # Predict future sales using the loaded model
    future_sales_predictions = pred_model.predict(features)[0, 0]
    future_sales_predictions = float(future_sales_predictions)

    # Inverse transform the prediction using the loaded scaler
    predicted_value = np.array([[future_sales_predictions]])
    future_sales_predictions = scaler.inverse_transform(predicted_value)
    future_sales_predictions = future_sales_predictions[0, 0]

    return {'future_sales_predictions': future_sales_predictions}

if __name__ == '__main__':
    # Run the API with uvicorn
    # Will run on http://127.0.0.1:8000
    uvicorn.run(app, host='127.0.0.1', port=8000)


# --------------------------------------------------------------------- #

# Command to run this python script -->
# uvicorn app:app --reload


# Testcase -->
# {
#   "lag_1": 0.47923356,
#   "lag_2": 0.46374405,
#   "lag_3": 0.48067927,
#   "lag_4": 0.74060375,
#   "lag_5": 0.84755089,
#   "lag_6": 0.77915399,
#   "lag_7": 0.56764024,
#   "lag_8": 0.48112283,
#   "lag_9": 0.4561315,
#   "lag_10": 0.51765537
# }
