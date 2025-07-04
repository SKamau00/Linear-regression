import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Hypothetical dataset
X = np.array([
    [50, 0.8, 200, 0.3, 5],  # speed limit, road condition (0-1), traffic density, weather (0-1), driver exp (years)
    [70, 0.5, 300, 0.7, 2],
    [30, 0.9, 100, 0.2, 10],
    [60, 0.6, 250, 0.4, 3],
    [80, 0.4, 350, 0.8, 1]
])
y = np.array([3, 7, 2, 5, 9])  # severity scores

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Example prediction for hypothetical values
hypothetical_data = np.array([[65, 0.7, 275, 0.5, 4]])  # speed=65, road=0.7, traffic=275, weather=0.5, exp=4
predicted_severity = model.predict(hypothetical_data)
print(f"Predicted accident severity: {predicted_severity[0]:.2f}")

# Explanation of coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)