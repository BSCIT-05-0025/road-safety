# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data for the example
np.random.seed(0)
n_samples = 100
road_conditions = np.random.choice([1, 2, 3], size=n_samples)
traffic_density = np.random.randint(100, 500, size=n_samples)
speed_limits = np.random.randint(30, 70, size=n_samples)
accidents = 10 + 2 * road_conditions + 0.5 * traffic_density - 0.8 * speed_limits + np.random.normal(0, 5, size=n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Road_Conditions': road_conditions,
    'Traffic_Density': traffic_density,
    'Speed_Limits': speed_limits,
    'Accidents': accidents
})

# Specify independent (X) and dependent (y) variables
X = data[['Road_Conditions', 'Traffic_Density', 'Speed_Limits']]
y = data['Accidents']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Accidents')
plt.ylabel('Predicted Accidents')
plt.title('Actual vs. Predicted Accidents')
plt.show()
