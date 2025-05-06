from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Prepare the data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Features (2D)
y = np.array([2, 4, 5, 4, 5])                # Labels (1D)

# Step 2: Create and train the model
model = LinearRegression()
model.fit(x, y)

# Step 3: Get slope (coefficient) and intercept
m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation: y = {m:.2f}x + {c:.2f}")

# Step 4: Make prediction
x_test = np.array([[6]])
y_pred = model.predict(x_test)
print(f"Prediction for x = 6: y = {y_pred[0]:.2f}")

print("hello")