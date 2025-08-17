import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Saving model to disk
pickle.dump(model, open('model_pmb.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model_pmb.pkl','rb'))

sample_data = np.array([[0.038075906, 0.05068012, 0.061696206, 0.021872354, -0.0442235,
                         -0.03482076, -0.04340085, -0.00259226, 0.01990749, -0.017646125]])
prediction = model.predict(sample_data)
print(f"Predicted disease progression: {prediction[0]:.2f}")



