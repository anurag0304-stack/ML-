#Anurag Singh
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x = np.array([[30, 75], [25, 80], [27, 60], [31, 65], [23, 85], [28, 65]])
y = np.array([0, 1, 0, 1, 0, 1])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
new_point = np.array([[26, 78]])
prediction = knn.predict(new_point)[0]
plt.figure(figsize=(6,8))
plt.scatter(x[y==0,0], x[y==0,1], label='Sunny',s=100)
plt.scatter(x[y==1,0], x[y==1,1], label='Rainy',s=100)
plt.scatter(
    new_point[0, 0],
    new_point[0, 1],
    marker="*",
    s=300,
    label='New Prediction',
    c='red'
    )
plt.xlabel('temperture(°C)')
plt.ylabel('humidity(%)')
plt.title('K-Nearest Neighbors Classification')
plt.legend()
plt.grid(alpha=0.3 )
plt.show()
if prediction == 0:
    print("The predicted weather condition for the new point is: Sunny")
else:

    print("The predicted weather condition for the new point is: Rainy")


