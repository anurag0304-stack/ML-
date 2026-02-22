#Anurag Singh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
new_point = np.array([[4, 4]])
prediction = knn.predict(new_point)[0]
plt.figure(figsize=(8, 6))
plt.scatter(x[y==0,0], x[y==0,1], c='blue', label='Sunny',s=100)
plt.scatter(x[y==1,0], x[y==1,1], c='red', label='Rainy',s=100)
plt.scatter(
    new_point[0, 0],
    new_point[0, 1],
    marker="*",
    s=200,
    label='New Prediction',
    c='green'
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