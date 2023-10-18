# RK
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

try:
    data = pd.read_csv("D:/blood1.csv")
except FileNotFoundError:
    print("Файл не найден.")
    exit(1)

X = data.drop("BloodGlucoseLevel", axis=1)
y = data["BloodGlucoseLevel"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = SVC(kernel='linear')
start_time = time.time()
clf.fit(X_train_scaled, y_train)
end_time = time.time()
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Точность SVM:", accuracy)
print("Время обучения модели:", end_time - start_time, "секунд")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='navy', label='Predicted', marker='o')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal', linestyle='--')
plt.title('Результаты SVM для BloodGlucoseLevel')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.legend()
plt.grid(True)
plt.text(100, 200, f'Коэффициент корреляции: {round(accuracy, 2)}', fontsize=12, color='black')
plt.show()
