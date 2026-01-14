
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



df = pd.read_csv('/kaggle/input/salary/Salary.csv')

\print("--- First 5 Rows of Data ---")
print(df.head())
print("-" * 30)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)

X = diabetes.data
Y = diabetes.target

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=99)

le = LinearRegression()
le.fit(train_x, train_y)

y_pred = le.predict(test_x)

result = pd.DataFrame({'Actual': test_y, 'Predict': y_pred})
print(result.head())

print('coefficient', le.coef_)
print('intercept', le.intercept_)
print("Mean Squared Error: ", mean_squared_error(test_y, y_pred))
print("R2 Score: ", r2_score(test_y, y_pred))


print("\n--- Converting to Classification for Requested Metrics ---")
print("(Threshold: Median value of the actual target)")
threshold = np.median(test_y)
y_test_binary = (test_y > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

cm = confusion_matrix(y_test_binary, y_pred_binary)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Thresholded Regression)')
plt.show()

acc = accuracy_score(y_test_binary, y_pred_binary)
print(f"Accuracy:  {acc:.4f}")

prec = precision_score(y_test_binary, y_pred_binary)
print(f"Precision: {prec:.4f}")

rec = recall_score(y_test_binary, y_pred_binary)
print(f"Recall:    {rec:.4f}")

f1 = f1_score(y_test_binary, y_pred_binary)
print(f"F1 Score:  {f1:.4f}"