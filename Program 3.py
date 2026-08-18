import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

data = load_breast_cancer()
X= pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target)
print("Dataset shape:",X.shape)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


bag_model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bag_model.fit(X_train,y_train)
y_pred_bag=bag_model.predict(X_test)

print("=== Bagging Results ===")
print("Accuracy:",accuracy_score(y_test,y_pred_bag))
print("Classification Report:\n",classification_report(y_test,y_pred_bag))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_bag))


boost_model=AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42)

boost_model.fit(X_train,y_train)
y_pred_boost=boost_model.predict(X_test)

print("\n=== AdaBoost Results ===")
print("Accuracy:",accuracy_score(y_test,y_pred_boost))
print("Classification Report:\n",classification_report(y_test,y_pred_boost))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_boost))

# print(data.feature_names)
# print(data.target_names)
# print(data.target)
# print(len(data.target))
