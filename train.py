import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df=pd.read_csv('C:\\Users\\Bhumi\\OneDrive\\Desktop\\ML_Project\\Woman_Safety_Dataset_Management.csv')

print(df.head())

print(df.columns)

print(df.dtypes)

print(df.describe())


# Numerical columns

cols=df[['latitude','longitude','crime_count',
         'lighting_score','police_station_distance_km',
         'crowd_density','safety_score']]

for i in cols:
    plt.figure(figsize=(10,5))
    plt.boxplot(df[i])
    plt.title(i)
    plt.show()


# City vs Crime Count

plt.figure(figsize=(15,5))
plt.bar(df['city'],df['crime_count'])
plt.title('City vs Crime Count')
plt.xlabel('City')
plt.ylabel('Crime Count')
plt.show()


# Top 5 Crime Count

top_5_crime_count=df.nlargest(5,"crime_count")

plt.figure(figsize=(10,5))
plt.bar(top_5_crime_count['area'],
        top_5_crime_count['crime_count'])
plt.title('Area vs Crime Count')
plt.xlabel('Area')
plt.ylabel('Crime Count')
plt.show()


# Crime Type

plt.figure(figsize=(10,5))
plt.pie(
    df['crime_type'].value_counts(),
    labels=df['crime_type'].unique(),
    autopct='%1.1f%%'
)
plt.title('Crime Type')
plt.show()


# Time of Day

plt.figure(figsize=(10,5))
plt.pie(
    df['time_of_day'].value_counts(),
    labels=df['time_of_day'].unique(),
    autopct='%1.1f%%'
)
plt.title('Time of Day')
plt.show()


# Weather Condition

plt.figure(figsize=(10,5))
plt.pie(
    df['weather_condition'].value_counts(),
    labels=df['weather_condition'].unique(),
    autopct='%1.1f%%'
)
plt.title('Weather Condition')
plt.show()


# Risk Level

plt.figure(figsize=(10,5))
plt.pie(
    df['risk_level'].value_counts(),
    labels=df['risk_level'].unique(),
    autopct='%1.1f%%'
)
plt.title('Risk Level')
plt.show()


print('City :',df['city'].unique())

print('Area : ',df['area'].unique())

print('Crime Type :',df['crime_type'].unique())

print('Time of day :',df['time_of_day'].unique())

print('Weather Condition :',df['weather_condition'].unique())

print("Risk Level : ",df['risk_level'].unique())


# Reload original dataframe

df=pd.read_csv('Woman_Safety_Dataset_Management.csv')


# Encoding columns

columns_to_encode=[
    'city',
    'area',
    'crime_type',
    'time_of_day',
    'weather_condition'
]


df=pd.get_dummies(
    df,
    columns=columns_to_encode,
    dtype=int
)


print(df.head())


# Remove incident id

df=df.drop(
    columns=['incident_id'],
    errors='ignore'
)


# Convert timestamp

df['incident_timestamp']=pd.to_datetime(
    df['incident_timestamp'],
    format='%d-%m-%Y %H:%M',
    errors='coerce'
)


# Extract hour

df['hour']=df['incident_timestamp'].dt.hour


# Extract day of week

df['day_of_week']=df['incident_timestamp'].dt.dayofweek


# Extract month

df['month']=df['incident_timestamp'].dt.month


# Weekend

df['is_weekend']=df[
    'incident_timestamp'
].dt.dayofweek.isin([5,6]).astype(int)


# AM / PM

df['is_am']=np.where(
    df['hour']<12,
    1,
    0
)

df['is_pm']=np.where(
    df['hour']>=12,
    1,
    0
)


# Remove timestamp

df=df.drop(
    columns=['incident_timestamp'],
    errors='ignore'
)


print(df.head())


# X and y

X=df.drop(
    columns=['risk_level'],
    errors='ignore'
)

y=df['risk_level']


# Save feature columns

feature_columns=X.columns.tolist()

joblib.dump(
    feature_columns,
    'feature_columns.joblib'
)

print("Feature Columns Saved Successfully!")


# Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Standard Scaling

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# Save scaler

joblib.dump(
    scaler,
    'scaler.joblib'
)

print("Scaler Saved Successfully!")


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=4,
    min_samples_leaf=3,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)


# Prediction

dt_y_pred=dt_model.predict(
    X_test
)


# Evaluation

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


print(
    "\n========================== Decision Tree Matrix ==========================\n"
)

print(
    "Accuracy :",
    accuracy_score(y_test,dt_y_pred)
)

print(
    "Precision :",
    precision_score(
        y_test,
        dt_y_pred,
        average='weighted'
    )
)

print(
    "Recall :",
    recall_score(
        y_test,
        dt_y_pred,
        average='weighted'
    )
)

print(
    "F1 Score :",
    f1_score(
        y_test,
        dt_y_pred,
        average='weighted'
    )
)

print(
    "Confusion Matrix :\n",
    confusion_matrix(y_test,dt_y_pred)
)


# Training accuracy

train_pred_dt=dt_model.predict(X_train)

print(
    "Training Accuracy:",
    accuracy_score(y_train,train_pred_dt)
)


# Testing accuracy

test_pred_dt=dt_model.predict(X_test)

print(
    "Testing Accuracy:",
    accuracy_score(y_test,test_pred_dt)
)


# Confusion Matrix

conf_matrix=confusion_matrix(
    y_test,
    dt_y_pred
)

class_labels=np.unique(y_test)

plt.figure(figsize=(8,6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.title(
    'Confusion Matrix for Decision Tree Model'
)

plt.show()


# Save Model

joblib.dump(
    dt_model,
    'dt_model.joblib'
)

print("Model Saved Successfully!")

print("\nAll files saved successfully!")

print("1. dt_model.joblib")
print("2. scaler.joblib")
print("3. feature_columns.joblib")