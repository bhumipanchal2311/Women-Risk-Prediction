#   Library Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#   Load the downloaded college placement dataset into a Pandas DataFrame
df=pd.DataFrame(pd.read_csv("D:\\PYTHON\\ML_Project\\CollegePlacement.csv"))

print("\n\t ---------- The first 5 rows of the dataset to verify the structure ----------\n ")
print(df.head())

print("\n\t ---------- The last 5 rows of the dataset to check the end of the data ----------\n ")
print(df.tail())

print("\n\t ---------- A random sample of 5 rows to observe data variation and distributions ----------\n ")
print(df.sample(5))

print("\n\t ---------- Check the shape of the dataset ----------\n ")  
print("Number of Rows :",df.shape[0])
print("Number of Columns :",df.shape[1])

print("\n\t ---------- Check the data type of each column to identify numerical and categorical features ----------\n ")
print(df.dtypes)

print("\n\t ---------- Separate columns into quantitative (numerical) and qualitative (categorical) lists ----------\n ")
quantitative_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
qualitative_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Quantitative (Numerical) Columns:\n", quantitative_cols)
print("\nQualitative (Categorical) Columns:\n", qualitative_cols)

print("\n\t ---------- A concise summary of the DataFrame, including the number of non-null entries and data types ----------\n ")
print(df.info())

#   Total Observations: 10,000 student records (indexed 0 to 9,999).
#   Total Features: 10 columns.
#   Data Completeness: 100% Complete. There are absolutely no missing (null) values in any of the columns.
#   Memory Footprint: Very lightweight (~781.4 KB), making it lightning-fast to train models on.


print("\n\t ---------- Count the number of missing (null) values in each column ----------\n")
print(df.isnull().sum())

#   Null Value :  Running df.isnull().sum() confirms that our dataset has 0 missing values across all 10 columns.


print("\n\t ---------- Check for any identical duplicate rows in the dataset ----------\n")
print("Number of Duplicate values : ",df.duplicated().sum())

#   Duplicate Record Check: Running df.duplicated().sum() confirms that there are 0 duplicate rows in our dataset.

print("\n\t ---------- View statistical summary of numerical features ----------\n")
print(df.describe())

#   CRITICAL ERROR: The CGPA column shows an impossible maximum value of 10.46. 

print("\n\t ---------- Check for any outliers in the CGPA column ----------\n")
print("Number of Outliers in CGPA column : ",df[df['CGPA']>10].shape[0])

#   Calculate the median from valid CGPA entries first
median_cgpa= df.loc[df['CGPA'] <= 10.00, 'CGPA'].median()
print("Median CGPA (valid values only):",median_cgpa)

print("Replacing outliers with median CGPA value")
df.loc[df['CGPA'] > 10.00, 'CGPA'] = median_cgpa

print("\n\t ---------- Verify that the outliers have been replaced ----------\n")
print(df['CGPA'].describe())

#   Generate a boxplot to inspect the distribution and look for extreme outliers in IQ
sns.boxplot(x=df['IQ'])
plt.title('Boxplot of IQ')
plt.xlabel('IQ')
plt.show()

# Calculate the 25th (Q1) and 75th (Q3) percentiles for the IQ column
q1 = df["IQ"].quantile(0.25)
q3 = df["IQ"].quantile(0.75)

# the Interquartile Range (IQR)
iqr_IQ = q3 - q1

# Display the calculated IQR value
print("IQR:", iqr_IQ)

# Calculate boundaries
lower_bound = q1 - 1.5 * iqr_IQ
upper_bound = q3 + 1.5 * iqr_IQ

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Remove outliers
df_cleaning= df[
    (df["IQ"] >= lower_bound) & 
    (df["IQ"] <= upper_bound)]

print("Original rows:", len(df))
print("After removing outliers:", len(df_cleaning))

# Plot the cleaned IQ column to verify that outliers outside the fences have been removed
sns.boxplot(x=df_cleaning['IQ'])
plt.title('Boxplot of IQ')
plt.xlabel('IQ')
plt.show()

#    Create the list of columns
print("\n\t ---------- Create a list of features for boxplot visualization ----------\n")
features = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
           'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

plt.figure(figsize=(18, 10))

# You must keep 'enumerate' so Python can count the chart positions (1, 2, 3...)
for i, col in enumerate(features, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=df_cleaning[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()

print("\n\t ---------- Create a list of features for boxplot visualization ----------\n")
features = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',
           'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

plt.figure(figsize=(18, 10))

for i, col in enumerate(features, 1):
    plt.subplot(2, 4, i)
    sns.histplot(x=df_cleaning[col])
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()

#   Pie chart to visualize the distribution of students who were placed vs not placed

print("\n\t ---------- Pie chart to visualize the distribution of students who were placed vs not placed ----------\n")
plt.pie(df_cleaning['Placement'].value_counts(), labels=['Not Placed', 'Placed'], autopct='%1.1f%%', colors=['#FF5722','#4CAF50'])
plt.title('Distribution of Placed vs Not Placed Students')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#   pie chart to visualize the distribution of students with internship vs without internship 

print("\n\t ---------- Pie chart to visualize the distribution of students with internship vs without internship ----------\n")
plt.pie(df_cleaning['Internship_Experience'].value_counts(), labels=['No Internship', 'Internship'], autopct='%1.1f%%', colors=['#FF5722','#4CAF50'])
plt.title('Distribution of Students with Internship vs without Internship')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#  Convert categorical variable 'Internship_Experience' into numerical format using one-hot encoding
df_cleaning["Internship_Experience"] = df_cleaning["Internship_Experience"].map({"Yes": 1,"No": 0})
df_cleaning["Placement"] = df_cleaning["Placement"].map({"Yes": 1,"No": 0})

print("\n\t ---------- Verify the conversion of categorical variables to numerical format ----------\n")
print(df_cleaning.head())


#   FEATURE ENGINERING

df_cleaning['profile_score']=df_cleaning['Academic_Performance'] + df_cleaning['Extra_Curricular_Score'] + df_cleaning['Communication_Skills'] + df_cleaning['Projects_Completed']
print("\n\t ---------- Verify the creation of the new feature 'profile_score' ----------\n")
print(df_cleaning[['profile_score']].head())

df_cleaning['IQ_VS_CGPA'] = df_cleaning['IQ'] * df_cleaning['CGPA']
print("\n\t ---------- Verify the creation of the new feature 'IQ_VS_CGPA' ----------\n")
print(df_cleaning[['IQ_VS_CGPA']].head())

df_cleaning['Interview_Readiness'] = df_cleaning['profile_score'] + df_cleaning['IQ_VS_CGPA']
print("\n\t ---------- Verify the creation of the new feature 'Interview_Readiness' ----------\n")
print(df_cleaning[['Interview_Readiness']].head())

df_cleaning['Total_Merit']=df_cleaning['profile_score'] + df_cleaning['IQ_VS_CGPA'] + df_cleaning['Interview_Readiness']
print("\n\t ---------- Verify the creation of the new feature 'Total_Merit' ----------\n")
print(df_cleaning[['Total_Merit']].head())

# correlation matrix to visualize the relationships between features and the target variable 'Placement'
correlation_matrix=df_cleaning.corr(numeric_only=True)

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

#   ML

#   Split the dataset into features (X) and target variable (y)
x=df_cleaning.drop(columns=['Placement','College_ID',])
y=df_cleaning['Placement']

#   Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#   Standardize the features using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

joblib.dump(scaler, "scaler.pkl")

print("\n\t ---------- Logistic Regression ----------\n")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
logistic_model = model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n\t ---------- Model Evaluation Metrics ----------\n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", cm)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
knn_cv_scores = cross_val_score(knn_model, x, y, cv=5)

print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN F1 Score:", knn_f1)
print("\nKNN Confusion Matrix:\n", knn_cm)
print("KNN Cross-Validation Scores:", knn_cv_scores)

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_predictions = svm_model.predict(x_test)
svm_cv_scores = cross_val_score(svm_model, x, y, cv=5)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)    
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_cm = confusion_matrix(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1 Score:", svm_f1)
print("\nSVM Confusion Matrix:\n", svm_cm)
print("SVM Cross-Validation Scores:", svm_cv_scores)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
dt_cv_scores = cross_val_score(dt_model, x, y, cv=5)

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)
dt_cm = confusion_matrix(y_test, dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Precision:", dt_precision) 
print("Decision Tree Recall:", dt_recall)
print("Decision Tree F1 Score:", dt_f1)
print("\nDecision Tree Confusion Matrix:\n", dt_cm)
print("Decision Tree Cross-Validation Scores:", dt_cv_scores)

dt_train_accuracy = accuracy_score(y_train, dt_model.predict(x_train))
print("Decision Tree Training Accuracy:", dt_train_accuracy)
dt_test_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Testing Accuracy:", dt_test_accuracy)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
rf_cv_scores = cross_val_score(rf_model, x, y, cv=5)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1 Score:", rf_f1)
print("\nRandom Forest Confusion Matrix:\n", rf_cm)
print("Random Forest Cross-Validation Scores:", rf_cv_scores)

rf_train_accuracy = accuracy_score(y_train, rf_model.predict(x_train))
print("Random Forest Training Accuracy:", rf_train_accuracy)
rf_test_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Testing Accuracy:", rf_test_accuracy)


joblib.dump(dt_model, 'D:\\PYTHON\\ML_Project\\dt_model.pkl')
print("\n\t ---------- Decision Tree model saved as 'dt_model.pkl' ----------\n")   