#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



"""Data Preprocessing"""

#Loading the Dataset
try:
    Data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please make sure its in the correct directory.")

#Data Cleaning and Transformation
print(Data.info())
print("\nMissing values:\n", Data.isnull().sum())

#Dropping unwanted columns
Data = Data.drop('Person ID', axis=1)

#Splitting the column 'Blood Pressure'
Data[['SystolicBP', 'DiastolicBP']] = Data['Blood Pressure'].str.split('/', expand=True).astype(int)
Data = Data.drop('Blood Pressure', axis=1)

#Handle 'BMI Category' for consistency
#The datasdet has similar entries like 'Normal' and 'Normal Weight'
Data['BMI Category'] = Data['BMI Category'].replace('Normal Weight', 'Normal')

#Checking & Handling missing values
print("\nMissing values in the col 'Sleep Disorder' before fix:\n", Data['Sleep Disorder'].isnull().sum())
Data.dropna(subset=["Sleep Disorder"], inplace=True)
print("After fixing:\n", Data['Sleep Disorder'].isnull().sum())

#Cleaned Data
print("\nDataFrame after Preprocessing")
print(Data.head(10))
print(Data.describe())
print("\nName of the Columns in the Dataset after Preprocessing:")
print(Data.columns)

"""Exploratory Data Analysis(EDA)"""

#Plot style
sns.set_style('whitegrid')

#Distribution of sleep Duration
plt.figure(figsize=(10, 6))
sns.histplot(x= 'Sleep Duration', y='Occupation', data=Data)
plt.title('Distribution of Sleep Duration')
plt.savefig('Distribution of Sleep Duration.png')
plt.show()

#Box plot of Sleep Quality by Occupation
plt.figure(figsize=(11, 7))
sns.boxplot(x='Occupation', y='Quality of Sleep', data=Data)
plt.title('Quality of Sleep by Occupation')
plt.xticks(rotation=30, ha='right')
plt.savefig('Quality of sleep by Occupation.png')
plt.show()

#Correlation Heatmap
plt.figure(figsize=(12, 10))
numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                  'Stress Level', 'Heart Rate', 'Daily Steps', 'SystolicBP', 'DiastolicBP']
sns.heatmap(Data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('Correlation Heatmap of Numerical Features.png')
plt.show()

"""Classification Model"""

#Predicting Sleep Disorder

X = Data.drop('Sleep Disorder', axis=1)
y = Data['Sleep Disorder']

#Train&Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print("\nNumerical features:")
print(numerical_features)
print("\nCategorical features:")
print(categorical_features)
print("\nSleep Disorder Count:")
print(pd.crosstab(index=Data['Sleep Disorder'], columns='Count'))

#Create a preprocessing pipeline
   #Onehotencoder for categorical features
   #StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_features),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)
    ])

#Creating full pipeline including the preprocessor and the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

#Train the model
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model's performance
print("\n-----Model Evaluation-----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Fit and Transform Data
X_scaled = preprocessor.fit_transform(Data)
print("\nData scaled successfully!")

#fit preprocessor to the data
preprocessor.fit(Data)

#Get the feature names to label the axis
feature_names = preprocessor.get_feature_names_out()

# Visualize using first two features
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], color='green', marker='*')
plt.title("KMeans Clustering on Sleep Health Dataset(first two features)")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.colorbar()
plt.grid(True)
plt.savefig('KMeans Clustering on Sleep Health.png')
plt.show()
plt.close()

#Finding optimal num.of clusters [Elbow method]
inertia = []
range_of_k = range(1, 11)
for k in range_of_k:
    kmeans = KMeans(n_clusters=k, random_state= 42, n_init= 18)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

#Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range_of_k, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range_of_k)
plt.savefig('elbow_method.png')
plt.show()
plt.close()
print("\nElbow Method plot saved as elbow_method.png")

#Based on the plot, let's assume k=3 is the optimal num.of clusters
optimal_k = 3

#Applying K-Means with Optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
Data['Cluster'] = kmeans.fit_predict(X_scaled)
print(f"\nK-Means clustering completed with k={optimal_k}. A 'Cluster column has been added to the DataFrame.")

#Analyzing the characteristics of the clusters
print("\nCluster sizes:")
print(Data['Cluster'].value_counts())

cluster_analysis = Data.groupby('Cluster')[numerical_features].mean
print("\nMean values of numerical features per cluster:")
print(cluster_analysis)

#Visualizing the Clusters using scatter plot with two main features
plt.figure(figsize=(11, 7))
sns.scatterplot(x='Age', y='Sleep Duration', hue='Cluster', data=Data, palette='viridis', style='Gender', s=100)
plt.title('K-Means Clustering: Age vs. Sleep Duration')
plt.xlabel('Age')
plt.ylabel('Sleep Duration (hrs)')
plt.legend(title='Cluster & Gender')
plt.savefig('kmeans_scatter_plot.png')
plt.show()
plt.close()
print("\nK-Means scatter plot saved as kmeans_scatter_plot.png")

