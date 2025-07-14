import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# Load dataset
data = pd.read_csv("Faculty_TNA_dataset12.csv")

# Extract TNA columns
tna_columns = list(data.columns)  # assuming file only has 12 TNA cols

# Create High Need label: top 25% of Total_TNA
data['Total_TNA'] = data[tna_columns].sum(axis=1)
threshold = data['Total_TNA'].quantile(0.75)
data['High_Need'] = (data['Total_TNA'] >= threshold).astype(int)

print(f"High Need threshold (75th percentile): {threshold:.2f}")
print(f"Class distribution:\n{data['High_Need'].value_counts()}")

# ==============================
# Train RandomForest classifier
X = data[tna_columns]
y = data['High_Need']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
pickle.dump(rf_model, open("tna_model.pkl", "wb"))
print("✅ Saved RandomForest to tna_model.pkl")

# ==============================
# Train scaler + KMeans clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
pickle.dump((scaler, kmeans), open("tna_cluster_model.pkl", "wb"))
print("✅ Saved scaler + KMeans to tna_cluster_model.pkl")
