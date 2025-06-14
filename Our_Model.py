import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load your cleaned dataset (adjust the path as needed)
df = pd.read_csv("airport_traffic_2016_2024_cleaned.csv")

# ------------------ CLUSTERING ------------------

# Aggregate total flights by airport and year
airport_yearly = df.groupby(['APT_ICAO', 'YEAR'])['FLT_TOT_1'].sum().reset_index()

# Pivot to create time-series-like features per airport
pivoted = airport_yearly.pivot(index='APT_ICAO', columns='YEAR', values='FLT_TOT_1').fillna(0)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivoted)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
pivoted['Cluster'] = kmeans.fit_predict(X_scaled)

# ------------------ CLASSIFICATION ------------------

# Create a binary growth target: High Growth vs. Low Growth
pivoted['Growth'] = pivoted[2024] - pivoted[2016]
pivoted['Growth_Label'] = pd.qcut(pivoted['Growth'], q=2, labels=["Low Growth", "High Growth"])

# Prepare features and labels
X = pivoted.drop(columns=['Cluster', 'Growth', 'Growth_Label'])
y = pivoted['Growth_Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Set plot size
plt.figure(figsize=(14, 8))

# Plot the decision tree
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=clf.classes_, 
          filled=True, 
          rounded=True, 
          fontsize=10)

plt.title("Decision Tree for Airport Growth Classification")
plt.show()


# Optional: save results to CSV
pivoted.to_csv("airport_clusters_and_growth_labels.csv")