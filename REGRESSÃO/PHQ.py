import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = "REGRESSÃO/PHQ-9_Dataset_5th Edition.csv"
dataset = pd.read_csv(path)

dataset.columns = dataset.columns.str.strip()

mapa_phq = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

phq_cols = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself—or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

for col in phq_cols:
    dataset[col] = dataset[col].str.strip().map(mapa_phq)

if 'PHQ_Severity' in dataset.columns:
    dataset = dataset.drop(columns=['PHQ_Severity'])

categorical_cols = ['Gender', 'Sleep Quality', 'Study Pressure', 'Financial Pressure']
encoder = LabelEncoder()
for col in categorical_cols:
    dataset[col] = encoder.fit_transform(dataset[col])

dataset['Mood'] = dataset[[
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless"
]].mean(axis=1)
dataset['Sleep'] = dataset[["Trouble falling or staying asleep, or sleeping too much"]].mean(axis=1)
dataset['Energy'] = dataset[["Feeling tired or having little energy"]].mean(axis=1)
dataset['Appetite'] = dataset[["Poor appetite or overeating"]].mean(axis=1)
dataset['SelfEsteem'] = dataset[["Feeling bad about yourself—or that you are a failure or have let yourself or your family down"]].mean(axis=1)
dataset['Concentration'] = dataset[["Trouble concentrating on things, such as reading the newspaper or watching television"]].mean(axis=1)
dataset['Psychomotor'] = dataset[["Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual"]].mean(axis=1)
dataset['Suicidal'] = dataset[["Thoughts that you would be better off dead or of hurting yourself in some way"]].mean(axis=1)

X = dataset.drop(columns=['PHQ_Total'] + phq_cols)
y = dataset['PHQ_Total']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': list(range(1, 41)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'p': [1, 2]
}

knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

melhor_knn = grid.best_estimator_
y_pred = melhor_knn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Melhor combinação: {grid.best_params_}")
print("\n=== KNN Regressor + PCA + Feature Engineering ===")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
print(f"Número de componentes PCA: {pca.n_components_}")
