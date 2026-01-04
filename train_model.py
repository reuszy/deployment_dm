import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=';')

# 2. Split Data
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training Model
print("Sedang melatih model")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi Sederhana
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Selesai! Akurasi Model: {akurasi:.2f}")

# 5. Simpan Model
with open('model_gagal_jantung.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Model berhasil disimpan sebagai 'model_gagal_jantung.pkl'")