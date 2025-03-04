import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Generar datos ficticios
data = {
    "edad": np.random.randint(18, 70, 500),
    "ingreso_mensual": np.random.randint(5000, 100000, 500),
    "historial_crediticio": np.random.choice(["Bueno", "Regular", "Malo"], 500),
    "deuda_actual": np.random.randint(0, 50000, 500),
    "aprobado": np.random.choice(["Si", "No"], 500)
}

df = pd.DataFrame(data)

# Codificar variables categóricas correctamente
le = LabelEncoder()
le.fit(["Bueno", "Regular", "Malo"])  # Asegurar que todas las clases están definidas
df["historial_crediticio"] = le.transform(df["historial_crediticio"])
df["aprobado"] = LabelEncoder().fit_transform(df["aprobado"])

# Separar variables predictoras y objetivo
X = df.drop(columns=["aprobado"])
y = df["aprobado"]

# Normalizar datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Función para predecir si un cliente obtiene crédito
def evaluar_cliente(edad, ingreso_mensual, historial_crediticio, deuda_actual):
    if historial_crediticio not in le.classes_:
        raise ValueError(f"Valor desconocido: {historial_crediticio}. Valores esperados: {le.classes_}")
    historial_codificado = le.transform([historial_crediticio])[0]
    entrada = np.array([[edad, ingreso_mensual, historial_codificado, deuda_actual]])
    entrada = scaler.transform(entrada)
    resultado = model.predict(entrada)
    return "Aprobado" if resultado[0] == 1 else "Rechazado"

# Ejemplo de predicción
print(evaluar_cliente(30, 30000, "Bueno", 10000))
