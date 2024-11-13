from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Cargar el dataset y seleccionar las primeras 10 filas
    try:
        df = pd.read_csv('DataSetAndroid/reducido.csv')
        data_preview = df.head(10).to_html(classes='table table-striped')
    except FileNotFoundError:
        data_preview = "<p>Error: el archivo 'reducido.csv' no se encuentra.</p>"

    # Simulación de datos para demostración
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'calss': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    # Paso 1: Transformar la variable de salida a valores numéricos
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]

    # Paso 2: Separar las variables predictoras y la variable de salida
    X_features = X.drop('calss', axis=1)
    y_target = X['calss']

    # Paso 3: Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

    # Paso 4: Crear y entrenar el modelo de regresión
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Paso 5: Realizar predicciones
    y_pred = model.predict(X_test)

    # Paso 6: Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Paso 7: Graficar
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.title('Predicciones vs Real')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.grid(True)

    # Guardar la gráfica en un objeto en memoria y convertirla en base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Renderizar la plantilla con los resultados y la vista previa de datos
    return render_template('index.html', mse=mse, r2=r2, plot_url=plot_url, data_preview=data_preview)

# En Render, configura el punto de entrada como: gunicorn app:app --bind 0.0.0.0:$PORT
# No necesitas ejecutar `app.run()` aquí, ya que Gunicorn manejará esto en producción.

