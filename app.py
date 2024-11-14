from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import os
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Cargar el dataset y seleccionar las primeras 10 filas
    if not os.path.isfile('DataSetAndroid/reducido.csv'):
        return "Error: No se encuentra el archivo 'DataSetAndroid/reducido.csv'"

    df = pd.read_csv('DataSetAndroid/reducido.csv')
    data_preview = df.head(10).to_html(classes='table table-striped')

    # Simulación de datos para demostración
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'calss': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    # Transformar la variable de salida a valores numéricos
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]

    # Separar las variables predictoras y la variable de salida
    X_features = X.drop('calss', axis=1)
    y_target = X['calss']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión
    model = RandomForestRegressor(n_estimators=10, random_state=42)  # n_estimators reducido
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Graficar
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

    
