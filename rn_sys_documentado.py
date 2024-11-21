# Importar librerías necesarias para la manipulación de datos, preprocesamiento, modelado, visualización e interfaz gráfica
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para cálculos numéricos avanzados y matrices
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Para codificación de etiquetas y escalado de características
from sklearn.metrics import confusion_matrix  # Para crear una matriz de confusión (evaluación del modelo)
from sklearn.impute import SimpleImputer  # Para imputar valores faltantes en el dataset
import tensorflow as tf  # Para construir y entrenar redes neuronales
from tensorflow.keras import layers, models, callbacks  # Para construir modelos de redes neuronales con capas, callbacks y compilación
import matplotlib.pyplot as plt  # Para gráficos y visualización de datos
import seaborn as sns  # Para visualización avanzada y estilo de gráficos
import streamlit as st  # Para crear la interfaz de usuario de la aplicación web
import joblib  # Para guardar y cargar modelos y objetos de preprocesamiento
import warnings  # Para manejar advertencias en el código
import os  # Para interactuar con el sistema de archivos

# Ignorar las advertencias
warnings.filterwarnings('ignore')

# Definir una clase para el modelo de ciberseguridad
class CyberSecurityModel:
    def __init__(self):
        # Inicializar el modelo, escalador, codificadores y otras variables de clase
        self.model = None  # Modelo de red neuronal que se definirá después
        self.scaler = StandardScaler()  # Escalador para normalizar datos numéricos
        self.label_encoders = {}  # Diccionario para almacenar codificadores de variables categóricas
        self.feature_names = None  # Lista de nombres de características (columnas) seleccionadas
        self.history = None  # Historial de entrenamiento del modelo

    def preprocess_data(self, data):
        """Preprocesamiento de datos"""
        # Crear una copia del conjunto de datos original para evitar modificarlo directamente
        data = data.copy()
        data.columns = data.columns.str.strip()  # Eliminar espacios en los nombres de columnas

        # Mostrar las columnas disponibles en el conjunto de datos
        print("Columnas disponibles en el dataset:", data.columns)

        # Definir los nombres de las características necesarias para el modelo
        self.feature_names = [
            'Source Port', 'Destination Port', 'Protocol',
            'Packet Length', 'Packet Type', 'Traffic Type',
            'Malware Indicators', 'Anomaly Scores', 'Severity Level',
            'Network Segment'
        ]

        # Verificar que todas las características seleccionadas estén presentes en el conjunto de datos
        for feature in self.feature_names:
            if feature not in data.columns:
                raise KeyError(f"La columna '{feature}' no existe en el dataset.")  # Genera un error si falta una columna

        # Separar características en numéricas y categóricas para su preprocesamiento
        numeric_features = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores', 'Severity Level']
        categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Malware Indicators', 'Network Segment']

        # Convertir la columna 'Severity Level' de texto a valores numéricos si es de tipo cadena
        if data['Severity Level'].dtype == 'object':
            severity_mapping = {'Low': 1, 'Medium': 3, 'High': 5}
            data['Severity Level'] = data['Severity Level'].replace(severity_mapping)  # Mapear niveles de severidad

        # Imputación de valores faltantes para características numéricas y categóricas
        num_imputer = SimpleImputer(strategy='mean')  # Imputar valores faltantes numéricos con la media
        data[numeric_features] = num_imputer.fit_transform(data[numeric_features])
        cat_imputer = SimpleImputer(strategy='most_frequent')  # Imputar valores categóricos con el valor más frecuente
        data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

        # Codificar características categóricas usando LabelEncoder
        for cat_feature in categorical_features:
            self.label_encoders[cat_feature] = LabelEncoder()  # Crear un codificador por característica
            data[cat_feature] = self.label_encoders[cat_feature].fit_transform(data[cat_feature])  # Codificar datos

        # Seleccionar y preparar las características para entrenamiento
        X = data[self.feature_names].values  # Seleccionar valores de características

        # Crear una columna objetivo ficticia, ya que no se especifica en el dataset
        y = np.zeros(X.shape[0])  # Vector de ceros para la columna objetivo (dado que no está presente)

        # Escalar las características para normalización
        X = self.scaler.fit_transform(X)  # Escalar las características numéricas

        return X, y  # Devolver características procesadas y columna objetivo ficticia

    def build_model(self, input_shape, num_classes=1):
        """Construcción de la red neuronal"""
        # Definir la estructura de la red neuronal
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),  # Capa densa de 256 neuronas con activación ReLU
            layers.BatchNormalization(),  # Normalización por lotes para estabilizar el entrenamiento
            layers.Dropout(0.3),  # Capa de Dropout para reducir el sobreajuste
            layers.Dense(128, activation='relu'),  # Capa densa de 128 neuronas
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),  # Capa densa de 64 neuronas
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),  # Capa densa de 32 neuronas
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='sigmoid')  # Capa de salida con activación sigmoid para clasificación binaria
        ])
        return model  # Devolver el modelo

    def train_model(self, X, y, epochs=50, batch_size=32):
        """Entrenamiento del modelo"""
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Configurar callbacks para detener temprano y reducir la tasa de aprendizaje
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        # Construir y compilar el modelo
        self.model = self.build_model((X_train.shape[1],))  # Especificar el tamaño de entrada
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Configurar modelo

        # Entrenar el modelo y almacenar el historial de entrenamiento
        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1
        )

        # Evaluar el modelo en el conjunto de prueba y mostrar la precisión
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f'\nPrecisión en el conjunto de prueba: {test_accuracy:.4f}')
        return X_test, y_test  # Devolver el conjunto de prueba para evaluación

    def save_model(self, path='modelo_cybersecurity'):
        """Guardar modelo y preprocesadores"""
        if self.model:
            self.model.save(f'{path}_model.h5')  # Guardar modelo en formato HDF5
            joblib.dump(self.scaler, f'{path}_scaler.pkl')  # Guardar escalador
            joblib.dump(self.label_encoders, f'{path}_encoders.pkl')  # Guardar codificadores
            print("Modelo guardado exitosamente.")
        else:
            print("No se ha entrenado ningún modelo para guardar.")

    def load_model(self, path='modelo_cybersecurity'):
        """Cargar modelo y preprocesadores"""
        self.model = tf.keras.models.load_model(f'{path}_model.h5')  # Cargar el modelo guardado
        self.scaler = joblib.load(f'{path}_scaler.pkl')  # Cargar el escalador
        self.label_encoders = joblib.load(f'{path}_encoders.pkl')  # Cargar codificadores

# Función para crear la aplicación de Streamlit
def create_streamlit_app():
    """Interfaz gráfica con Streamlit"""
    st.title('Sistema de Detección de Ataques de Ciberseguridad')  # Título de la aplicación

    # Cargar el modelo en el estado de sesión de Streamlit
    if 'model' not in st.session_state:
        model = CyberSecurityModel()  # Crear una instancia del modelo
        try:
            # Verificar si el modelo preentrenado existe en el sistema
            if not os.path.exists('modelo_cybersecurity_model.h5'):
                st.warning('No se encontró el modelo preentrenado. Entrenando el modelo...')

                # Cargar los datos desde una URL
                url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTDx2deUjDJ6kbKMnSuLzMSsVnv5VbqSMHQlOba8xYFRlyX8zCJN1ehrQZS1a9MAMzIN_MVWVQmBmH-/pub?output=csv"
                data = pd.read_csv(url)  # Leer datos desde URL
                X, y = model.preprocess_data(data)  # Preprocesar datos
                model.train_model(X, y)  # Entrenar modelo

                model.save_model()  # Guardar modelo entrenado
                st.success('Modelo entrenado y guardado exitosamente')
            
            # Cargar el modelo guardado
            model.load_model()
            st.session_state['model'] = model  # Almacenar en la sesión
            st.success('Modelo cargado exitosamente')
        except Exception as e:
            st.error(f'Error al cargar o entrenar el modelo: {e}')
            st.stop()

    # Formulario para ingresar datos del paquete de red y realizar predicciones
    with st.form('prediction_form'):
        st.subheader('Ingrese los datos del paquete de red:')
        source_port = st.number_input('Puerto de origen', min_value=0, max_value=65535)
        dest_port = st.number_input('Puerto de destino', min_value=0, max_value=65535)
        packet_length = st.number_input('Longitud del paquete', min_value=0)
        anomaly_score = st.slider('Puntuación de anomalía', 0.0, 1.0, 0.5)
        severity = st.slider('Nivel de severidad', 1, 5, 3)
        protocol = st.selectbox('Protocolo', ['TCP', 'UDP', 'ICMP'])
        packet_type = st.selectbox('Tipo de paquete', ['Data', 'Control', 'Management'])
        traffic_type = st.selectbox('Tipo de tráfico', ['Normal', 'Suspicious', 'Malicious'])
        malware_indicator = st.selectbox('Indicador de malware', ['Yes', 'No'])
        network_segment = st.selectbox('Segmento de red', ['Internal', 'DMZ', 'External'])

        submitted = st.form_submit_button('Analizar Paquete')

        if submitted:
            # Crear DataFrame con los datos de entrada del usuario
            input_data = pd.DataFrame({
                'Source Port': [source_port],
                'Destination Port': [dest_port],
                'Protocol': [protocol],
                'Packet Length': [packet_length],
                'Packet Type': [packet_type],
                'Traffic Type': [traffic_type],
                'Malware Indicators': [malware_indicator],
                'Anomaly Scores': [anomaly_score],
                'Severity Level': [severity],
                'Network Segment': [network_segment]
            })

            model = st.session_state['model']
            X, _ = model.preprocess_data(input_data)  # Preprocesar datos de entrada
            prediction = model.model.predict(X)  # Realizar predicción
            predicted_class = (prediction > 0.5).astype(int)[0][0]  # Convertir predicción a clase binaria

            st.write(f'Resultado de predicción: {"Ataque" if predicted_class == 1 else "No Ataque"}')  # Mostrar resultado

# Ejecutar la aplicación de Streamlit si se llama directamente
if __name__ == "__main__":
    create_streamlit_app()
