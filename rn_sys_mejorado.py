import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class CyberSecurityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.history = None

    def preprocess_data(self, data):
        """Preprocesamiento de datos"""
        data = data.copy()
        data.columns = data.columns.str.strip()

        # Mostrar columnas disponibles
        print("Columnas disponibles en el dataset:", data.columns)

        # Selección de características
        self.feature_names = [
            'Source Port', 'Destination Port', 'Protocol',
            'Packet Length', 'Packet Type', 'Traffic Type',
            'Malware Indicators', 'Anomaly Scores', 'Severity Level',
            'Network Segment'
        ]

        # Verificar que todas las características existan en el dataset
        for feature in self.feature_names:
            if feature not in data.columns:
                raise KeyError(f"La columna '{feature}' no existe en el dataset.")

        # Manejo de valores nulos
        numeric_features = ['Source Port', 'Destination Port', 'Packet Length', 'Anomaly Scores', 'Severity Level']
        categorical_features = ['Protocol', 'Packet Type', 'Traffic Type', 'Malware Indicators', 'Network Segment']

        # Conversión de 'Severity Level' de texto a valores numéricos si es necesario
        if data['Severity Level'].dtype == 'object':
            severity_mapping = {'Low': 1, 'Medium': 3, 'High': 5}
            data['Severity Level'] = data['Severity Level'].replace(severity_mapping)

        # Imputación de valores
        num_imputer = SimpleImputer(strategy='mean')
        data[numeric_features] = num_imputer.fit_transform(data[numeric_features])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

        # Codificación de variables categóricas
        for cat_feature in categorical_features:
            self.label_encoders[cat_feature] = LabelEncoder()
            data[cat_feature] = self.label_encoders[cat_feature].fit_transform(data[cat_feature])

        # Preparar características
        X = data[self.feature_names].values

        # No hay columna de objetivo en el dataset, usar un valor ficticio
        y = np.zeros(X.shape[0])

        # Normalización de características
        X = self.scaler.fit_transform(X)

        return X, y

    def build_model(self, input_shape, num_classes=1):
        """Construcción de la red neuronal"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='sigmoid')
        ])
        return model

    def train_model(self, X, y, epochs=50, batch_size=32):
        """Entrenamiento del modelo"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        # Construir y compilar el modelo
        self.model = self.build_model((X_train.shape[1],))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Entrenar el modelo
        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1
        )

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f'\nPrecisión en el conjunto de prueba: {test_accuracy:.4f}')
        return X_test, y_test

    def save_model(self, path='modelo_cybersecurity'):
        """Guardar modelo y preprocesadores"""
        if self.model:
            self.model.save(f'{path}_model.h5')
            joblib.dump(self.scaler, f'{path}_scaler.pkl')
            joblib.dump(self.label_encoders, f'{path}_encoders.pkl')
            print("Modelo guardado exitosamente.")
        else:
            print("No se ha entrenado ningún modelo para guardar.")

    def load_model(self, path='modelo_cybersecurity'):
        """Cargar modelo y preprocesadores"""
        self.model = tf.keras.models.load_model(f'{path}_model.h5')
        self.scaler = joblib.load(f'{path}_scaler.pkl')
        self.label_encoders = joblib.load(f'{path}_encoders.pkl')

def create_streamlit_app():
    """Interfaz gráfica con Streamlit"""
    st.title('Sistema de Detección de Ataques de Ciberseguridad')

    if 'model' not in st.session_state:
        model = CyberSecurityModel()
        try:
            # Verifica si el archivo del modelo existe
            if not os.path.exists('modelo_cybersecurity_model.h5'):
                st.warning('No se encontró el modelo preentrenado. Entrenando el modelo...')

                # Cargar los datos desde la URL
                url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTDx2deUjDJ6kbKMnSuLzMSsVnv5VbqSMHQlOba8xYFRlyX8zCJN1ehrQZS1a9MAMzIN_MVWVQmBmH-/pub?output=csv"
                data = pd.read_csv(url)
                X, y = model.preprocess_data(data)
                model.train_model(X, y)

                model.save_model()
                st.success('Modelo entrenado y guardado exitosamente')
            
            model.load_model()
            st.session_state['model'] = model
            st.success('Modelo cargado exitosamente')
        except Exception as e:
            st.error(f'Error al cargar o entrenar el modelo: {e}')
            st.stop()

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
            X, _ = model.preprocess_data(input_data)
            prediction = model.model.predict(X)
            predicted_class = (prediction > 0.5).astype(int)[0][0]

            st.write(f'Resultado de predicción: {"Ataque" if predicted_class == 1 else "No Ataque"}')

if __name__ == "__main__":
    create_streamlit_app()
