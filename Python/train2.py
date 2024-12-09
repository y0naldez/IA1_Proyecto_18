import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.snowball import SnowballStemmer
import re

# Inicializar el stemmer para procesar palabras en español
stemmer = SnowballStemmer("spanish")

# Archivo para guardar los datos de depuración
archivo_debug = "debug.json"

# Estructura para almacenar los datos de depuración
datos_debug = {
    "tokens": [],
    "raices": [],
    "vector_bolsa_de_palabras": []
}

def guardar_debug():
    """Guarda los datos de depuración en un archivo JSON."""
    with open(archivo_debug, "w") as archivo:
        json.dump(datos_debug, archivo, indent=4, ensure_ascii=False)

def dividir_en_palabras(oracion):
    """
    Divide una oración en palabras o tokens y convierte a minúsculas.
    """
    oracion = oracion.lower()
    oracion = re.sub(r'([\+\-\*/])', r' \1 ', oracion)  # Separar operadores matemáticos
    tokens = nltk.word_tokenize(oracion, language="spanish")
    
    # Guardar tokens en datos de depuración
    datos_debug["tokens"].append({"oracion": oracion, "tokens": tokens})
    guardar_debug()
    return tokens

def obtener_raiz(palabra):
    """
    Reduce una palabra a su forma raíz, excepto para palabras clave matemáticas y números.
    """
    excepciones = ["suma", "resta", "multiplicacion", "division", "+", "-", "*", "/"]
    if palabra in excepciones or palabra.isdigit():
        return palabra
    return stemmer.stem(palabra)

def vector_bolsa_de_palabras(oracion_tokenizada, palabras_conocidas):
    """
    Convierte una oración tokenizada en un vector de bolsa de palabras.
    """
    palabras_raiz = [obtener_raiz(palabra) for palabra in oracion_tokenizada]
    
    # Guardar raíces generadas en datos de depuración
    datos_debug["raices"].append({"tokens": oracion_tokenizada, "raices": palabras_raiz})
    
    vector = np.zeros(len(palabras_conocidas), dtype=np.float32)
    for idx, palabra in enumerate(palabras_conocidas):
        if palabra in palabras_raiz:
            vector[idx] = 1
    
    # Guardar vector generado en datos de depuración
    datos_debug["vector_bolsa_de_palabras"].append({"raices": palabras_raiz, "vector": vector.tolist()})
    guardar_debug()
    return vector

# Leer intenciones
with open('intents.json', 'r') as archivo:
    intenciones = json.load(archivo)

# Procesar datos
palabras = []
etiquetas = []
datos_entrenamiento = []

ignorar = ['?', '.', '!', ',', ':', ';', '...', '(', ')', '[', ']', '{', '}', '-', '_', '/', '|', '\\', '*', '=', '"', "'", '«', '»']

for intencion in intenciones['intents']:
    etiqueta = intencion['tag']
    etiquetas.append(etiqueta)
    for patron in intencion['patterns']:
        tokens = dividir_en_palabras(patron)
        palabras.extend([obtener_raiz(token) for token in tokens if token not in ignorar])
        datos_entrenamiento.append((tokens, etiqueta))

palabras = sorted(set(palabras))
etiquetas = sorted(set(etiquetas))

# Crear vectores de bolsa de palabras
entradas = []
salidas = []

for tokens, etiqueta in datos_entrenamiento:
    vector = vector_bolsa_de_palabras(tokens, palabras)
    entradas.append(vector)
    salidas.append(etiquetas.index(etiqueta))

entradas = np.array(entradas)
salidas = tf.keras.utils.to_categorical(salidas, num_classes=len(etiquetas))

# Crear modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(entradas[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(etiquetas), activation='softmax')
])

# Compilar el modelo
modelo.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(entradas, salidas, epochs=1000, batch_size=8, verbose=1)

# Guardar vocabulario en un archivo JSON
with open("vocabulario.json", "w") as archivo:
    json.dump(palabras, archivo, indent=4, ensure_ascii=False)
print("Vocabulario guardado como 'vocabulario.json'")

print(f"Tamaño del vocabulario utilizado: {len(palabras)}")

# Guardar el modelo en formato SavedModel
modelo.export("modelo_chatbot_tf")

print("Modelo guardado como SavedModel en 'modelo_chatbot_tf'")
print(f"[DEBUG] Vocabulario Entrenado: {palabras}")
