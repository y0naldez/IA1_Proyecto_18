# **Manual Tecnico Proyecto Fase 2 Inteligencia Artificial 1**

## **Introducción**
En este documento se describe los aspectos técnicos informáticos de la aplicación de escritorio, de manera que cualquier técnico informático pueda entender y comprender la lógica dentro del programa, para así poder darle mantenimiento y actualizarla si es necesario.


## **Objetivo Gerneral**

- Implementar lo aprendido en la clase magistral y laboratorio en la creación de un
modelo que sea capaz de entender entradas de texto y responder de forma coherente
a las mismas.

## **Objetivos específicos**
- Seleccionar una biblioteca en el lenguaje de JavaScript que permita la creación de un
modelo de inteligencia artificial.
- Desarrollar un modelo que sea capaz de recibir entradas en idioma español y
responder a las mismas.

## **Requisitos del sistema**
- CPU, Intel Core 3  2 GHz recomendado.
- RAM, 2 GB recomendado. 
- Sistema Operativo windows 10,11.
- Navegador web.

## Red Neuronal feed-forward con dos capas ocultas
![alt text](RED.png)
En la imagen se visualiza la representación de texto mediante el método de Bolsa de Palabras (Bag of Words) en un conjunto de datos de entrenamiento para un modelo de clasificación, se forma un vector y dependiendo de el vector lo enlaza con una intención.

## Librerias utilizadas

- random: Para seleccionar respuestas aleatorias.
- json: Para cargar datos del archivo intents.json.
- torch: Para trabajar con PyTorch.
- NeuralNet: Modelo de red neuronal definido en model.py.
- vector_bolsa_de_palabras y dividir_en_palabras: Funciones auxiliares para procesar texto.
- re: Para manejar expresiones regulares.

```
import random
import json
import torch
from model import NeuralNet
from nltk_utils import vector_bolsa_de_palabras, dividir_en_palabras
import re

```

## Configuración del dispostivo

```
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```


## Cargar Datos y modelo Entrenado 

Se cargan los datos de intenciones y el modelo entrenado previamente guardado en modelo_chatbot.pth.

```
with open('intents.json', 'r') as archivo:
    intenciones = json.load(archivo)

modelo_entrenado = torch.load("modelo_chatbot.pth")

```

## Configuracion del modelo 

Se extraen los tamaños de entrada, salida y capa oculta, así como las palabras y etiquetas. Luego, se carga el modelo y se establece en modo de evaluación (eval()).



```
tam_entrada = modelo_entrenado["input_size"]
tam_oculto = modelo_entrenado["hidden_size"]
tam_salida = modelo_entrenado["output_size"]
palabras = modelo_entrenado["all_words"]
etiquetas = modelo_entrenado["tags"]
estado_modelo = modelo_entrenado["model_state"]

modelo = NeuralNet(tam_entrada, tam_oculto, tam_salida).to(dispositivo)
modelo.load_state_dict(estado_modelo)
modelo.eval()

```

## Red Neuronal

Clasificacion de texto para el chatbot, recibe como entrada un vector de caracteristicas, produce las salidas para cada clase, uso de softmax para probablidad de clasificacion.

Este método define cómo la red procesa los datos en el paso de inferencia:

- Primera Capa: Se pasa x a través de l1 y se aplica ReLU.
- Segunda Capa: Se pasa a l2 y se aplica ReLU nuevamente.
- Tercera Capa: Se pasa a l3

```
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
```

## Stemer, Depuracion y Almacenado de datos

El archivo nlkt_utils.py contiene el stemer para poder procesar las palabras en español, archiva los datos de depuracion y almacena los datos, los datos se almacenan en un archivo JSON, se dividen en tokens y se convierten a minusculas, se maneja a travez de vectores.

```
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import json

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

```

## Lectura, procesamiento, filtrado y entrenamiento 

En este archivo train.py contiene secciones importantes filtra palabras irrelevantes y procesar raíces, crea conjuntos de entrenamiento, se encarga de crear el dataset personalizado.

- Configura el dispositivo.
- Crea el modelo
- Entrenamiento
- Guardar modelo entrenado.
  
```
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nltk_utils import vector_bolsa_de_palabras, dividir_en_palabras, obtener_raiz
from model import NeuralNet

# Leer el archivo de intenciones
with open('intents.json', 'r') as archivo:
    intenciones = json.load(archivo)

palabras = []
etiquetas = []
datos_entrenamiento = []

# Procesar el archivo de intenciones
for intencion in intenciones['intents']:
    etiqueta = intencion['tag']
    etiquetas.append(etiqueta)
    for patron in intencion['patterns']:
        tokens = dividir_en_palabras(patron)
        palabras.extend(tokens)
        datos_entrenamiento.append((tokens, etiqueta))

# Filtrar palabras irrelevantes y procesar raíces
ignorar = ignorar = ['?', '.', '!', ',', ':', ';', '...', '(', ')', '[', ']', '{', '}', '-', '_', '/', '|', '\\', '*', '=', '"', "'", '«', '»']
palabras = [obtener_raiz(palabra) for palabra in palabras if palabra not in ignorar]
palabras = sorted(set(palabras))
etiquetas = sorted(set(etiquetas))

# Crear conjuntos de entrenamiento
entradas = []
salidas = []
for (tokens, etiqueta) in datos_entrenamiento:
    vector = vector_bolsa_de_palabras(tokens, palabras)
    entradas.append(vector)
    salidas.append(etiquetas.index(etiqueta))

entradas = np.array(entradas)
salidas = np.array(salidas)

# Hiperparámetros
epocas = 1000
tamanio_lote = 8
tasa_aprendizaje = 0.001
tam_entrada = len(entradas[0])
tam_oculto = 8
tam_salida = len(etiquetas)

# Crear el dataset personalizado
class DatasetConversacional(Dataset):
    def __init__(self):
        self.num_muestras = len(entradas)
        self.x = entradas
        self.y = salidas

    def __getitem__(self, indice):
        return self.x[indice], self.y[indice]

    def __len__(self):
        return self.num_muestras

dataset = DatasetConversacional()
loader = DataLoader(dataset=dataset, batch_size=tamanio_lote, shuffle=True)

# Configurar dispositivo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Crear el modelo
modelo = NeuralNet(tam_entrada, tam_oculto, tam_salida).to(dispositivo)

# Configurar función de pérdida y optimizador
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

# Entrenamiento
for epoca in range(epocas):
    with tqdm(loader, desc=f"Época {epoca+1}/{epocas}") as barra:  # Barra de progreso
        for lotes_x, lotes_y in barra:
            lotes_x = lotes_x.to(dispositivo)
            lotes_y = lotes_y.to(dispositivo)

            salida = modelo(lotes_x)
            perdida = criterio(salida, lotes_y)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            # Actualizar la barra con la pérdida actual
            barra.set_postfix({"Pérdida": perdida.item()})

    if (epoca + 1) % 100 == 0:
        print(f"Época [{epoca+1}/{epocas}], Pérdida: {perdida.item():.4f}")

print(f"Pérdida final: {perdida.item():.4f}")

# Guardar modelo entrenado
modelo_entrenado = {
    "model_state": modelo.state_dict(),
    "input_size": tam_entrada,
    "hidden_size": tam_oculto,
    "output_size": tam_salida,
    "all_words": palabras,
    "tags": etiquetas,
}

print(f"Tamaño del vocabulario utilizado: {len(palabras)}")

torch.save(modelo_entrenado, "modelo_chatbot.pth")
print("Entrenamiento completo. Modelo guardado como 'modelo_chatbot.pth'")
print(f"[DEBUG] Vocabulario Entrenado: {palabras}")

```

- El archivo train2.py contiene exportado el vocabulario en un Json y exporta el modelo en .pth para poder transformar a tensor Flow y despues de esto poder pasarlo a JS.
  
```
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

```

# Procesamiento de Palabras en Español e Inglés con JavaScript

   El enfoque incluye normalización, tokenización, mapeo dinámico de raíces y la creación de un vector de bolsa de palabras.

  

---

  

## 1. Tokenización con `wink-tokenizer`

  

Para dividir una oración en palabras, utilizamos la librería `wink-tokenizer`. El código inicializa un tokenizador y define funciones para procesar palabras:

  

```javascript

import  winkTokenizer  from  'wink-tokenizer';

  

const  tokenizer = new  winkTokenizer();

```

  

---

  

## 2. Normalización de Palabras

  

La función `normalizarToken` convierte palabras a minúsculas y elimina acentos para uniformizar los tokens, facilitando su comparación.

  

```javascript

/**

* Normaliza un token para comparación (elimina tildes, pasa a minúsculas).

* @param  {string}  token - Token a normalizar.

* @returns  {string} - Token normalizado.

*/

const  normalizarToken = (token) =>

token.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();

```

  

### Ejemplo:

Entrada: `"Avión"`

Salida: `"avion"`

  

---

  

## 3. Búsqueda de Raíces en el Vocabulario

  

La función `encontrarRaizDinamica` busca la forma más relevante de una palabra en un vocabulario conocido. Esto incluye coincidencias exactas, búsqueda de prefijos y formas con prefijos `##`.

  

```javascript

/**

* Encuentra la raíz más relevante en el vocabulario, considerando prefijos.

* @param  {string}  token - La palabra a buscar.

* @param  {Set<string>}  vocabulario - Un conjunto de palabras conocidas (para eficiencia).

* @returns  {string} - La raíz más cercana encontrada en el vocabulario o el token original.

*/

const  encontrarRaizDinamica = (token, vocabulario) => {

const  tokenNormalizado = normalizarToken(token);

  

if (vocabulario.has(tokenNormalizado)) {

return  tokenNormalizado;

}

  

for (let  i = tokenNormalizado.length; i > 0; i--) {

const  prefijo = tokenNormalizado.slice(0, i);

if (vocabulario.has(prefijo)) {

return  prefijo;

}

}

  

for (let  item  of  vocabulario) {

if (item.startsWith('##') && item.slice(2) === tokenNormalizado) {

return  item;

}

}

  

return  tokenNormalizado;

};

```

  

---

  

## 4. División en Palabras y Mapeo de Raíces

  

La función `dividirEnPalabras` tokeniza una oración y convierte cada palabra a su raíz más relevante.

  

```javascript

/**

* Divide una oración en palabras y aplica un mapeo dinámico de raíces.

* @param  {string}  oracion - La entrada del usuario.

* @param  {string[]}  palabrasConocidas - Lista de palabras en el vocabulario.

* @returns  {string[]} - Lista de raíces de las palabras (tokens procesados).

*/

export  const  dividirEnPalabras = (oracion, palabrasConocidas) => {

const  vocabulario = new  Set(palabrasConocidas.map(normalizarToken));

const  tokens = tokenizer.tokenize(oracion.toLowerCase())

.filter(token  =>  token.tag === 'word' || token.tag === 'number')

.map(token  =>  token.value);

  

const  tokensConRaices = tokens.map(token  =>  encontrarRaizDinamica(token, vocabulario));

  

console.log(`[DEBUG] Tokens originales: ${tokens}`);

console.log(`[DEBUG] Tokens con raíces: ${tokensConRaices}`);

return  tokensConRaices;

};

```

  

---

  

## 5. Creación de un Vector de Bolsa de Palabras

  

La función `vectorBolsaDePalabras` genera un vector binario indicando la presencia de cada palabra del vocabulario en los tokens procesados.

  

```javascript

/**

* Convierte una lista de raíces en un vector de bolsa de palabras.

* @param  {string[]}  tokens - Raíces de las palabras procesadas.

* @param  {string[]}  palabrasConocidas - Palabras del vocabulario.

* @returns  {number[]} - Vector de bolsa de palabras.

*/

export  const  vectorBolsaDePalabras = (tokens, palabrasConocidas) => {

const  palabrasNormalizadas = palabrasConocidas.map(palabra  =>  palabra.toLowerCase());

const  vector = Array(palabrasNormalizadas.length).fill(0);

  

tokens.forEach(token  => {

const  tokenNormalizado = token.toLowerCase();

  

let  indice = palabrasNormalizadas.indexOf(tokenNormalizado);

  

if (indice === -1) {

indice = palabrasNormalizadas.indexOf(`##${tokenNormalizado}`);

}

  

if (indice !== -1) {

vector[indice] = 1;

console.log(`[DEBUG] Token "${token}" encontrado en índice ${indice}`);

} else {

console.warn(`[DEBUG] Token "${token}" no encontrado en el vocabulario`);

}

});

  

console.log(`[DEBUG] Vector de bolsa de palabras generado: ${vector}`);

return  vector;

};

```

  

### Ejemplo:

-  **Entrada Tokens**: `["hola", "mundo"]`

-  **Vocabulario**: `["hola", "mundo", "adios"]`

-  **Vector Generado**: `[1, 1, 0]`

  

---

  

## 6. Flujo del Proceso

  

1.  **Entrada del Usuario**: Una oración o frase.

2.  **Tokenización**: División en palabras.

3.  **Normalización**: Eliminar acentos y convertir a minúsculas.

4.  **Mapeo de Raíces**: Buscar formas relevantes en el vocabulario.

5.  **Bolsa de Palabras**: Generar un vector indicando la presencia de cada palabra en la entrada.

  

---

  
    

# Entrenamiento del Chatbot en Google Colab

  

Este documento explica el enfoque utilizado para entrenar el modelo de chatbot en Google Colab. Se destacan las herramientas elegidas, incluyendo la lematización para inglés, el stemming para español y el tokenizador multilingüe de Hugging Face.

  

---

  

## ¿Por qué se usaron estas herramientas específicas?

  

1.  **Lematizador (Inglés):** Se utilizó el `WordNetLemmatizer` de NLTK por su eficiencia para manejar palabras en inglés, ya que ayuda a reducir las palabras a su forma base en el diccionario.

2.  **Stemmer (Español):** El `SnowballStemmer` es eficiente para manejar formas flexionadas en español, simplificando la gestión del vocabulario.

3.  **Tokenizador de Hugging Face:** Se empleó el tokenizador `bert-base-multilingual-cased` por su robustez para manejar múltiples idiomas, proporcionando salidas tokenizadas que soportan tanto texto en español como en inglés.

4.  **Optimización:** Operaciones de TensorFlow como tensores irregulares (`ragged tensors`) y comparaciones vectorizadas aceleran significativamente la generación de vectores de bolsa de palabras, especialmente para datos por lotes.

  

---

  

## Explicación del Código

  

### Importación de Librerías

  

```python

import json

import numpy as np

import tensorflow as tf

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

from langdetect import detect

from tokenizers import Tokenizer

from tokenizers.pre_tokenizers import Whitespace

```

  

### Inicialización del Tokenizador, Stemmer y Lematizador

  

```python

# Stemmer para español y lematizador para inglés

stemmer_es = SnowballStemmer("spanish")

lemmatizer_en = WordNetLemmatizer()

  

# Tokenizador de Hugging Face

tokenizer = Tokenizer.from_pretrained("bert-base-multilingual-cased")

tokenizer.pre_tokenizer = Whitespace()

```

  

---

  

## Funciones Clave

  

### Tokenización y Depuración

  

El tokenizador divide las oraciones en tokens y los datos de depuración se almacenan para inspección posterior.

  

```python

def  dividir_en_palabras_lote(oraciones):

resultados = [tokenizer.encode(oracion.lower()).tokens for oracion in oraciones]

for oracion, tokens in  zip(oraciones, resultados):

datos_debug["tokens"].append({"oracion": oracion, "tokens": tokens})

guardar_debug()

return resultados

```

  

### Generación de Raíces por Lote

  

Dependiendo del idioma detectado, las palabras se procesan con el stemmer para español o el lematizador para inglés.

  

```python

def  obtener_raices_lote_optimizado(oraciones_tokenizadas, idioma="spanish"):

excepciones = set(["suma", "resta", "multiplicacion", "add", "subtract", "multiply", "division", "divide", "+", "-", "*", "/"])

raices = []

  

if idioma == "spanish":

for oracion in oraciones_tokenizadas:

raices.append([stemmer_es.stem(palabra) if palabra not  in excepciones else palabra for palabra in oracion])

elif idioma == "english":

for oracion in oraciones_tokenizadas:

raices.append([lemmatizer_en.lemmatize(palabra) if palabra not  in excepciones else palabra for palabra in oracion])

else:

raise  ValueError(f"Idioma '{idioma}' no soportado.")

  

return raices

```

  

---

  

## Entrenamiento del Modelo

  

El modelo utiliza una red neuronal con capas densas y dropout para evitar sobreajuste. Se entrenó utilizando los vectores de bolsa de palabras generados a partir de las raíces procesadas.

  

### Creación de Vectores de Bolsa de Palabras

  

```python

def  vector_bolsa_de_palabras_lote_optimizado(oraciones_tokenizadas, palabras_conocidas, idioma="spanish"):

raices_por_oracion = obtener_raices_lote_optimizado(oraciones_tokenizadas, idioma)

  

palabras_conocidas_tf = tf.constant(palabras_conocidas, dtype=tf.string)

raices_por_oracion_tf = tf.ragged.constant(raices_por_oracion, dtype=tf.string)

  

comparaciones = tf.equal(raices_por_oracion_tf[:, :, None], palabras_conocidas_tf[None, None, :])

vectores = tf.reduce_any(comparaciones, axis=1)

vectores = tf.cast(vectores, dtype=tf.float32)

  

return vectores.numpy()

```

  

### Definición del Modelo

  

```python

modelo = tf.keras.Sequential([

tf.keras.layers.Dense(128, input_shape=(len(entradas[0]),), activation='relu'),

tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(len(etiquetas), activation='softmax')

])

  

modelo.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

```

  

---

  

## Resultados y Guardado

  

1.  **Vocabulario:** Se guarda en un archivo JSON para su reutilización.

2.  **Modelo:** Exportado en formato `SavedModel` para facilitar su despliegue.

  

```python

with  open("vocabulario.json", "w") as archivo:

json.dump({"palabras": palabras, "etiquetas": etiquetas}, archivo, indent=4, ensure_ascii=False)

  

modelo.export("modelo_chatbot_tf")

print("Modelo guardado como SavedModel en 'modelo_chatbot_tf'")

```

  

---

  

Este enfoque permite procesar múltiples idiomas de manera eficiente y entrenar el chatbot optimizado para tareas específicas.

