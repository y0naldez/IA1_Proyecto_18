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

# Guardar vocabulario en un archivo JSON
with open("vocabulario.json", "w") as archivo:
    json.dump(palabras, archivo, indent=4, ensure_ascii=False)
print("Vocabulario guardado como 'vocabulario.json'")

print(f"Tamaño del vocabulario utilizado: {len(palabras)}")


torch.save(modelo_entrenado, "modelo_chatbot.pth")
print("Entrenamiento completo. Modelo guardado como 'modelo_chatbot.pth'")
print(f"[DEBUG] Vocabulario Entrenado: {palabras}")
