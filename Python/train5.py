import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from nltk_utils import vector_bolsa_de_palabras, dividir_en_palabras, obtener_raiz, detectar_idioma
from model import NeuralNet

# Depuración: Inicio
print("[DEBUG] Inicio del script")

# Leer el archivo de intenciones
print("[DEBUG] Leyendo archivo de intenciones...")
with open('intents.json', 'r') as archivo:
    intenciones = json.load(archivo)
print("[DEBUG] Archivo de intenciones cargado exitosamente.")

palabras = []
etiquetas = []
datos_entrenamiento = []

# Procesar el archivo de intenciones
print("[DEBUG] Procesando intenciones...")
for intencion in intenciones['intents']:
    etiqueta = intencion['tag']
    etiquetas.append(etiqueta)
    for patron in intencion['patterns']:
        idioma = detectar_idioma(patron)
        tokens = dividir_en_palabras(patron, idioma=idioma)
        palabras.extend([obtener_raiz(token, idioma) for token in tokens])
        datos_entrenamiento.append((tokens, etiqueta, idioma))
print(f"[DEBUG] Procesamiento de intenciones finalizado. Total patrones procesados: {len(datos_entrenamiento)}")

# Filtrar palabras irrelevantes y procesar raíces
print("[DEBUG] Filtrando palabras irrelevantes...")
ignorar = ['?', '.', '!', ',', ':', ';', '...', '(', ')', '[', ']', '{', '}', '-', '_', '/', '|', '\\', '*', '=', '"', "'", '«', '»']
palabras = [palabra for palabra in palabras if palabra not in ignorar]
palabras = sorted(set(palabras))
etiquetas = sorted(set(etiquetas))
print(f"[DEBUG] Total palabras relevantes: {len(palabras)}")
print(f"[DEBUG] Total etiquetas únicas: {len(etiquetas)}")

# Crear conjuntos de entrenamiento
print("[DEBUG] Creando conjuntos de entrenamiento...")
entradas = []
salidas = []

# Barra de progreso para el procesamiento del conjunto de entrenamiento
with tqdm(total=len(datos_entrenamiento), desc="Procesando conjunto de entrenamiento") as barra_entrenamiento:
    for (tokens, etiqueta, idioma) in datos_entrenamiento:
        vector = vector_bolsa_de_palabras(tokens, palabras, idioma=idioma)
        entradas.append(vector)
        salidas.append(etiquetas.index(etiqueta))
        barra_entrenamiento.update(1)

entradas = np.array(entradas)
salidas = np.array(salidas)

print(f"[DEBUG] Conjuntos de entrenamiento creados. Tamaño de entradas: {entradas.shape}, Tamaño de salidas: {salidas.shape}")

# Hiperparámetros
epocas = 50
tamanio_lote = 8
tasa_aprendizaje = 0.001
tam_entrada = len(entradas[0])
tam_oculto = 8
tam_salida = len(etiquetas)
print(f"[DEBUG] Hiperparámetros configurados. Tamaño de entrada: {tam_entrada}, Tamaño de salida: {tam_salida}")

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
print("[DEBUG] Dataset y DataLoader creados exitosamente.")

# Configurar dispositivo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEBUG] Dispositivo configurado: {dispositivo}")

# Crear el modelo
modelo = NeuralNet(tam_entrada, tam_oculto, tam_salida).to(dispositivo)
print("[DEBUG] Modelo inicializado.")

# Configurar función de pérdida y optimizador
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
print("[DEBUG] Función de pérdida y optimizador configurados.")

# Depuración del modelo con datos ficticios
print("[DEBUG] Verificando pase hacia adelante del modelo...")
entrada_prueba = torch.rand(1, tam_entrada).to(dispositivo)
salida_prueba = modelo(entrada_prueba)
print(f"[DEBUG] Pase hacia adelante exitoso. Salida de prueba: {salida_prueba}")

# Entrenamiento
print("[DEBUG] Iniciando entrenamiento...")
for epoca in trange(epocas, desc="Progreso de Épocas"):
    with tqdm(loader, desc=f"Época {epoca+1}/{epocas}") as barra: 
        for lotes_x, lotes_y in barra:
            lotes_x = lotes_x.to(dispositivo)
            lotes_y = lotes_y.to(dispositivo)

            salida = modelo(lotes_x)
            perdida = criterio(salida, lotes_y)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            barra.set_postfix({"Pérdida": perdida.item()})

    if (epoca + 1) % 100 == 0:
        print(f"[DEBUG] Época [{epoca+1}/{epocas}], Pérdida: {perdida.item():.4f}")

print(f"[DEBUG] Entrenamiento finalizado. Pérdida final: {perdida.item():.4f}")

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
    json.dump({"palabras": palabras, "etiquetas": etiquetas}, archivo, indent=4, ensure_ascii=False)
print("[DEBUG] Vocabulario guardado como 'vocabulario.json'")
print(f"[DEBUG] Tamaño del vocabulario utilizado: {len(palabras)}")

# Guardar modelo entrenado
torch.save(modelo_entrenado, "modelo_chatbot.pth")
print("[DEBUG] Entrenamiento completo. Modelo guardado como 'modelo_chatbot.pth'")
print(f"[DEBUG] Vocabulario Entrenado: {palabras}")
