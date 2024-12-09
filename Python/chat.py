import random
import json
import torch
from model import NeuralNet
from nltk_utils import vector_bolsa_de_palabras, dividir_en_palabras
import re 

# Configurar dispositivo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar datos del modelo entrenado
with open('intents.json', 'r') as archivo:
    intenciones = json.load(archivo)

modelo_entrenado = torch.load("modelo_chatbot.pth")

tam_entrada = modelo_entrenado["input_size"]
tam_oculto = modelo_entrenado["hidden_size"]
tam_salida = modelo_entrenado["output_size"]
palabras = modelo_entrenado["all_words"]
etiquetas = modelo_entrenado["tags"]
estado_modelo = modelo_entrenado["model_state"]

# Cargar el modelo
modelo = NeuralNet(tam_entrada, tam_oculto, tam_salida).to(dispositivo)
modelo.load_state_dict(estado_modelo)
modelo.eval()

# Variables de contexto
contexto_actual = {}

nombre_bot = "ChatBot"
print("¡Hablemos! (Escribe 'salir' para terminar)")

while True:
    texto_usuario = input("Tú: ")
    if texto_usuario.lower() == "salir":
        print(f"{nombre_bot}: ¡Hasta luego!")
        break

    # Tokenizar la entrada del usuario
    tokens = dividir_en_palabras(texto_usuario)
    #print(f"[DEBUG] Tokens Generados: {tokens}")  # Depuración: Ver tokens

    # Crear el vector de bolsa de palabras
    vector = vector_bolsa_de_palabras(tokens, palabras)
    #print(f"[DEBUG] Vector de Bolsa de Palabras: {vector}")  # Depuración: Ver vector

    # Convertir el vector a tensor y pasarlo al modelo
    vector = vector.reshape(1, vector.shape[0])
    vector = torch.from_numpy(vector).to(dispositivo)

    salida = modelo(vector)
    _, prediccion = torch.max(salida, dim=1)
    etiqueta = etiquetas[prediccion.item()]

    probabilidad = torch.softmax(salida, dim=1)
    confianza = probabilidad[0][prediccion.item()]

    # Verificar si la confianza es suficiente
    if confianza.item() > 0.75:
        for intencion in intenciones["intents"]:
            if etiqueta == intencion["tag"]:
                print(f"{nombre_bot}: {random.choice(intencion['responses'])}")
                
                # Depurar números extraídos
                if etiqueta in ["suma", "resta", "multiplicacion", "division"]:
                    numeros = [float(num) for num in re.findall(r'\d+', texto_usuario)]
                    #print(f"[DEBUG] Números Extraídos: {numeros}")  # Depuración: Ver números

                    if len(numeros) >= 2:
                        # Realizar la operación matemática
                        num1, num2 = numeros[:2]
                        if etiqueta == "suma":
                            resultado = num1 + num2
                        elif etiqueta == "resta":
                            resultado = num1 - num2
                        elif etiqueta == "multiplicacion":
                            resultado = num1 * num2
                        elif etiqueta == "division":
                            if num2 != 0:
                                resultado = num1 / num2
                            else:
                                resultado = "No puedo dividir entre cero. 😅"
                        print(f"{nombre_bot}: El resultado es {resultado}")
                    else:
                        print(f"{nombre_bot}: Parece que faltan números para calcular. Intenta de nuevo.")
    else:
        print(f"{nombre_bot}: Lo siento, no entiendo eso.")

