import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import json
from langdetect import detect
import torch
from model import NeuralNet
import random


# Inicializar el stemmer para español y el lematizador para inglés
stemmer_es = SnowballStemmer("spanish")
lemmatizer_en = WordNetLemmatizer()

# Funciones auxiliares para procesar
def detectar_idioma(oracion):
    try:
        idioma = detect(oracion)
        return "spanish" if idioma == "es" else "english"
    except:
        return "spanish"  # Por defecto español

def dividir_en_palabras(oracion, idioma="spanish"):
    oracion = oracion.lower()
    oracion = re.sub(r'[^\w\s]', '', oracion)  # Eliminar signos de puntuación
    tokens = nltk.word_tokenize(oracion, language=idioma)
    return tokens

def obtener_pos_etiqueta(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def obtener_raiz(palabra, idioma="spanish"):
    excepciones = ["suma", "resta", "multiplicacion", "division", "+", "-", "*", "/"]
    if palabra in excepciones or palabra.isdigit():
        return palabra
    if idioma == "spanish":
        raiz = stemmer_es.stem(palabra)
        #print(f"[DEBUG] Palabra original: {palabra}, Raíz en español: {raiz}")
        return raiz
    elif idioma == "english":
        pos_tag = nltk.pos_tag([palabra])[0][1]
        wordnet_tag = obtener_pos_etiqueta(pos_tag)
        raiz = lemmatizer_en.lemmatize(palabra, pos=wordnet_tag)
        #print(f"[DEBUG] Palabra original: {palabra}, Raíz en inglés: {raiz}")
        return raiz

def vector_bolsa_de_palabras(oracion_tokenizada, palabras_conocidas, idioma="spanish"):
    palabras_raiz = [obtener_raiz(palabra, idioma) for palabra in oracion_tokenizada]
    palabras_no_encontradas = [palabra for palabra in palabras_raiz if palabra not in palabras_conocidas]
    #print(f"[DEBUG] Palabras raíz generadas: {palabras_raiz}")
    #print(f"[DEBUG] Palabras raíz no encontradas en all_words: {palabras_no_encontradas}")

    vector = np.zeros(len(palabras_conocidas), dtype=np.float32)
    for idx, palabra in enumerate(palabras_conocidas):
        if palabra in palabras_raiz:
            vector[idx] = 1
    return vector

'''
# Cargar modelo y datos
def cargar_datos_modelo(ruta_intents, ruta_modelo):
    with open(ruta_intents, 'r') as archivo:
        intenciones = json.load(archivo)

    modelo_entrenado = torch.load(ruta_modelo)

    return (
        intenciones,
        modelo_entrenado["input_size"],
        modelo_entrenado["hidden_size"],
        modelo_entrenado["output_size"],
        modelo_entrenado["all_words"],
        modelo_entrenado["tags"],
        modelo_entrenado["model_state"]
    )


def cargar_modelo(input_size, hidden_size, output_size, model_state):
    modelo = NeuralNet(input_size, hidden_size, output_size)
    modelo.load_state_dict(model_state)
    modelo.eval()
    return modelo


# Función principal del chatbot
def ejecutar_chatbot(modelo, palabras, etiquetas, intenciones):
    print("¡Hablemos! (Escribe 'salir' para terminar)")
    while True:
        texto_usuario = input("Tú: ")
        if texto_usuario.lower() == "salir":
            print("ChatBot: ¡Hasta luego!")
            break

        # Detectar idioma
        idioma = detectar_idioma(texto_usuario)
        print(f"[DEBUG] Idioma detectado: {idioma}")

        # Procesar entrada
        tokens = dividir_en_palabras(texto_usuario, idioma=idioma)
        print(f"[DEBUG] Tokens procesados: {tokens}")

        # Generar vector de bolsa de palabras
        vector = vector_bolsa_de_palabras(tokens, palabras, idioma=idioma).reshape(1, -1)
        print(f"[DEBUG] Vector de entrada al modelo: {vector}")

        # Predicción
        salida = modelo(torch.from_numpy(vector).float())
        probabilidades = torch.softmax(salida, dim=1)
        _, prediccion = torch.max(salida, dim=1)
        etiqueta = etiquetas[prediccion.item()]
        print(f"[DEBUG] Salida del modelo: {salida}")
        print(f"[DEBUG] Probabilidades: {probabilidades}")
        print(f"[DEBUG] Predicción: {prediccion.item()}, Etiqueta: {etiqueta}")

        # Responder según la etiqueta y confianza
        confianza = probabilidades[0][prediccion.item()]
        if confianza > 0.75:
            for intencion in intenciones["intents"]:
                if etiqueta == intencion["tag"]:
                    respuesta = random.choice(intencion["responses"])
                    print(f"ChatBot: {respuesta}")
                    break
        else:
            print("ChatBot: Lo siento, no entiendo eso.")


# Main
if __name__ == "__main__":
    # Rutas
    ruta_intents = "intents.json"
    ruta_modelo = "modelo_chatbot.pth"

    # Cargar datos y modelo
    (intenciones, tam_entrada, tam_oculto, tam_salida, palabras, etiquetas, estado_modelo) = cargar_datos_modelo(ruta_intents, ruta_modelo)
    modelo = cargar_modelo(tam_entrada, tam_oculto, tam_salida, estado_modelo)

    # Ejecutar chatbot
    ejecutar_chatbot(modelo, palabras, etiquetas, intenciones)
'''