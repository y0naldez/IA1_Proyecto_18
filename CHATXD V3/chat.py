import random
import json
import torch
from model import NeuralNet
from nltk_utils import vector_bolsa_de_palabras, dividir_en_palabras, detectar_idioma
import re
import random
from lark import Lark, Transformer, UnexpectedCharacters, UnexpectedToken, UnexpectedInput
import spacy
import unicodedata
import tkinter as tk
from tkinter import scrolledtext
import os
import sys

# Configurar dispositivo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Almacenamiento temporal del último bucle
ultimo_bucle = {"codigo": "", "tipo": ""}

# Cargar datos del modelo entrenado
def cargar_datos_modelo(ruta_intents, ruta_modelo):
    with open(ruta_intents, 'r', encoding='utf-8') as archivo:
        intenciones = json.load(archivo)


    # Mapea al CPU si no hay GPU disponible
    modelo_entrenado = torch.load(ruta_modelo, map_location=torch.device('cpu'))

    # Crear un diccionario de contextos
    contextos = {}
    for intencion in intenciones["intents"]:
        tag = intencion["tag"]
        contextos[tag] = {
            "filter": intencion.get("context_filter", None),
            "set": intencion.get("context_set", None)
        }

    return (
        intenciones,
        modelo_entrenado["input_size"],
        modelo_entrenado["hidden_size"],
        modelo_entrenado["output_size"],
        modelo_entrenado["all_words"],
        modelo_entrenado["tags"],
        modelo_entrenado["model_state"],
        contextos
    )


# Cargar modelo y datos necesarios
def cargar_modelo(input_size, hidden_size, output_size, model_state):
    modelo = NeuralNet(input_size, hidden_size, output_size).to(dispositivo)
    modelo.load_state_dict(model_state)
    modelo.eval()
    return modelo

# Detección de palabras clave
def detectar_palabras_clave(texto_usuario, intenciones):
    palabras_usuario = set(texto_usuario.lower().split())
    for intencion in intenciones["intents"]:
        for patron in intencion["patterns"]:
            palabras_patron = set(patron.lower().split())
            # Verificar intersección de palabras clave (al menos 2 palabras deben coincidir)
            if len(palabras_usuario & palabras_patron) >= 2:
                return intencion["tag"]
    return None



def cargar_modelo_spacy(modelo="es_core_news_sm"):
    try:
        if hasattr(sys, '_MEIPASS'):
            base_path = os.path.join(sys._MEIPASS, modelo)
        else:
            base_path = modelo

        nlp = spacy.load(base_path)
        return nlp
    except OSError:
        print(f"Error: No se encontró el modelo '{modelo}'. Descargándolo...")
        os.system(f"python -m spacy download {modelo}")
        return spacy.load(modelo)

nlp = cargar_modelo_spacy()

def generar_codigo_for(entrada_usuario):
    global ultimo_bucle
    numeros = re.findall(r'\d+', entrada_usuario)
    anidar = "anidar" in entrada_usuario.lower() or "dentro" in entrada_usuario.lower()

    # Depuración
    print(f"[DEBUG] Último bucle antes de procesar: {ultimo_bucle}")

    if len(numeros) == 2:
        inicio, fin = int(numeros[0]), int(numeros[1])

        if ultimo_bucle["codigo"] and anidar:
            print(f"[DEBUG] Anidando dentro del último bucle. Tipo: {ultimo_bucle['tipo']}")

            codigo_base = re.sub(r'```python|```', '', ultimo_bucle["codigo"])
            lineas = codigo_base.split('\n')

            nivel_indentacion = 0
            for i in range(len(lineas) - 1, -1, -1):
                if "for" in lineas[i]:
                    nivel_indentacion = lineas[i].find("for") + 4
                    break

            niveles = ["j", "k", "m", "n", "p", "q"]
            usado = re.findall(r'for (\w) in', codigo_base)
            variable_nueva = niveles[len(usado) % len(niveles)]

            nueva_indentacion = ' ' * nivel_indentacion
            nuevo_bucle = (
                f"{nueva_indentacion}for {variable_nueva} in range({inicio}, {fin+1}):\n"
                f"{nueva_indentacion}    print(f'Iteración {variable_nueva}={{ {variable_nueva} }}')"
            )

            for i in range(len(lineas) - 1, -1, -1):
                if "print(i)" in lineas[i] or "print(j)" in lineas[i]:
                    lineas.insert(i, nuevo_bucle)
                    break

            codigo_anidado = '\n'.join(lineas)
            print(f"[DEBUG] Código anidado generado:\n{codigo_anidado}")

            ultimo_bucle = {"codigo": f"```python\n{codigo_anidado}\n```", "tipo": "for_anidado"}
            return f"Aquí tienes el bucle anidado dentro del último ciclo:\n\n```python\n{codigo_anidado}\n```"

        else:
            codigo = f"for i in range({inicio}, {fin+1}):\n    print(i)"
            print(f"[DEBUG] Bucle básico generado:\n{codigo}")

            ultimo_bucle = {"codigo": f"```python\n{codigo}\n```", "tipo": "for_simple"}
            return f"Aquí tienes un bucle for de {inicio} a {fin}:\n\n```python\n{codigo}\n```"

    return "Por favor, proporciona un rango. Ejemplo: 'bucle for de 5 a 10'."

gramatica = """
    start: (if_stmt | elif_stmt | else_stmt)+
    
    if_stmt: "if" expr ":" suite
    elif_stmt: "elif" expr ":" suite
    else_stmt: "else" ":" suite
    
    expr: NAME OP (NUMBER | NAME | STRING)
        | "len(" NAME ")" OP NUMBER
        | expr "and" expr  -> and_expr
        | expr "or" expr   -> or_expr
        | NAME OP NUMBER

    suite: (INDENT statement+ DEDENT) | statement
    statement: "print(" STRING ")"
    
    STRING: /'[^']*'|\"[^\"]*\"/
    OP: ">" | "<" | ">=" | "<=" | "==" | "!="
    
    %import common.CNAME -> NAME
    %import common.INT -> NUMBER
    %import common.WS
    %ignore WS

    INDENT: "    "
    DEDENT: /(?!\\A)\\n(?=[^\\s])/
"""



class CodeGenerator(Transformer):
    def __init__(self):
        self.codigo_actual = []

    def start(self, items):
        # Une todas las partes del árbol en código legible
        return "\n".join(items)

    def if_stmt(self, items):
        condition, suite = items
        return f"if {condition}:\n{suite}"
    
    def elif_stmt(self, items):
        condition, suite = items
        return f"elif {condition}:\n{suite}"
    
    def else_stmt(self, items):
        suite = items[0]
        return f"else:\n{suite}"
    
    def expr(self, items):
        return " ".join(str(i) for i in items)
    
    def and_expr(self, items):
        return f"({items[0]} and {items[1]})"
    
    def or_expr(self, items):
        return f"({items[0]} or {items[1]})"
    
    def suite(self, items):
        return "".join(items)
    
    def statement(self, items):
        return f"    print({items[0]})\n"


def preprocesar_instruccion(entrada_usuario):
    """
    Detecta lo que el usuario pide (if, elif, else, mayor que, menor que, igual a)
    y retorna un bloque de código Python correspondiente.
    """
    entrada_usuario = entrada_usuario.lower()

    # 1) Detectar 'elif' primero para no chocar con "igual a", "mayor que", etc.
    if "elif" in entrada_usuario:
        if "mayor que" in entrada_usuario:
            condicion = entrada_usuario.split("mayor que")[-1].strip()
            return f"elif i > {condicion}:\n    print('i es mayor que {condicion}')"
        elif "menor que" in entrada_usuario:
            condicion = entrada_usuario.split("menor que")[-1].strip()
            return f"elif i < {condicion}:\n    print('i es menor que {condicion}')"
        elif "igual a" in entrada_usuario:
            condicion = entrada_usuario.split("igual a")[-1].strip()
            return f"elif i == {condicion}:\n    print('i es igual a {condicion}')"
        # Caso por defecto si no detectamos frase concreta
        return "elif i == 0:\n    print('i es igual a 0')"

    # 2) Si no es 'elif', buscamos if con mayor/menor/igual
    if "mayor que" in entrada_usuario:
        condicion = entrada_usuario.split("mayor que")[-1].strip()
        return f"if i > {condicion}:\n    print('i es mayor que {condicion}')"
    
    if "menor que" in entrada_usuario:
        condicion = entrada_usuario.split("menor que")[-1].strip()
        return f"if i < {condicion}:\n    print('i es menor que {condicion}')"
    
    if "igual a" in entrada_usuario:
        condicion = entrada_usuario.split("igual a")[-1].strip()
        return f"if i == {condicion}:\n    print('i es igual a {condicion}')"
    
    # 3) Manejar 'else'
    if "else" in entrada_usuario:
        return "else:\n    print('No se cumple la condición')"
    
    # Si no se detecta nada, devolver la entrada original
    return entrada_usuario



def generar_codigo_if(entrada_usuario):
    """
    Genera y combina bloques if, elif y else, reusando la lógica
    de 'preprocesar_instruccion' y el parser Lark.
    """
    global ultimo_bucle

    print(f"[DEBUG] Entrada recibida: {entrada_usuario}")
    
    # Preprocesar la instrucción
    codigo_procesado = preprocesar_instruccion(entrada_usuario)
    print(f"[DEBUG] Entrada procesada para Lark: {codigo_procesado}")
    
    parser = Lark(gramatica, parser='lalr', transformer=CodeGenerator(), debug=True)
    
    try:
        print("[DEBUG] Iniciando análisis sintáctico...")
        arbol = parser.parse(codigo_procesado)
        
        # Generar el código a partir del árbol
        codigo_generado = "".join(arbol)
        print(f"[DEBUG] Código generado sin elif/else:\n{codigo_generado}")
        
        # Si el usuario ha pedido un elif o un else, combinamos con el código anterior
        if "else" in entrada_usuario.lower() or "elif" in entrada_usuario.lower():
            if ultimo_bucle.get("codigo"):
                # 1) Quitar las marcas de bloque
                codigo_base = re.sub(r'```python|```', '', ultimo_bucle["codigo"])
                # 2) Separar en líneas y eliminar vacías
                lineas = [l for l in codigo_base.split('\n') if l.strip()]

                # 3) Insertar el nuevo bloque (elif/else) donde corresponda
                codigo_generado = codigo_generado.strip('\n')  # Asegúrate de limpiar

                def get_indentation(line):
                    return len(line) - len(line.lstrip())

                es_elif = codigo_generado.startswith("elif ")
                es_else = codigo_generado.startswith("else:")

                inserted = False
                # Recorre de abajo hacia arriba para ubicar if/elif/else
                for i in range(len(lineas) - 1, -1, -1):
                    linea_actual = lineas[i].rstrip()

                    if es_elif:
                        # Si encuentras 'else:' antes, insertas justo arriba de ese 'else:'
                        if linea_actual.strip().startswith("else:"):
                            indent = get_indentation(linea_actual)
                            lineas.insert(i, ' ' * indent + codigo_generado)
                            inserted = True
                            break
                        # Si encuentras 'if' o 'elif', insertas justo abajo
                        if linea_actual.strip().startswith("if ") or linea_actual.strip().startswith("elif"):
                            indent = get_indentation(linea_actual)
                            lineas.insert(i + 1, ' ' * indent + codigo_generado)
                            inserted = True
                            break

                    elif es_else:
                        # Si ya existe otro 'else', lo agregamos antes; si no, al final
                        if linea_actual.strip().startswith("else:"):
                            indent = get_indentation(linea_actual)
                            lineas.insert(i, ' ' * indent + codigo_generado)
                            inserted = True
                            break

                # Si no se insertó en el bucle, va al final
                if not inserted:
                    lineas.append(codigo_generado)

                # 4) Eliminar líneas vacías otra vez
                lineas = [l for l in lineas if l.strip()]

                # 5) Unir líneas en un solo bloque
                codigo_generado = '\n'.join(lineas)

                print(f"[DEBUG] Código combinado con bloque anterior:\n{codigo_generado}")



        print(f"[DEBUG] Código final generado:\n{codigo_generado}")
        
        # Actualizar la variable global con el código completo
        ultimo_bucle = {"codigo": f"```python\n{codigo_generado}\n```", "tipo": "if"}
        return f"Aquí tienes el bloque de código:\n\n```python\n{codigo_generado}\n```"
    
    except (UnexpectedCharacters, UnexpectedToken, UnexpectedInput) as e:
        print(f"[DEBUG] Error de Lark detectado: {type(e).__name__}")
        print(f"[DEBUG] Contexto del error:\n{e.get_context(codigo_procesado)}")
        return "Lo siento, hubo un error de sintaxis en el bloque if. Revisa tu entrada."
    
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error completo:\n{traceback.format_exc()}")
        return "Lo siento, no pude generar un bloque if válido. Verifica la sintaxis e inténtalo nuevamente."

try:
    # Verifica si se está ejecutando como binario PyInstaller
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        # Si no es un binario, buscar el modelo en el entorno local (site-packages)
        base_path = os.path.abspath(".")

    # Intenta cargar el modelo desde el directorio del ejecutable
    modelo_path = os.path.join(base_path, 'es_core_news_sm')
    
    if not os.path.exists(modelo_path):
        # Si no se encuentra en el directorio local, buscar en site-packages
        modelo_path = 'es_core_news_sm'
        
    nlp = spacy.load(modelo_path)

except OSError:
    print("Error: No se encontró el modelo 'es_core_news_sm'. Asegúrate de que está instalado.")
    exit(1)


def extraer_condicion_natural_spacy(entrada_usuario):
    doc = nlp(entrada_usuario.lower())
    
    variable = None
    operador = None
    valor = None
    mensaje_if = None
    mensaje_else = None

    # --- 1) Detectar "variable" o "numero" ---
    for i, token in enumerate(doc):
        if token.text in ["variable", "numero"]:
            # Verificamos si la siguiente palabra es comilla
            if i+2 < len(doc) and doc[i+1].text in ['"', "'"]:
                variable = doc[i+2].text
            else:
                if i+1 < len(doc):
                    variable = doc[i+1].text
                else:
                    variable = None

            # Limpieza de comillas
            if variable:
                variable = variable.replace('"', '').replace("'", "")

    # --- 2) Detectar operador ---
    entrada_lower = entrada_usuario.lower()

    if "mayor o igual que" in entrada_lower or "mayor o igual a" in entrada_lower:
        operador = ">="
    elif "menor o igual que" in entrada_lower or "menor o igual a" in entrada_lower:
        operador = "<="
    elif ("diferente a" in entrada_lower 
        or "distinto a" in entrada_lower
        or "distinto de" in entrada_lower):
        operador = "!="
    elif "mayor que" in entrada_lower or "mayor a" in entrada_lower:
        operador = ">"
    elif "menor que" in entrada_lower or "menor a" in entrada_lower:
        operador = "<"
    elif "igual a" in entrada_lower:
        operador = "=="


    # --- 3) Detectar valor (número) ---
    num_list = [t.text for t in doc if t.like_num]

    # Si la "variable" es un número y está en esa lista, la removemos
    # para no volver a usarla como valor
    if variable and variable.isdigit() and variable in num_list:
        num_list.remove(variable)

    # Ahora, si todavía queda al menos un número en la lista,
    # tomamos el primero como el "valor"
    if len(num_list) > 0:
        valor = num_list[0]
    else:
        valor = None

    # --- 4) Detectar mensajes "imprima" / "imprime" ---
    import re
    patron_prints = re.findall(r'(?:imprime|imprima)\s+"([^"]+)"', entrada_lower)
    # El patrón busca:  'imprime' o 'imprima', espacio, comillas, luego el texto, cierra comillas
    # Ej: 'imprima "es mayor"'

    if len(patron_prints) == 1:
        mensaje_if = patron_prints[0]
    elif len(patron_prints) >= 2:
        mensaje_if = patron_prints[0]
        mensaje_else = patron_prints[1]

    # 5) Si no encontramos uno de los mensajes, ponemos valores por defecto
    if not mensaje_if:
        if operador in [">", ">=", "!="]:
            mensaje_if = "es mayor"
        else:
            mensaje_if = "cumple la condición"

    if not mensaje_else:
        if operador in [">", ">=", "!="]:
            mensaje_else = "es menor"
        else:
            mensaje_else = "no cumple la condición"

    # 6) Validar que variable, operador y valor estén presentes
    if not variable or not operador or not valor:
        return None

    return {
        "variable": variable,
        "operador": operador,
        "valor": valor,
        "mensaje_if": mensaje_if,
        "mensaje_else": mensaje_else
    }



def generar_codigo_if_natural(entrada_usuario):
    datos = extraer_condicion_natural_spacy(entrada_usuario)
    if not datos:
        return "No pude entender bien la condición. Asegúrate de especificar variable, operador, valor y mensajes."

    variable = datos["variable"]
    operador = datos["operador"]
    valor = datos["valor"]
    mensaje_if = datos["mensaje_if"]
    mensaje_else = datos["mensaje_else"]
    
    # Construir el bloque de código en Python
    codigo = f"""if {variable} {operador} {valor}:
    print("{mensaje_if}")
else:
    print("{mensaje_else}")"""
    
    return f"Aquí tienes tu bloque de código:\n\n```python\n{codigo}\n```"

ALGORITMOS_CODIGO = {
    # Ordenamientos
    "burbuja": "bubble_sort",
    "bubble sort": "bubble_sort",
    "ordenamiento burbuja": "bubble_sort",
    "ordenar por burbuja": "bubble_sort",
    "bubble": "bubble_sort",
    "sort by bubble": "bubble_sort",
    "perform bubble sort": "bubble_sort",
    "do bubble sort": "bubble_sort",
    "use bubble sort": "bubble_sort",

    "quicksort": "quick_sort",
    "insercion": "insertion_sort",
    "insertion sort": "insertion_sort",
    "seleccion": "selection_sort",
    "selection sort": "selection_sort",
    

    # Factorial - Iterativo
    "factorial_iterativo": "factorial_iterative",
    "iterative factorial": "factorial_iterative",
    "factorial por iteracion": "factorial_iterative",
    "calcula el factorial iterativo": "factorial_iterative",
    "factorial iterativo": "factorial_iterative",
    "factorial by iteration": "factorial_iterative",
    "find factorial iteratively": "factorial_iterative",
    
    # Factorial - Recursivo
    "factorial_recursivo": "factorial_recursive",
    "recursive factorial": "factorial_recursive",
    "factorial por recursion": "factorial_recursive",
    "calcula el factorial recursivo": "factorial_recursive",
    "factorial recursivo": "factorial_recursive",
    "factorial by recursion": "factorial_recursive",
    "find factorial recursively": "factorial_recursive",
    
    # Factorial - General
    "factorial": "factorial_general",
    "calcula el factorial": "factorial_general",
    "calculate factorial": "factorial_general",
    "factorial of": "factorial_general",
    "find factorial": "factorial_general",
    "compute factorial": "factorial_general",
    
    # Fibonacci - Iterativo
    "fibonacci_iterativo": "fibonacci_iterative",
    "iterative fibonacci": "fibonacci_iterative",
    "fibonacci por iteracion": "fibonacci_iterative",
    "calcula el fibonacci iterativo": "fibonacci_iterative",
    "fibonacci iterativo": "fibonacci_iterative",
    "fibonacci by iteration": "fibonacci_iterative",
    "find fibonacci iteratively": "fibonacci_iterative",
    "iterative calculation of fibonacci": "fibonacci_iterative",
    "compute iterative fibonacci": "fibonacci_iterative",
    "iterative fibonacci of": "fibonacci_iterative",
    "calculate iterative fibonacci": "fibonacci_iterative",

    # Fibonacci - Recursivo
    "fibonacci_recursivo": "fibonacci_recursive",
    "recursive fibonacci": "fibonacci_recursive",
    "fibonacci por recursion": "fibonacci_recursive",
    "calcula el fibonacci recursivo": "fibonacci_recursive",
    "fibonacci recursivo": "fibonacci_recursive",
    "fibonacci by recursion": "fibonacci_recursive",
    "find fibonacci recursively": "fibonacci_recursive",
    "recursive calculation of fibonacci": "fibonacci_recursive",
    "compute recursive fibonacci": "fibonacci_recursive",
    "recursive fibonacci of": "fibonacci_recursive",
    "calculate recursive fibonacci": "fibonacci_recursive",

    # Fibonacci - General (sin especificar recursivo o iterativo)
    "fibonacci": "fibonacci_general",
    "serie de fibonacci": "fibonacci_general",
    "find fibonacci": "fibonacci_general",
    "calculate fibonacci": "fibonacci_general",
    "compute fibonacci": "fibonacci_general",
    "fibonacci of": "fibonacci_general",
    "calculate the fibonacci of": "fibonacci_general",
    
    # Número primo
    "primo": "prime_number",
    "número primo": "prime_number",
    "es primo": "prime_number",
    "verificar primo": "prime_number",
    "prime": "prime_number",
    "is prime": "prime_number",
    "check prime": "prime_number",
    "prime number": "prime_number",
    "find prime": "prime_number",
    "calculate prime": "prime_number",
    
    # MCD - Iterativo
    "mcd_iterativo": "gcd_iterative",
    "iterative gcd": "gcd_iterative",
    "calculate gcd iteratively": "gcd_iterative",
    "find gcd iteratively": "gcd_iterative",
    "mcd por iteracion": "gcd_iterative",
    "máximo común divisor iterativo": "gcd_iterative",
    "iterative greatest common divisor": "gcd_iterative",
    "iterative euclidean algorithm": "gcd_iterative",
    "gcd by iteration": "gcd_iterative",
    
    # MCD - Recursivo
    "mcd_recursivo": "gcd_recursive",
    "recursive gcd": "gcd_recursive",
    "calculate gcd recursively": "gcd_recursive",
    "find gcd recursively": "gcd_recursive",
    "mcd por recursion": "gcd_recursive",
    "mcd recursivo": "gcd_recursive",
    "máximo común divisor recursivo": "gcd_recursive",
    "recursive greatest common divisor": "gcd_recursive",
    "recursive euclidean algorithm": "gcd_recursive",
    "gcd by recursion": "gcd_recursive",
    
    # MCD - General
    "mcd": "gcd_general",
    "gcd": "gcd_general",
    "find gcd": "gcd_general",
    "calculate gcd": "gcd_general",
    "greatest common divisor": "gcd_general",
    "euclidean algorithm": "gcd_general",
    "máximo común divisor": "gcd_general",
    "maximo comun divisor": "gcd_general",

    # Mínimo común múltiplo
    "mcm": "lcm",
    "lcm": "lcm",
    "mínimo común múltiplo": "lcm",
    "least common multiple": "lcm",
    "least common multiple of": "lcm",
    "find lcm": "lcm",
    "calcula el mcm": "lcm",
    "calcula el minimo comun multiplo": "lcm",
    "find the least common multiple of": "lcm",
    "calculate the least common multiple of": "lcm",
    
    # Potenciación
    "potenciacion": "fast_exponentiation",
    "exponentiation": "fast_exponentiation",
    "fast exponentiation": "fast_exponentiation",
    "power": "fast_exponentiation",
    
    # Contar dígitos
    "cuenta los digitos": "contar_digitos", 
    "cuenta los dígitos": "contar_digitos",
    "contar_digitos": "digit_count",
    "count digits": "digit_count",
    "count the digits": "digit_count",
    "count the digits of ": "digit_count",
    "cuenta los dígitos": "digit_count",
    "cuenta los digitos": "digit_count",
    "cuenta dígitos": "digit_count",
    "cuenta digitos": "digit_count",

    # Suma de elementos de una lista
    "suma_lista": "sum_list",
    "sumar elementos de la lista": "sum_list",
    "suma de lista": "sum_list",
    "dame la suma de la lista": "sum_list",
    "calcula la suma de la lista": "sum_list",
    "suma de esta lista": "sum_list",
    "podrías calcular la suma de la lista": "sum_list",
    "me puedes dar la suma de la lista": "sum_list",
    "puedes calcular la suma de la lista": "sum_list",
    "sum elements of list": "sum_list",
    "sum of list": "sum_list",
    "calculate sum of list": "sum_list",
    "add all elements in list": "sum_list",
    "total sum of list": "sum_list",
    
    
    # Promedio de una lista
    "promedio_lista": "average_list",
    "calcula el promedio de la lista": "average_list",
    "promedio de lista": "average_list",
    "dame el promedio de la lista": "average_list",
    "promedio de esta lista": "average_list",
    "average elements of list": "average_list",
    "average of list": "average_list",
    "calculate average of list": "average_list",
    "get average of list": "average_list",
    
    # Máximo y mínimo de una lista
    "max_min_lista": "max_min_list",
    "máximo y mínimo de la lista": "max_min_list",
    "máximo y mínimo": "max_min_list",
    "máximo mínimo lista": "max_min_list",
    "encontrar el máximo y mínimo": "max_min_list",
    "find max and min": "max_min_list",
    "maximum and minimum in list": "max_min_list",
    "calculate max and min": "max_min_list",
    "get max and min from list": "max_min_list",
    "highest and lowest values": "max_min_list",
    "biggest and smallest in list": "max_min_list",
    "find the maximum and minimum": "max_min_list",
    "get the maximum and minimum of": "max_min_list",
    "calculate the maximum and minimum of": "max_min_list",
    "find the biggest and smallest of": "max_min_list",
    
    # Búsqueda lineal
    "busqueda_lineal": "linear_search",
    "buscar elemento en lista": "linear_search",
    "buscar en lista": "linear_search",
    "buscar": "linear_search",
    "busca": "linear_search",
    "encontrar elemento en lista": "linear_search",
    "búsqueda lineal": "linear_search",
    "búscame el elemento en la lista": "linear_search",
    "buscar número en lista": "linear_search",
    "buscar un número en la lista": "linear_search",
    "busca el elemento en la lista": "linear_search",
    "localiza elemento en lista": "linear_search",
    "localiza el número en la lista": "linear_search",
    "linear search": "linear_search",
    "find element in list": "linear_search",
    "search element in list": "linear_search",
    "search in list": "linear_search",
    "look for element in list": "linear_search",
    "locate element in list": "linear_search",
    "search for number in list": "linear_search",
    "find the number in list": "linear_search",
    "locate the element in list": "linear_search",
    
    # Invertir lista
    "invertir_lista": "reverse_list",
    "invertir lista": "reverse_list",
    "reversa de lista": "reverse_list",
    "invierte la lista": "reverse_list",
    "lista invertida": "reverse_list",
    "da la vuelta a la lista": "reverse_list",
    "voltea la lista": "reverse_list",
    "reverse list": "reverse_list",
    "invert list": "reverse_list",
    "invert list elements": "reverse_list",
    "reverse the list": "reverse_list",
    "flip the list": "reverse_list",
    "turn the list around": "reverse_list",
    "list reversed": "reverse_list",
    
    # Eliminar duplicados
    "eliminar_duplicados": "remove_duplicates",
    "eliminar duplicados de lista": "remove_duplicates",
    "quita duplicados de lista": "remove_duplicates",
    "elimina duplicados de lista": "remove_duplicates",
    "elimina los duplicados de la lista": "remove_duplicates",
    "borra duplicados de lista": "remove_duplicates",
    "sin duplicados en lista": "remove_duplicates",
    "remove duplicates from list": "remove_duplicates",
    "delete duplicates in list": "remove_duplicates",
    "get unique values from list": "remove_duplicates",
    "filter out duplicates": "remove_duplicates",
    "no duplicates in list": "remove_duplicates",
    "remove repeated elements": "remove_duplicates",
    "deduplicate list": "remove_duplicates",
    "elimina duplicados": "remove_duplicates",
    "quita los duplicados": "remove_duplicates",
    "borra los duplicados": "remove_duplicates",
    "remove the duplicates": "remove_duplicates",
    "delete repeated items": "remove_duplicates"

}

def preprocesar_instruccion(entrada_usuario):
    # Normaliza a minúsculas y elimina tildes
    entrada_usuario = entrada_usuario.lower()
    entrada_usuario = ''.join(
        c for c in unicodedata.normalize('NFD', entrada_usuario)
        if unicodedata.category(c) != 'Mn'
    )
    return entrada_usuario

def detectar_ordenamiento_numeros(entrada_usuario):
    nombre_bot = "ChatBot"

    # Aplicar preprocesamiento (eliminar tildes y pasar a minúsculas)
    entrada_usuario = preprocesar_instruccion(entrada_usuario)
    doc = nlp(entrada_usuario.lower())
    numeros = [token.text for token in doc if token.like_num]
    
    # Depuración
    print("[DEBUG] detectar_ordenamiento_numeros => entrada_usuario.lower():", entrada_usuario.lower())
    print("[DEBUG] Numeros detectados =>", numeros)

    # ------ Filtro conceptual ------
    if any(frase in entrada_usuario for frase in ["por que", "porque", "para que", "para qué", "que hace"]):
        print("[DEBUG] -> Se detecta pregunta conceptual (por qué, para qué). No generamos código, pero dejamos seguir.")
        # No retornamos nada, para que el flujo continúe y pueda pasar a intenciones
        pass

    # ------ Filtro: ¿pide el código de forma explícita? ------
    # Puedes ajustar estas palabras a tu conveniencia:
    palabras_clave_codigo = ["codigo", "algoritmo", "generame", "código", 
                             "generar", "genérame", "genera"]
    
    # Si no contiene ninguna de esas palabras, no generamos código.
    if not any(pkc in entrada_usuario for pkc in palabras_clave_codigo):
        print("[DEBUG] -> El usuario NO pidió el código explícitamente. No generamos burbuja.")
        return None

    # Detectar lenguaje solicitado
    if "python" in entrada_usuario.lower():
        lenguaje = "python"
    elif "javascript" in entrada_usuario.lower() or "js" in entrada_usuario.lower():
        lenguaje = "javascript"
    else:
        lenguaje = "python"
    
    # ------ Búsqueda del algoritmo en la frase ------
    for algoritmo in ALGORITMOS_CODIGO:
        if algoritmo in entrada_usuario.lower() and numeros:
            print("[DEBUG] -> COINCIDE con", algoritmo)
            respuesta = extraer_lista_numeros_spacy(entrada_usuario, lenguaje, algoritmo)
            if respuesta:
                print(f"{nombre_bot}: {respuesta}")
                return respuesta
    
    print("[DEBUG] -> No coincidió ningún algoritmo. Regreso None")
    return None




ALGORITMOS_CODIGO_CADENAS = {
    # Revertir cadena
    "revertir cadena": "reverse_string",
    "invertir texto": "reverse_string",
    "reverse string": "reverse_string",
    "al revés": "reverse_string",
    "invertir cadena": "reverse_string",
    "invierte": "reverse_string",
    "invierte esta cadena": "reverse_string",
    "invertir esta oración": "reverse_string",
    "reverse string": "reverse_string",
    "flip text": "reverse_string",
    "invert sentence": "reverse_string",
    "reverse phrase": "reverse_string",

        # Convertir a mayúsculas
    "poner en mayusculas": "to_uppercase",
    "convertir a mayusculas": "to_uppercase",
    "uppercase": "to_uppercase",
    "poner todo en mayúsculas": "to_uppercase",
    "convertir texto a mayúsculas": "to_uppercase",
    "capitalizar texto": "to_uppercase",
    "capitalize text": "to_uppercase",
    "make uppercase": "to_uppercase",
    "convert text to uppercase": "to_uppercase",
    "set string to uppercase": "to_uppercase",
    "poner frase en mayúsculas": "to_uppercase",
    "all caps text": "to_uppercase",
    
    # Contar vocales
    "contar vocales": "count_vowels",
    "cuantas vocales tiene": "count_vowels",
    "count vowels": "count_vowels",
    "vowel count": "count_vowels",
    "vowels in string": "count_vowels",
    "how many vowels": "count_vowels",
    "calculate vowels in text": "count_vowels",
    "cuantas vocales hay": "count_vowels",
    "check vowels in sentence": "count_vowels",
    "vowel total in text": "count_vowels",
    "vowel count for string": "count_vowels",
    
    # Contar consonantes
    "contar consonantes": "count_consonants",
    "cuantas consonantes tiene": "count_consonants",
    "count consonants": "count_consonants",
    "consonant count": "count_consonants",
    "consonants in string": "count_consonants",
    "how many consonants": "count_consonants",
    "calculate consonants in text": "count_consonants",
    "cuantas consonantes hay": "count_consonants",
    "check consonants in sentence": "count_consonants",
    "consonant total in text": "count_consonants",
    "consonant count for string": "count_consonants",
    
    # Verificar palíndromo
    "es palindromo": "check_palindrome",
    "verificar palindromo": "check_palindrome",
    "palindrome check": "check_palindrome",
    "is palindrome": "check_palindrome",
    "ver si es palindromo": "check_palindrome",
    "es esta frase palindromo": "check_palindrome",
    "is this a palindrome": "check_palindrome",
    "check if text is palindrome": "check_palindrome",
    "detect palindrome": "check_palindrome",
    "palindrome verification": "check_palindrome",
    "palindrome checker": "check_palindrome",
    
    # Eliminar espacios
    "eliminar espacios": "remove_spaces",
    "quitar espacios": "remove_spaces",
    "remove spaces": "remove_spaces",
    "delete spaces": "remove_spaces",
    "strip spaces from text": "remove_spaces",
    "erase spaces": "remove_spaces",
    "remove spaces from sentence": "remove_spaces",
    "clean text from spaces": "remove_spaces",
    "remove blank spaces": "remove_spaces",
    "eliminar espacios de frase": "remove_spaces",
    "get rid of spaces": "remove_spaces",
    
    # Contar palabras
    "contar palabras": "count_words",
    "cuantas palabras tiene": "count_words",
    "count words": "count_words",
    "word count": "count_words",
    "number of words": "count_words",
    "how many words in string": "count_words",
    "palabra total en texto": "count_words",
    "words in sentence": "count_words",
    "calcular palabras en texto": "count_words",
    "conteo de palabras": "count_words",
    "cuantas palabras contiene": "count_words",
    
    # Reemplazar palabras
    "reemplazar palabra": "replace_word",
    "sustituir palabra": "replace_word",
    "replace word": "replace_word",
    "replace text": "replace_word",
    "change word in string": "replace_word",
    "replace this word": "replace_word",
    "replace term in sentence": "replace_word",
    "swap words": "replace_word",
    "replace text with new one": "replace_word",
    "intercambiar palabras": "replace_word",
    "modificar palabra": "replace_word",
    

}


def detectar_operacion_cadenas(entrada_usuario):
    nombre_bot = "ChatBot"

    # Preprocesamiento
    entrada_usuario = preprocesar_instruccion(entrada_usuario)
    
    # Depuración
    print("[DEBUG] detectar_operacion_cadenas => entrada_usuario.lower():", entrada_usuario.lower())

    # --------------------
    # Detectar lenguaje solicitado
    # --------------------
    if "python" in entrada_usuario.lower():
        lenguaje = "python"
    elif "javascript" in entrada_usuario.lower() or "js" in entrada_usuario.lower():
        lenguaje = "javascript"
    else:
        lenguaje = "python"

    print(f"[DEBUG] -> Lenguaje detectado: {lenguaje}")

    # --------------------
    # Mapeo Directo de Algoritmos (Exacta y Flexible)
    # --------------------
    MAPEO_DIRECTO = {
        "contar vocales": "count_vowels",
        "cuantas vocales tiene": "count_vowels",
        "contar consonantes": "count_consonants",
        "cuantas consonantes tiene": "count_consonants",
        "es palindromo": "check_palindrome",
        "verificar palindromo": "check_palindrome",
        "eliminar espacios": "remove_spaces",
        "contar palabras": "count_words",
        "reemplazar palabra": "replace_word",
        "poner en mayusculas": "to_uppercase",
        "invertir texto": "reverse_string",
        "invertir cadena": "reverse_string",
        "invertir la cadena": "reverse_string",


        # Inglés
        "count vowels": "count_vowels",
        "how many vowels": "count_vowels",
        "count consonants": "count_consonants",
        "how many consonants": "count_consonants",
        "is palindrome": "check_palindrome",
        "check palindrome": "check_palindrome",
        "how many words": "count_words",
        "remove spaces": "remove_spaces",
        "remove spaces from": "remove_spaces",
        "count words": "count_words",
        "replace word": "replace_word",
        "make uppercase": "to_uppercase",
        "reverse text": "reverse_string",
    }

    # Mapeo Flexible (Permite palabras en medio)
    MAPEO_FLEXIBLE = {
        r"contar\s+\w*\s*palabras": "count_words",  # Detecta "contar palabras", "contar las palabras"
        r"contar\s+\w*\s*vocales": "count_vowels",
        r"contar\s+\w*\s*consonantes": "count_consonants",
        r"remove\s+\w*\s*spaces": "remove_spaces",
        r"how many\s+\w*\s*words": "count_words",
        r"reemplazar\s+['\"].*?['\"]\s+por\s+['\"].*?['\"]": "replace_word",
        r"replace\s+['\"].*?['\"]\s+with\s+['\"].*?['\"]": "replace_word",
    }

    # 1. Buscar coincidencia exacta primero
    algoritmo_detectado = None
    for clave in MAPEO_DIRECTO.keys():
        if clave in entrada_usuario.lower():
            algoritmo_detectado = MAPEO_DIRECTO[clave]
            break

    # 2. Si no hay coincidencia exacta, probar con mapeo flexible
    if not algoritmo_detectado:
        for patron, algoritmo in MAPEO_FLEXIBLE.items():
            if re.search(patron, entrada_usuario.lower()):
                algoritmo_detectado = algoritmo
                break

    # --------------------
    # Validación de Algoritmo Detectado
    # --------------------
    if algoritmo_detectado:
        print("[DEBUG] -> COINCIDE con operación de cadena:", algoritmo_detectado)

        # --------------------
        # Extracción de Texto
        # --------------------
        doc = nlp(entrada_usuario)
        texto_detectado = None

        # 1. Extraer texto entre comillas
        texto_comillas = re.findall(r'["\'](.*?)["\']', entrada_usuario)

        if texto_comillas:
            texto_detectado = texto_comillas[0]

        # 2. Si no hay texto entre comillas, buscar tras palabras clave
        if not texto_detectado:
            palabras_clave = ["cadena", "texto", "frase", "oración"]
            tokens = [token.text.lower() for token in doc]

            for i, token in enumerate(tokens):
                if token in palabras_clave and i < len(tokens) - 1:
                    texto_detectado = " ".join(tokens[i + 1:])
                    break

        # --------------------
        # Procesar Texto Detectado
        # --------------------
        if texto_detectado:
            cadena_a_procesar = texto_detectado.strip()
            print("[DEBUG] -> Cadena a procesar:", cadena_a_procesar)

            respuesta = procesar_cadena_spacy(cadena_a_procesar, lenguaje, algoritmo_detectado)
            print(f"{nombre_bot} ({lenguaje}):\n{respuesta}")
            return respuesta
        else:
            print(f"{nombre_bot}: No se detectó texto para {algoritmo_detectado}. ¿Puedes especificarlo?")
            return f"No se detectó texto para {algoritmo_detectado}. ¿Puedes especificarlo?"
    else:
        print("[DEBUG] -> No coincidió ninguna operación de cadena. Regreso None")
        return None




def procesar_cadena_spacy(entrada_usuario, lenguaje="python", algoritmo="revertir"):
    # Depuración
    print("[DEBUG] -> procesar_cadena_spacy")
    print("[DEBUG] -> Entrada del usuario:", entrada_usuario)
    print("[DEBUG] -> Algoritmo recibido:", algoritmo)
    
    # Si no hay entrada, retornamos None
    if not entrada_usuario:
        print("[DEBUG] -> Entrada vacía. Regreso None.")
        return None

    # Variable donde guardaremos el código generado
    codigo = ""

    # ---------------------
    # MAPEO DE ALGORITMOS
    # ---------------------
    MAPEO_ALGORITMOS = {
        "reverse_string": "revertir",
        "to_uppercase": "mayusculas",
        "string_length": "longitud",
        "contar vocales": "count_vowels",
        "cuantas vocales tiene": "count_vowels",
        "vowel count": "count_vowels",
        "count_vowels": "contar_vocales",
        "count_consonants": "contar_consonantes",
        "check_palindrome": "verificar_palindromo",
        "remove_spaces": "eliminar_espacios",
        "count_words": "contar_palabras",
        "replace_word": "reemplazar_palabra"
    }

    # Traducir el algoritmo a su equivalente esperado
    algoritmo = MAPEO_ALGORITMOS.get(algoritmo, algoritmo)

    # ---------------------
    # ALGORITMOS DE CADENAS
    # ---------------------
    if algoritmo in ["revertir", "reverse", "invertir"]:
        print("[DEBUG] -> Entró al if 'revertir'")
        if lenguaje == "python":
            codigo = f'''def revertir_cadena(cadena):
    return cadena[::-1]

cadena = "{entrada_usuario}"
cadena_revertida = revertir_cadena(cadena)
print("Cadena revertida:", cadena_revertida)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function revertirCadena(cadena) {{
    return cadena.split('').reverse().join('');
}}

let cadena = "{entrada_usuario}";
let cadenaRevertida = revertirCadena(cadena);
console.log("Cadena revertida:", cadenaRevertida);'''

    elif algoritmo in ["mayusculas", "uppercase"]:
        print("[DEBUG] -> Entró al if 'mayusculas'")
        if lenguaje == "python":
            codigo = f'''def convertir_mayusculas(cadena):
    return cadena.upper()

cadena = "{entrada_usuario}"
cadena_mayusculas = convertir_mayusculas(cadena)
print("Cadena en mayúsculas:", cadena_mayusculas)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function convertirMayusculas(cadena) {{
    return cadena.toUpperCase();
}}

let cadena = "{entrada_usuario}";
let cadenaMayusculas = convertirMayusculas(cadena);
console.log("Cadena en mayúsculas:", cadenaMayusculas);'''

    elif algoritmo in ["longitud", "length"]:
        print("[DEBUG] -> Entró al if 'longitud'")
        if lenguaje == "python":
            codigo = f'''def calcular_longitud(cadena):
    return len(cadena)

cadena = "{entrada_usuario}"
longitud = calcular_longitud(cadena)
print("Longitud de la cadena:", longitud)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function calcularLongitud(cadena) {{
    return cadena.length;
}}

let cadena = "{entrada_usuario}";
let longitud = calcularLongitud(cadena);
console.log("Longitud de la cadena:", longitud);'''
            
    # ---------------------
    # CONTAR VOCALES
    # ---------------------
    elif algoritmo in ["contar_vocales", "vowels"]:
        print("[DEBUG] -> Entró al if 'contar_vocales'")
        if lenguaje == "python":
            codigo = f'''def contar_vocales(cadena):
    return sum(1 for c in cadena.lower() if c in 'aeiouáéíóú')

cadena = "{entrada_usuario}"
total_vocales = contar_vocales(cadena)
print("Total de vocales:", total_vocales)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function contarVocales(cadena) {{
    return cadena.match(/[aeiouáéíóú]/gi)?.length || 0;
}}

let cadena = "{entrada_usuario}";
let totalVocales = contarVocales(cadena);
console.log("Total de vocales:", totalVocales);'''

    # ---------------------
    # CONTAR CONSONANTES
    # ---------------------
    elif algoritmo in ["contar_consonantes", "consonants"]:
        print("[DEBUG] -> Entró al if 'contar_consonantes'")
        if lenguaje == "python":
            codigo = f'''def contar_consonantes(cadena):
    return sum(1 for c in cadena.lower() if c.isalpha() and c not in 'aeiouáéíóú')

cadena = "{entrada_usuario}"
total_consonantes = contar_consonantes(cadena)
print("Total de consonantes:", total_consonantes)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function contarConsonantes(cadena) {{
    return cadena.match(/[^aeiouáéíóú\s\d\W]/gi)?.length || 0;
}}

let cadena = "{entrada_usuario}";
let totalConsonantes = contarConsonantes(cadena);
console.log("Total de consonantes:", totalConsonantes);'''

    # ---------------------
    # VERIFICAR PALÍNDROMO
    # ---------------------
    elif algoritmo in ["verificar_palindromo", "palindrome"]:
        print("[DEBUG] -> Entró al if 'verificar_palindromo'")
        if lenguaje == "python":
            codigo = f'''def es_palindromo(cadena):
    limpio = ''.join(c.lower() for c in cadena if c.isalnum())
    return limpio == limpio[::-1]

cadena = "{entrada_usuario}"
es_pal = es_palindromo(cadena)
print("¿Es palíndromo?:", es_pal)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function esPalindromo(cadena) {{
    let limpio = cadena.toLowerCase().replace(/[^a-z0-9áéíóú]/gi, '');
    return limpio === limpio.split('').reverse().join('');
}}

let cadena = "{entrada_usuario}";
let esPal = esPalindromo(cadena);
console.log("¿Es palíndromo?:", esPal);'''
            

    # ---------------------
    # ELIMINAR ESPACIOS
    # ---------------------
    elif algoritmo in ["eliminar_espacios", "remove_spaces"]:
        print("[DEBUG] -> Entró al if 'eliminar_espacios'")
        if lenguaje == "python":
            codigo = f'''def eliminar_espacios(cadena):
    return cadena.replace(" ", "")

cadena = "{entrada_usuario}"
cadena_sin_espacios = eliminar_espacios(cadena)
print("Cadena sin espacios:", cadena_sin_espacios)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function eliminarEspacios(cadena) {{
    return cadena.replace(/\\s+/g, '');
}}

let cadena = "{entrada_usuario}";
let cadenaSinEspacios = eliminarEspacios(cadena);
console.log("Cadena sin espacios:", cadenaSinEspacios);'''
            

    # ---------------------
    # CONTAR PALABRAS
    # ---------------------
    elif algoritmo in ["contar_palabras", "count_words"]:
        print("[DEBUG] -> Entró al if 'contar_palabras'")
        if lenguaje == "python":
            codigo = f'''def contar_palabras(cadena):
    return len(cadena.split())

cadena = "{entrada_usuario}"
total_palabras = contar_palabras(cadena)
print("Total de palabras:", total_palabras)'''
        
        elif lenguaje == "javascript":
            codigo = f'''function contarPalabras(cadena) {{
    return cadena.trim().split(/\\s+/).length;
}}

let cadena = "{entrada_usuario}";
let totalPalabras = contarPalabras(cadena);
console.log("Total de palabras:", totalPalabras);'''

    else:
        print("[DEBUG] -> Algoritmo no reconocido. Regreso None.")
        return None

    return f"// Lenguaje detectado: {lenguaje}\n{codigo}"


def extraer_lista_numeros_spacy(entrada_usuario, lenguaje="python", algoritmo="burbuja"):
    # Procesamiento con spaCy (ajusta 'nlp' según tu entorno)
    doc = nlp(entrada_usuario.lower())
    numeros = [token.text for token in doc if token.like_num]
    
    # Mensaje de depuración: ¿qué detecta spaCy como números?
    print("[DEBUG] -> extraer_lista_numeros_spacy")
    print("[DEBUG] -> Entrada del usuario:", entrada_usuario)
    print("[DEBUG] -> Algoritmo recibido:", algoritmo)
    print("[DEBUG] -> Numeros detectados:", numeros)

    if not numeros:
        print("[DEBUG] -> No se detectaron números (numeros está vacío). Regreso None.")
        return None
    
    # Recorremos los tokens y sólo añadimos a la lista los que se puedan convertir a entero
    arr_numeros = []
    for token in doc:
        if token.like_num:
            try:
                arr_numeros.append(int(token.text))
            except ValueError:
                # Ignoramos las palabras como "dos", "tres", etc.
                pass

    # También generamos una representación en texto (p.ej. "7, 3, 9")
    lista_str = ", ".join(numeros)
    
    # Variable donde guardaremos el código
    codigo = ""

    # ---------------------
    # ALGORITMOS DE ORDENAMIENTO
    # ---------------------
    if algoritmo in ["burbuja", "bubble_sort", "bubble sort", "bubble"]:
        print("[DEBUG] -> Entró al if 'burbuja'")
        if lenguaje == "python":
            codigo = f'''def ordenamiento_burbuja(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

lista = [{lista_str}]
lista_ordenada = ordenamiento_burbuja(lista)
print("Lista ordenada:", lista_ordenada)'''
        elif lenguaje == "javascript":
            codigo = f'''function ordenamientoBurbuja(arr) {{
    let n = arr.length;
    for (let i = 0; i < n; i++) {{
        for (let j = 0; j < n - i - 1; j++) {{
            if (arr[j] > arr[j + 1]) {{
                let temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }}
        }}
    }}
    return arr;
}}

let lista = [{lista_str}];
let listaOrdenada = ordenamientoBurbuja(lista);
console.log("Lista ordenada:", listaOrdenada);'''

    elif algoritmo == "quicksort":
        print("[DEBUG] -> Entró al if 'quicksort'")
        if lenguaje == "python":
            codigo = f'''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivote = arr[0]
    menores = [x for x in arr[1:] if x <= pivote]
    mayores = [x for x in arr[1:] if x > pivote]
    return quicksort(menores) + [pivote] + quicksort(mayores)

lista = [{lista_str}]
lista_ordenada = quicksort(lista)
print("Lista ordenada:", lista_ordenada)'''
        elif lenguaje == "javascript":
            codigo = f'''function quicksort(arr) {{
    if (arr.length <= 1) {{
        return arr;
    }}
    let pivote = arr[0];
    let menores = arr.slice(1).filter(x => x <= pivote);
    let mayores = arr.slice(1).filter(x => x > pivote);
    return [...quicksort(menores), pivote, ...quicksort(mayores)];
}}

let lista = [{lista_str}];
let listaOrdenada = quicksort(lista);
console.log("Lista ordenada:", listaOrdenada);'''

    elif algoritmo in ["insercion", "inserción"]:
        print("[DEBUG] -> Entró al if 'insercion/inserción'")
        if lenguaje == "python":
            codigo = f'''def ordenamiento_insercion(arr):
    for i in range(1, len(arr)):
        clave = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > clave:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = clave
    return arr

lista = [{lista_str}]
lista_ordenada = ordenamiento_insercion(lista)
print("Lista ordenada:", lista_ordenada)'''
        elif lenguaje == "javascript":
            codigo = f'''function ordenamientoInsercion(arr) {{
    for (let i = 1; i < arr.length; i++) {{
        let clave = arr[i];
        let j = i - 1;
        while (j >= 0 && arr[j] > clave) {{
            arr[j + 1] = arr[j];
            j--;
        }}
        arr[j + 1] = clave;
    }}
    return arr;
}}

let lista = [{lista_str}];
let listaOrdenada = ordenamientoInsercion(lista);
console.log("Lista ordenada:", listaOrdenada);'''

    elif algoritmo in ["seleccion", "selección"]:
        print("[DEBUG] -> Entró al if 'seleccion/selección'")
        if lenguaje == "python":
            codigo = f'''def ordenamiento_seleccion(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

lista = [{lista_str}]
lista_ordenada = ordenamiento_seleccion(lista)
print("Lista ordenada:", lista_ordenada)'''
        elif lenguaje == "javascript":
            codigo = f'''function ordenamientoSeleccion(arr) {{
    for (let i = 0; i < arr.length; i++) {{
        let min_idx = i;
        for (let j = i + 1; j < arr.length; j++) {{
            if (arr[j] < arr[min_idx]) {{
                min_idx = j;
            }}
        }}
        [arr[i], arr[min_idx]] = [arr[min_idx], arr[i]];
    }}
    return arr;
}}

let lista = [{lista_str}];
let listaOrdenada = ordenamientoSeleccion(lista);
console.log("Lista ordenada:", listaOrdenada);'''

    # ---------------------
    # FACTORIAL - Recursivo (Detecta "recursivo" en el texto)
    # ---------------------
    elif "factorial" in algoritmo and (
    "recursivo" in entrada_usuario.lower() or 
    "recursive" in entrada_usuario.lower()
    ):
        print("[DEBUG] -> Detección de factorial recursivo")
        algoritmo = "factorial_recursivo"  # Mapeo directo a recursivo
        if lenguaje == "python":
            codigo = f'''def factorial_recursivo(n):
        if n <= 1:
            return 1
        return n * factorial_recursivo(n - 1)

lista = [{lista_str}]
for numero in lista:
print(f"Factorial recursivo de {{numero}} =", factorial_recursivo(numero))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function factorialRecursivo(n) {{
        if (n <= 1) {{
            return 1;
        }}
        return n * factorialRecursivo(n - 1);
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Factorial recursivo de", numero, "=", factorialRecursivo(numero));
}}'''


    # ---------------------
    # FACTORIAL - Iterativo (Detecta "iterativo" en el texto)
    # ---------------------
    elif "factorial" in algoritmo and (
    "iterativo" in entrada_usuario.lower() or 
    "iterative" in entrada_usuario.lower()
    ):
        print("[DEBUG] -> Detección de factorial iterativo")
        algoritmo = "factorial_iterativo"  # Mapeo directo a iterativo
        if lenguaje == "python":
            codigo = f'''def factorial_iterativo(n):
        resultado = 1
        for i in range(1, n+1):
            resultado *= i
        return resultado

lista = [{lista_str}]
for numero in lista:
print(f"Factorial iterativo de {{numero}} =", factorial_iterativo(numero))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function factorialIterativo(n) {{
        let resultado = 1;
        for (let i = 1; i <= n; i++) {{
            resultado *= i;
        }}
        return resultado;
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Factorial iterativo de", numero, "=", factorialIterativo(numero));
}}'''


    # ---------------------
    # FACTORIAL - General (Detecta solo "factorial" sin especificar)
    # ---------------------
    elif "factorial" in algoritmo:
        print("[DEBUG] -> Detección de factorial general")
        if lenguaje == "python":
            codigo = f'''def factorial(n):
        resultado = 1
        for i in range(1, n+1):
            resultado *= i
        return resultado

lista = [{lista_str}]
for numero in lista:
print(f"Factorial de {{numero}} =", factorial(numero))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function factorial(n) {{
        let resultado = 1;
        for (let i = 1; i <= n; i++) {{
            resultado *= i;
        }}
        return resultado;
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Factorial de", numero, "=", factorial(numero));
}}'''


    # ---------------------
    # FIBONACCI - Recursivo (Detectar "recursivo" o "recursive")
    # ---------------------
    elif "fibonacci" in algoritmo and (
        "recursivo" in entrada_usuario.lower() or 
        "recursive" in entrada_usuario.lower()
    ):
        print("[DEBUG] -> Detección de fibonacci recursivo")
        algoritmo = "fibonacci_recursivo"
        if lenguaje == "python":
            codigo = f'''def fibonacci_recursivo(n):
        if n < 2:
            return n
        return fibonacci_recursivo(n - 1) + fibonacci_recursivo(n - 2)

lista = [{lista_str}]
for numero in lista:
print(f"Fibonacci recursivo de {{numero}} = {{fibonacci_recursivo(numero)}}")'''
        
        elif lenguaje == "javascript":
            codigo = f'''function fibonacciRecursivo(n) {{
        if (n < 2) {{
            return n;
        }}
        return fibonacciRecursivo(n - 1) + fibonacciRecursivo(n - 2);
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Fibonacci recursivo de", numero, "=", fibonacciRecursivo(numero));
}}'''


    # ---------------------
    # FIBONACCI - Iterativo (Detectar "iterativo" o "iterative")
    # ---------------------
    elif "fibonacci" in algoritmo and (
        "iterativo" in entrada_usuario.lower() or 
        "iterative" in entrada_usuario.lower()
    ):
        print("[DEBUG] -> Detección de fibonacci iterativo")
        algoritmo = "fibonacci_iterativo"
        if lenguaje == "python":
            codigo = f'''def fibonacci_iterativo(n):
        if n < 2:
            return n
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

lista = [{lista_str}]
for numero in lista:
print(f"Fibonacci iterativo de {{numero}} = {{fibonacci_iterativo(numero)}}")'''
        
        elif lenguaje == "javascript":
            codigo = f'''function fibonacciIterativo(n) {{
        if (n < 2) return n;
        let a = 0, b = 1;
        for (let i = 2; i <= n; i++) {{
            let temp = a + b;
            a = b;
            b = temp;
        }}
        return b;
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Fibonacci iterativo de", numero, "=", fibonacciIterativo(numero));
}}'''


    # ---------------------
    # FIBONACCI - General (Detecta solo "fibonacci")
    # ---------------------
    elif "fibonacci" in algoritmo:
        print("[DEBUG] -> Detección de fibonacci general")
        if lenguaje == "python":
            codigo = f'''def fibonacci(n):
        if n < 2:
            return n
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

lista = [{lista_str}]
for numero in lista:
print(f"Fibonacci de {{numero}} = {{fibonacci(numero)}}")'''
        
        elif lenguaje == "javascript":
            codigo = f'''function fibonacci(n) {{
        if (n < 2) return n;
        let a = 0, b = 1;
        for (let i = 2; i <= n; i++) {{
            let temp = a + b;
            a = b;
            b = temp;
        }}
        return b;
    }}

let lista = [{lista_str}];
for (let numero of lista) {{
console.log("Fibonacci de", numero, "=", fibonacci(numero));
}}'''

    # ---------------------
    # NÚMERO PRIMO
    # ---------------------
    elif algoritmo in [
    "primo",
    "número primo",
    "es primo",
    "verificar primo",
    "prime",
    "is prime",
    "check prime",
    "prime number",
    "find prime",
    "calculate prime"
    ]:
        print("[DEBUG] -> Entró al if 'primo'")
        if lenguaje == "python":
            codigo = f'''def es_primo(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

lista = [{lista_str}]
for numero in lista:
    if es_primo(numero):
        print(f"{{numero}} es primo.")
    else:
        print(f"{{numero}} NO es primo.")'''
        elif lenguaje == "javascript":
            codigo = f'''function esPrimo(n) {{
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {{
        if (n % i === 0) {{
            return false;
        }}
    }}
    return true;
}}

let lista = [{lista_str}];
for (let numero of lista) {{
    if (esPrimo(numero)) {{
        console.log(numero + " es primo.");
    }} else {{
        console.log(numero + " NO es primo.");
    }}
}}'''

    # ---------------------
    # MÁXIMO COMÚN DIVISOR (MCD) - EUCLIDES (Iterativo, Recursivo y General)
    # ---------------------

    # MCD - Recursivo
    elif "mcd" in algoritmo and (
        "recursivo" in entrada_usuario.lower() or 
        "recursively" in entrada_usuario.lower() or 
        "recursive" in entrada_usuario.lower() 
    ):
        print("[DEBUG] -> Detección de MCD recursivo")
        algoritmo = "mcd_recursivo"

        if lenguaje == "python":
            codigo = f'''def mcd_euclides_recursivo(a, b):
        if b == 0:
            return a
        return mcd_euclides_recursivo(b, a % b)

    def mcd_lista_recursiva(arr):
        if not arr:
            return None
    def helper(i, acumulado):
            if i == len(arr):
                return acumulado
            return helper(i+1, mcd_euclides_recursivo(acumulado, arr[i]))
        return helper(1, arr[0])

lista = [{lista_str}]
print("MCD recursivo de la lista:", mcd_lista_recursiva(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function mcdEuclidesRecursivo(a, b) {{
        if (b === 0) {{
            return a;
        }}
        return mcdEuclidesRecursivo(b, a % b);
    }}

    function mcdListaRecursiva(arr) {{
        if (arr.length === 0) return null;
        function helper(i, acumulado) {{
            if (i === arr.length) return acumulado;
            return helper(i + 1, mcdEuclidesRecursivo(acumulado, arr[i]));
        }}
        return helper(1, arr[0]);
    }}

let lista = [{lista_str}];
console.log("MCD recursivo de la lista:", mcdListaRecursiva(lista));'''


    # MCD - Iterativo
    elif "mcd" in algoritmo and (
        "iterativo" in entrada_usuario.lower() or 
        "iterative" in entrada_usuario.lower()
    ):
        print("[DEBUG] -> Detección de MCD iterativo")
        algoritmo = "mcd_iterativo"
        
        if lenguaje == "python":
            codigo = f'''def mcd_euclides_iterativo(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def mcd_lista_iterativa(arr):
        if not arr:
            return None
        resultado = arr[0]
        for i in range(1, len(arr)):
            resultado = mcd_euclides_iterativo(resultado, arr[i])
        return resultado

lista = [{lista_str}]
print("MCD iterativo de la lista:", mcd_lista_iterativa(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function mcdEuclidesIterativo(a, b) {{
        while (b !== 0) {{
            let temp = b;
            b = a % b;
            a = temp;
        }}
        return a;
    }}

    function mcdListaIterativa(arr) {{
        if (arr.length === 0) return null;
        let resultado = arr[0];
        for (let i = 1; i < arr.length; i++) {{
            resultado = mcdEuclidesIterativo(resultado, arr[i]);
        }}
        return resultado;
    }}

let lista = [{lista_str}];
console.log("MCD iterativo de la lista:", mcdListaIterativa(lista));'''


    # MCD - General (sin especificar iterativo o recursivo)
    elif any(key in algoritmo for key in [
    "mcd", 
    "gcd", 
    "greatest common divisor", 
    "calculate gcd", 
    "find gcd", 
    "euclidean algorithm",
    "máximo común divisor",
    "maximo comun divisor",
    ]):
        print("[DEBUG] -> Detección de MCD general")
        algoritmo = "gcd_general"
        
        if lenguaje == "python":
            codigo = f'''def mcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def mcd_lista(arr):
        if not arr:
            return None
        resultado = arr[0]
        for i in range(1, len(arr)):
            resultado = mcd(resultado, arr[i])
        return resultado

lista = [{lista_str}]
print("MCD de la lista:", mcd_lista(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function mcd(a, b) {{
        while (b !== 0) {{
            let temp = b;
            b = a % b;
            a = temp;
        }}
        return a;
    }}

    function mcdLista(arr) {{
        if (arr.length === 0) return null;
        let resultado = arr[0];
        for (let i = 1; i < arr.length; i++) {{
            resultado = mcd(resultado, arr[i]);
        }}
        return resultado;
    }}

let lista = [{lista_str}];
console.log("MCD de la lista:", mcdLista(lista));'''


    # ---------------------
    # MÍNIMO COMÚN MÚLTIPLO (MCM)
    # ---------------------
    elif algoritmo in [
    "mcm",
    "lcm",
    "mínimo común múltiplo",
    "least common multiple",
    "least common multiple of",
    "find lcm",
    "calcula el mcm",
    "calcula el minimo comun multiplo",
    "find the least common multiple of",
    "calculate the least common multiple of"
    ]:
        print("[DEBUG] -> Entró al if 'mcm'")
        if lenguaje == "python":
            codigo = f'''def mcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def mcm(a, b):
    return abs(a*b) // mcd(a, b) if a and b else 0

def mcm_lista(arr):
    if not arr:
        return None
    resultado = arr[0]
    for i in range(1, len(arr)):
        resultado = mcm(resultado, arr[i])
    return resultado

lista = [{lista_str}]
print("MCM de la lista:", mcm_lista(lista))'''
        elif lenguaje == "javascript":
            codigo = f'''function mcd(a, b) {{
    while (b !== 0) {{
        let temp = b;
        b = a % b;
        a = temp;
    }}
    return a;
}}

function mcm(a, b) {{
    if (a === 0 || b === 0) return 0;
    return Math.abs(a * b) / mcd(a, b);
}}

function mcmLista(arr) {{
    if (arr.length === 0) return null;
    let resultado = arr[0];
    for (let i = 1; i < arr.length; i++) {{
        resultado = mcm(resultado, arr[i]);
    }}
    return resultado;
}}

let lista = [{lista_str}];
console.log("MCM de la lista:", mcmLista(lista));'''

    # ---------------------
    # POTENCIACIÓN (Exponenciación rápida)
    # ---------------------
    elif algoritmo in ["potenciacion", "fast_exponentiation", "exponentiation", "power"]:
        print("[DEBUG] -> Entró al if 'potenciacion'")
        if lenguaje == "python":
            codigo = f'''def potencia_rapida(base, exp):
    if exp < 0:
        return 1 / potencia_rapida(base, -exp)
    if exp == 0:
        return 1
    if exp == 1:
        return base
    if exp % 2 == 0:
        half = potencia_rapida(base, exp // 2)
        return half * half
    else:
        return base * potencia_rapida(base, exp - 1)

base = {arr_numeros[0] if len(arr_numeros) > 0 else 2}
exponente = {arr_numeros[1] if len(arr_numeros) > 1 else 3}

print(f"Potencia {{base}}^{{exponente}} =", potencia_rapida(base, exponente))'''
        elif lenguaje == "javascript":
            codigo = f'''function potenciaRapida(base, exp) {{
    if (exp < 0) {{
        return 1 / potenciaRapida(base, -exp);
    }}
    if (exp === 0) {{
        return 1;
    }}
    if (exp === 1) {{
        return base;
    }}
    if (exp % 2 === 0) {{
        let half = potenciaRapida(base, exp / 2);
        return half * half;
    }} else {{
        return base * potenciaRapida(base, exp - 1);
    }}
}}

let base = {arr_numeros[0] if len(arr_numeros) > 0 else 2};
let exponente = {arr_numeros[1] if len(arr_numeros) > 1 else 3};

console.log("Potencia", base + "^" + exponente, "=", potenciaRapida(base, exponente));'''

    # ---------------------
    # CONTAR DÍGITOS DE UN NÚMERO
    # ---------------------
    elif algoritmo in ["digit_count", "contar_digitos", "cuenta los digitos", "count the digits"]:
        print("[DEBUG] -> Entró al if 'digit_count/contar_digitos'")
        if lenguaje == "python":
            codigo = f'''def contar_digitos(n):
    if n == 0:
        return 1  # el 0 tiene 1 dígito
    n_abs = abs(n)
    contador = 0
    while n_abs > 0:
        n_abs //= 10
        contador += 1
    return contador

lista = [{lista_str}]
for numero in lista:
    print(f"El número {{numero}} tiene {{contar_digitos(numero)}} dígitos.")'''
        elif lenguaje == "javascript":
            codigo = f'''function contarDigitos(n) {{
    if (n === 0) return 1;
    let n_abs = Math.abs(n);
    let contador = 0;
    while (n_abs > 0) {{
        n_abs = Math.floor(n_abs / 10);
        contador++;
    }}
    return contador;
}}

let lista = [{lista_str}];
for (let numero of lista) {{
    console.log("El número", numero, "tiene", contarDigitos(numero), "dígitos.");
}}'''
            
    
    # ---------------------
    # Suma de elementos de una lista
    # ---------------------
    elif any(key in entrada_usuario.lower() for key in [
        "suma_lista", 
        "sumar elementos de la lista",
        "suma de lista",
        "dame la suma de la lista",
        "suma de esta lista",
        "podrías calcular la suma de la lista",
        "me puedes dar la suma de la lista",
        "calcula la suma de la lista",
        "puedes calcular la suma de la lista",
        "sum elements of list",
        "sum of list",
        "calculate sum of list",
        "add all elements in list",
        "total sum of list"
    ]):
        print("[DEBUG] -> Entró al if 'suma_lista'")
        
        # Extraer los números detectados
        lista_str = ', '.join(numeros) if numeros else "0"
        
        # Generar código en Python o JavaScript
        if lenguaje == "python":
            codigo = f'''def suma_lista(arr):
        return sum(arr)

lista = [{lista_str}]
print("Suma de la lista:", suma_lista(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function sumaLista(arr) {{
        return arr.reduce((acc, val) => acc + val, 0);
    }}

let lista = [{lista_str}];
console.log("Suma de la lista:", sumaLista(lista));'''


    # ---------------------
    # Promedio de elementos de una lista
    # ---------------------
    elif any(key in entrada_usuario.lower() for key in [
        "promedio_lista", 
        "calcula el promedio de la lista",
        "promedio de lista",
        "dame el promedio de la lista",
        "promedio de esta lista",
        "average elements of list",
        "average of list",
        "calculate average of list",
        "get average of list"
    ]):
        print("[DEBUG] -> Entró al if 'promedio_lista'")
        
        # Extraer los números detectados
        lista_str = ', '.join(numeros) if numeros else "0"
        
        # Generar código en Python o JavaScript
        if lenguaje == "python":
            codigo = f'''def promedio_lista(arr):
        return sum(arr) / len(arr) if arr else 0

lista = [{lista_str}]
print("Promedio de la lista:", promedio_lista(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function promedioLista(arr) {{
        return arr.length > 0 ? arr.reduce((acc, val) => acc + val, 0) / arr.length : 0;
}}

let lista = [{lista_str}];
console.log("Promedio de la lista:", promedioLista(lista));'''

    # ---------------------
    # Máximo y mínimo de una lista
    # ---------------------
    elif any(key in algoritmo for key in [
    "max_min_lista", 
    "find max and min", 
    "maximum and minimum in list", 
    "calculate max and min",
    "get max and min from list",
    "máximo y mínimo",
    "highest and lowest values",
    "biggest and smallest in list",
    "find the maximum and minimum",
    "calculate the maximum and minimum of",
    "get the maximum and minimum of"
]):
        print("[DEBUG] -> Entró al if 'max_min_lista'")
        if lenguaje == "python":
            codigo = f'''def max_min_lista(arr):
        return max(arr), min(arr)

lista = [{lista_str}]
maximo, minimo = max_min_lista(lista)
print(f"Máximo: {{maximo}}, Mínimo: {{minimo}}")'''
    
        elif lenguaje == "javascript":
            codigo = f'''function maxMinLista(arr) {{
        return [Math.max(...arr), Math.min(...arr)];
}}

let lista = [{lista_str}];
let [maximo, minimo] = maxMinLista(lista);
console.log("Máximo:", maximo, "Mínimo:", minimo);'''



    # ---------------------
    # Búsqueda Lineal de un elemento (Activación por palabras clave)
    # ---------------------
    elif re.search(r"\b(busca|find|locate|search)\b.*?\b(lista|list|array)\b", entrada_usuario.lower()):
        
        print(f"[DEBUG] -> Entrada Usuario: {entrada_usuario}")
        
        # Detectar número a buscar con regex
        match = re.search(r"\b(busca|find|locate|search)\b.*?(\d+).*?\b(lista|list|array)\b", entrada_usuario.lower())
        
        if match:
            print(f"[DEBUG] -> Coincidencia Regex: {match.group()}")
            elemento = match.group(2)  # Número encontrado
        else:
            elemento = lista_str.split(",")[0]  # Por defecto, primer número de la lista
            print(f"[DEBUG] -> No coincidió la regex. Entrada: {entrada_usuario.lower()}")
        
        print(f"[DEBUG] -> Elemento a buscar: {elemento}")
        
        # Generar código de búsqueda lineal
        if lenguaje == "python":
            codigo = f'''def busqueda_lineal(arr, x):
    return x in arr

lista = [{lista_str}]
elemento = {elemento}
print(f"Elemento {{elemento}} encontrado:", busqueda_lineal(lista, elemento))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function busquedaLineal(arr, x) {{
    return arr.includes(x);
}}

let lista = [{lista_str}];
let elemento = {elemento};
console.log("Elemento encontrado:", busquedaLineal(lista, elemento));'''

    # ---------------------
    # Invertir Lista
    # ---------------------
    elif any(key in algoritmo for key in [
        "invertir_lista", 
        "reverse_list", 
        "invertir lista",
        "reversa de lista",
        "invierte la lista",
        "lista invertida",
        "da la vuelta a la lista",
        "voltea la lista",
        "reverse list",
        "invert list",
        "invert list elements",
        "reverse the list",
        "flip the list",
        "turn the list around",
        "list reversed"
    ]):
        print("[DEBUG] -> Entró al if 'invertir_lista'")
        if lenguaje == "python":
            codigo = f'''def invertir_lista(arr):
        return arr[::-1]

lista = [{lista_str}]
print("Lista invertida:", invertir_lista(lista))'''
        
        elif lenguaje == "javascript":
            codigo = f'''function invertirLista(arr) {{
        return arr.reverse();
}}

let lista = [{lista_str}];
console.log("Lista invertida:", invertirLista(lista));'''



    # ---------------------
    # Eliminar Duplicados de una lista
    # ---------------------
    elif any(key in algoritmo for key in [
    "eliminar_duplicados", 
    "remove_duplicates", 
    "eliminar duplicados de lista",
    "elimina los duplicados de la lista",
    "quita duplicados de lista",
    "borra duplicados de lista",
    "sin duplicados en lista",
    "remove duplicates from list",
    "delete duplicates in list",
    "get unique values from list",
    "filter out duplicates",
    "no duplicates in list",
    "remove repeated elements",
    "deduplicate list",
    "elimina duplicados",
    "quita los duplicados",
    "borra los duplicados",
    "remove the duplicates"
]):
        print("[DEBUG] -> Entró al if 'eliminar_duplicados'")
        if lenguaje == "python":
            codigo = f'''def eliminar_duplicados(arr):
        return list(set(arr))

lista = [{lista_str}]
print("Lista sin duplicados:", eliminar_duplicados(lista))'''
    
        elif lenguaje == "javascript":
            codigo = f'''function eliminarDuplicados(arr) {{
        return [...new Set(arr)];
}}

let lista = [{lista_str}];
console.log("Lista sin duplicados:", eliminarDuplicados(lista));'''




    else:
        print("[DEBUG] -> Entró al else final => no coincide con ningún if/elif.")
        return "Lo siento, aún no sé generar código para ese algoritmo."

    # (Opcional) Si deseas guardar la última ejecución en alguna variable global
    global ultimo_bucle
    ultimo_bucle = {
        "codigo": f"```{lenguaje}\n{codigo}\n```",
        "tipo": algoritmo
    }

    print("[DEBUG] -> Se generó el código con éxito. Retornando el bloque final.")
    return (
        f"Aquí tienes el algoritmo de {algoritmo} en {lenguaje} "
        f"para la lista [{lista_str}]:\n\n"
        f"```{lenguaje}\n{codigo}\n```"
    )

    



def generar_codigo_personalizado(entrada_usuario):
    print(f"[DEBUG] Detectando patrón en: {entrada_usuario}")
    
    # --- Filtro conceptual: SI detecta "por que", etc. NO cortamos; solo seguimos
    if any(frase in entrada_usuario.lower() for frase in ["por que", "porque", "para que", "para qué"]):
        print("[DEBUG] -> Es una pregunta conceptual (por qué, para qué). Pero dejamos continuar.")
        # Simplemente no hacemos return. 'pass' o nada
        pass
    
    # --- Filtro: ¿El usuario pide explícitamente código o algoritmo?
    # Si no menciona nada como "codigo", "algoritmo", "muestrame", etc., devolvemos None.
    # Ajusta las palabras según necesites.
    palabras_clave_codigo = [
        "codigo", "algoritmo", "muéstrame", "muestrame", 
        "quiero ver", "enséñame", "ensename", "muestra",
        "genera", "genérame", "generar"
    ]
    if not any(pk in entrada_usuario.lower() for pk in palabras_clave_codigo):
        print("[DEBUG] -> El usuario NO pidió explíctamente ver o generar código. Retornamos None.")
        return None

    # --- Si sí pide código, entonces revisamos si habla de "ordenamiento", "for", "if" ...
    if "ordenamiento" in entrada_usuario.lower() or "bubble sort" in entrada_usuario.lower():
        print("[DEBUG] Bubble sort detectado")
        return extraer_lista_numeros_spacy(entrada_usuario)
    
    elif "for" in entrada_usuario.lower():
        print("[DEBUG] Bucle for detectado")
        return generar_codigo_for(entrada_usuario)
    
    elif "if" in entrada_usuario.lower():
        print("[DEBUG] Condicional if detectado")
        if "genera codigo" in entrada_usuario.lower():
            return generar_codigo_if_natural(entrada_usuario)
        else:
            return generar_codigo_if(entrada_usuario)
    
    else:
        print("[DEBUG] -> No coincide con 'ordenamiento', 'for' ni 'if'. Devuelvo None para que siga el flujo.")
        return None



from rapidfuzz import fuzz

def detectar_intent_fuzzy(texto_usuario, intenciones, umbral=70):
    """
    Compara la frase del usuario con cada pattern de las intenciones
    usando coincidencia difusa (fuzzy). Retorna el tag con mayor similitud
    si supera 'umbral'. De lo contrario, None.
    """
    texto_usuario = texto_usuario.lower()

    mejor_tag = None
    mayor_similitud = 0

    for intent in intenciones["intents"]:
        for patron in intent["patterns"]:
            patron_lower = patron.lower()
            similitud = fuzz.partial_ratio(texto_usuario, patron_lower)  # 0..100
            
            if similitud > mayor_similitud:
                mayor_similitud = similitud
                mejor_tag = intent["tag"]
    
    if mayor_similitud >= umbral:
        return mejor_tag
    return None


import tkinter as tk
from tkinter import scrolledtext


        
from tkinter import Frame, Label, Entry, Button, Scrollbar, Canvas

import tkinter as tk
from tkinter import Frame, Label, Entry, Button, Scrollbar, Canvas

class ChatbotUI(tk.Tk):
    def __init__(self, ruta_intents="intents.json", ruta_modelo="modelo_chatbot.pth"):
        super().__init__()

        self.nombre_bot = "CHATXD"
        self.menu_copiar = tk.Menu(self, tearoff=0)
        self.menu_copiar.add_command(label="Copiar", command=self.copiar_texto)
        
        self.title("CHATXD v3")
        self.geometry("660x600")
        self.configure(bg="#252526")  # Fondo oscuro ligeramente más claro
        
        # Inicialización de variables
        self.palabras = []  # Evita el error de atributo no definido
        
        (self.intenciones,
         self.tam_entrada,
         self.tam_oculto,
         self.tam_salida,
         self.palabras,
         self.etiquetas,
         self.estado_modelo,
         self.contextos) = cargar_datos_modelo(ruta_intents, ruta_modelo)
        
        
        self.modelo = cargar_modelo(self.tam_entrada, self.tam_oculto, self.tam_salida, self.estado_modelo)
        
        self.contexto_usuario = {"contexto": None, "siguiente_etiqueta": None}  # Añadido para evitar errores
        
        # Contenedor principal con scrollbar
        self.frame_principal = Frame(self, bg="#252526")
        self.frame_principal.pack(fill="both", expand=True)
        
        self.canvas = Canvas(self.frame_principal, bg="#252526")
        self.frame_contenedor = Frame(self.canvas, bg="#252526")
        self.scrollbar = Scrollbar(self.frame_principal, orient="vertical", command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame_contenedor, anchor="nw")
        
        # Entrada y botón abajo
        self.entry_frame = Frame(self, bg="#252526")
        self.entry_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        
        self.entry_user = Entry(self.entry_frame, font=("Arial", 14, "bold"), bg="#3C3C3C", fg="white", insertbackground="white")
        self.entry_user.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.btn_send = Button(self.entry_frame, text="Enviar", command=self.on_send, bg="#3B82F6", fg="white", font=("Arial", 14, "bold"))
        self.btn_send.pack(side="right")
        
        self.entry_user.bind("<Return>", self.on_send_event)
        
        self.frame_contenedor.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
    
    def _imprimir_en_chat(self, remitente, mensaje, es_usuario=False):
        chat_frame = Frame(self.frame_contenedor, bg="#252526", pady=5)
        chat_frame.pack(fill="x", pady=5, padx=10, anchor="e" if es_usuario else "w")
        
        

        fuente = ("Arial", 12, "bold")
        color_fondo = "#3B82F6" if es_usuario else "#4CAF50"
        label = Label(chat_frame, text=f"{remitente}: {mensaje}", wraplength=600, justify="left",
                      font=fuente, bg=color_fondo, fg="white", padx=10, pady=5)
        label.bind("<Button-3>", lambda e, t=mensaje: self.mostrar_menu_copiar(e, t))
        label.pack(anchor="e" if es_usuario else "w")
        
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1)
    
    def on_send_event(self, event):
        self.on_send()
        
    def on_send(self):
        texto_usuario = self.entry_user.get().strip()
        if not texto_usuario:
            return
        
        self._imprimir_en_chat("TÚ", texto_usuario, es_usuario=True)
        self.entry_user.delete(0, tk.END)
        
        respuesta = self.procesar_entrada_usuario(texto_usuario)
        
        self._imprimir_en_chat("CHATXD", respuesta, es_usuario=False)
    
    def mostrar_menu_copiar(self, event, texto):
        self.texto_a_copiar = texto
        self.menu_copiar.tk_popup(event.x_root, event.y_root)

    def copiar_texto(self):
        self.clipboard_clear()
        self.clipboard_append(self.texto_a_copiar)

    def procesar_entrada_usuario(self, texto_usuario):
        print("[DEBUG] => Iniciando procesar_entrada_usuario con texto:", texto_usuario)

        texto_limpio = texto_usuario.lower().strip()

        # A) ¿Pide código explícitamente?
        palabras_pide_codigo = [
            "genera codigo", "genera un codigo", "dame codigo", "dame un codigo",
            "quiero ver codigo", "muéstrame el código", "muestrame el codigo",
            "generate code", "generate a code", "give me code", "gimme a code",
            "show me the code", "i want code", "i want a code"
        ]

        if any(frase in texto_limpio for frase in palabras_pide_codigo):
            print("[DEBUG] => El usuario está solicitando código explícitamente. Saltamos intenciones...")

            resp_ordenamiento = detectar_ordenamiento_numeros(texto_usuario)
            if resp_ordenamiento is not None:
                print("[DEBUG] => Se generó código de ordenamiento.")
                return resp_ordenamiento

            resp_cadenas = detectar_operacion_cadenas(texto_usuario)
            if resp_cadenas:
                print("[DEBUG] => Se generó código de cadenas.")
                return resp_cadenas

            if "genera codigo" in texto_limpio:
                print("[DEBUG] => Se detectó 'genera codigo' if natural.")
                return generar_codigo_if_natural(texto_usuario)

            if "for" in texto_limpio or "if" in texto_limpio or "anidar" in texto_limpio:
                print("[DEBUG] => Se detectó for/if/anidar. Generamos código personalizado.")
                return generar_codigo_personalizado(texto_usuario)

            if "anidar" in texto_limpio and not ultimo_bucle["codigo"]:
                print("[DEBUG] => Se pidió anidar, pero no hay un bloque previo.")
                return "No tengo un bucle o condición previa en memoria. Pídeme que genere uno primero."

            print("[DEBUG] => No se halló ningún algoritmo tras la petición de código. Fallback código.")
            return "No encontré un algoritmo o bucle específico. ¿Podrías detallar qué código quieres generar?"

        # B) No se pidió código explícito - Revisar intenciones (modelo/fuzzy)
        print("[DEBUG] => El usuario NO pidió código explícitamente. Revisamos intenciones...")

        idioma = detectar_idioma(texto_usuario)
        print("[DEBUG] => Idioma detectado:", idioma)
        tokens = dividir_en_palabras(texto_usuario, idioma=idioma)
        vector = vector_bolsa_de_palabras(tokens, self.palabras, idioma=idioma).reshape(1, -1)
        vector = torch.from_numpy(vector).to(dispositivo)

        salida = self.modelo(vector)
        _, prediccion = torch.max(salida, dim=1)
        etiqueta = self.etiquetas[prediccion.item()]

        probabilidad = torch.softmax(salida, dim=1)
        confianza = probabilidad[0][prediccion.item()]

        print(f"[DEBUG] => Etiqueta predicha: {etiqueta}, Confianza: {confianza.item():.2f}")

        if confianza.item() > 0.50:
            # Generar respuesta si la etiqueta existe
            for intencion in self.intenciones["intents"]:
                if etiqueta == intencion["tag"]:
                    return random.choice(intencion["responses"])  # Devolver solo respuesta, no imprimir

            # Manejo de etiquetas especiales
            if etiqueta == "rechazo_detalles_emociones":
                return random.choice([
                    i["responses"] for i in self.intenciones["intents"]
                    if i["tag"] == etiqueta
                ][0])

            elif etiqueta in ["emocion_feliz", "emocion_tranquilo", "emocion_enojado",
                            "emocion_normal", "emocion_cansado", "emocion_estresado"]:
                return random.choice([
                    i["responses"] for i in self.intenciones["intents"]
                    if i["tag"] == etiqueta
                ][0])

        # ===============================
        # ELSE => Fallback a fuzzy matching
        # ===============================
        etiqueta_fuzzy = detectar_intent_fuzzy(texto_usuario, self.intenciones, umbral=70)
        if etiqueta_fuzzy:
            for intencion in self.intenciones["intents"]:
                if etiqueta_fuzzy == intencion["tag"]:
                    return random.choice(intencion["responses"])

        # Fallback final => No hay coincidencias
        return "Lo siento, no he entendido lo que dijiste."






# ===== MAIN para arrancar la ventana =====
if __name__ == "__main__":
    app = ChatbotUI(ruta_intents="intents.json", ruta_modelo="modelo_chatbot.pth")
    app.mainloop()