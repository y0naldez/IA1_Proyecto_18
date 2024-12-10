import stemmer from 'stemmer';
import { Tokenizer } from 'wink-tokenizer';

const tokenizer = new Tokenizer();

// Configura el tokenizador para que reconozca palabras, números, y operadores matemáticos
tokenizer.defineConfig({
    currency: false,
    number: true,
    email: false,
    url: false,
    mention: false,
    hashtag: false,
});

export function dividirEnPalabras(oracion) {
    // Tokeniza la oración en palabras o símbolos
    const tokens = tokenizer.tokenize(oracion.toLowerCase());
    return tokens
        .filter(token => token.tag === 'word' || token.tag === 'number' || token.value.match(/[\+\-\*/]/)) // Filtra palabras y operadores
        .map(token => token.value);
}

export function obtenerRaiz(palabra) {
    const excepciones = ["suma", "resta", "multiplicacion", "division", "+", "-", "*", "/"];
    if (excepciones.includes(palabra) || !isNaN(palabra)) {
        return palabra; // Devuelve la palabra original si es una excepción o número
    }
    return stemmer(palabra); // Devuelve la raíz de la palabra
}

export function vectorBolsaDePalabras(oracionTokenizada, palabrasConocidas) {
    const palabrasRaiz = oracionTokenizada.map(obtenerRaiz);
    const vector = Array(palabrasConocidas.length).fill(0);
    palabrasRaiz.forEach(raiz => {
        const indice = palabrasConocidas.indexOf(raiz);
        if (indice !== -1) {
            vector[indice] = 1;
        }
    });
    return vector;
}
