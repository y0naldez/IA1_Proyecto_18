import { stemmer } from 'stemmer';
import winkTokenizer from 'wink-tokenizer';

const tokenizer = new winkTokenizer();

/**
 * Divide una oración en palabras y aplica stemming.
 * @param {string} oracion - La entrada del usuario.
 * @returns {string[]} - Lista de raíces de las palabras (tokens procesados).
 */
export const dividirEnPalabras = (oracion) => {
    // Convertir a minúsculas y separar operadores matemáticos
    oracion = oracion.toLowerCase().replace(/([+\-*/])/g, ' $1 ');
    const tokens = tokenizer.tokenize(oracion)
        .filter(token => token.tag === 'word' || token.tag === 'number' || /[+\-*/]/.test(token.value))
        .map(token => token.value);

    // Aplicar el stemmer para obtener las raíces
    const tokensConRaices = tokens.map(token => (isNaN(token) ? stemmer(token) : token));

    console.log(`[DEBUG] Tokens originales: ${tokens}`);
    console.log(`[DEBUG] Tokens con raíces: ${tokensConRaices}`);
    return tokensConRaices;
};

/**
 * Convierte una lista de raíces en un vector de bolsa de palabras.
 * @param {string[]} tokens - Raíces de las palabras procesadas.
 * @param {string[]} palabrasConocidas - Palabras del vocabulario.
 * @returns {number[]} - Vector de bolsa de palabras.
 */
export const vectorBolsaDePalabras = (tokens, palabrasConocidas) => {
    const vector = Array(palabrasConocidas.length).fill(0);

    tokens.forEach(token => {
        const indice = palabrasConocidas.indexOf(token);
        if (indice !== -1) {
            vector[indice] = 1;
        } else {
            console.warn(`[DEBUG] Token no encontrado en el vocabulario: ${token}`);
        }
    });

    console.log(`[DEBUG] Vector de bolsa de palabras generado: ${vector}`);
    return vector;
};
