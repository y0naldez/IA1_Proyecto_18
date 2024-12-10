import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { dividirEnPalabras, vectorBolsaDePalabras } from './utils/textProcessing.js';
import './App.css';

const App = () => {
    const [modelo, setModelo] = useState(null);
    const [intents, setIntents] = useState(null);
    const [palabras, setPalabras] = useState([]);
    const [input, setInput] = useState('');
    const [chat, setChat] = useState([]);
    const [etiquetas, setEtiquetas] = useState([]);
    const [escribiendo, setEscribiendo] = useState(false);
    const chatContainerRef = useRef(null);

    useEffect(() => {
        const cargarDatos = async () => {
            try {
                const modeloCargado = await tf.loadGraphModel('./model.json');
                const intentsRes = await fetch('./intents.json').then(res => res.json());
                const vocabularioRes = await fetch('./vocabulario.json').then(res => res.json());

                setModelo(modeloCargado);
                setIntents(intentsRes);
                setPalabras(vocabularioRes.palabras || []);
                setEtiquetas(vocabularioRes.etiquetas || []);
            } catch (error) {
                console.error('Error al cargar los datos:', error);
            }
        };
        cargarDatos();
    }, []);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chat]);

    const manejarEntrada = async (texto) => {
        if (!modelo || !intents || palabras.length === 0) return;

        const tokens = dividirEnPalabras(texto);
        const vector = vectorBolsaDePalabras(tokens, palabras);
        const tensor = tf.tensor([vector]);

        setEscribiendo(true); // Mostrar indicador de "escribiendo..."

        try {
            // Simular un retraso para la respuesta del bot
            await new Promise(resolve => setTimeout(resolve, 1500));

            const salida = modelo.predict(tensor);
            const prediccion = salida.argMax(-1).dataSync()[0];
            const confianza = salida.dataSync()[prediccion];

            if (confianza > 0.75) {
                const etiqueta = etiquetas[prediccion];

                if (["suma", "resta", "multiplicacion", "division"].includes(etiqueta)) {
                    const numeros = texto.match(/\d+/g)?.map(Number) || [];
                    if (numeros.length >= 2) {
                        const [num1, num2] = numeros;
                        const resultado =
                            etiqueta === "suma"
                                ? num1 + num2
                                : etiqueta === "resta"
                                ? num1 - num2
                                : etiqueta === "multiplicacion"
                                ? num1 * num2
                                : num2 !== 0
                                ? num1 / num2
                                : "No puedo dividir entre cero. 😅";
                        setChat(prevChat => [...prevChat, { tipo: 'bot', mensaje: `El resultado es ${resultado}` }]);
                    } else {
                        setChat(prevChat => [...prevChat, { tipo: 'bot', mensaje: 'Faltan números para calcular. Intenta de nuevo.' }]);
                    }
                } else {
                    const respuestas = intents.intents.find(intent => intent.tag === etiqueta).responses;
                    const respuestaAleatoria = respuestas[Math.floor(Math.random() * respuestas.length)];
                    setChat(prevChat => [...prevChat, { tipo: 'bot', mensaje: respuestaAleatoria }]);
                }
            } else {
                setChat(prevChat => [...prevChat, { tipo: 'bot', mensaje: 'Lo siento, no entiendo eso.' }]);
            }
        } catch (error) {
            console.error('Error durante la predicción:', error);
        } finally {
            setEscribiendo(false); // Ocultar indicador de "escribiendo..."
        }
    };

    const manejarSubmit = (e) => {
        e.preventDefault();
        if (input.trim()) {
            setChat(prevChat => [...prevChat, { tipo: 'usuario', mensaje: input }]);
            manejarEntrada(input);
            setInput('');
        }
    };

    return (
        <div className="chat-container">
            <h1 className="chat-title">ChatBot</h1>
            <div ref={chatContainerRef} className="chat-box">
                {chat.map((linea, index) => (
                    <div
                        key={index}
                        className={`chat-message ${linea.tipo === 'usuario' ? 'user' : 'bot'}`}
                    >
                        {linea.mensaje}
                    </div>
                ))}
                {escribiendo && (
                    <div className="chat-message bot typing-indicator">
                        <span>.</span><span>.</span><span>.</span>
                    </div>
                )}
            </div>
            <form onSubmit={manejarSubmit} className="chat-form">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Escribe tu mensaje..."
                    className="chat-input"
                />
                <button type="submit" className="chat-submit">Enviar</button>
            </form>
        </div>
    );
};

export default App;
