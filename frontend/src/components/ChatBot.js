import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { dividirEnPalabras, vectorBolsaDePalabras } from '../utils/textProcessing.js'; 

const ChatBot = () => {
  const [mensaje, setMensaje] = useState('');
  const [chat, setChat] = useState([]);
  const [modelo, setModelo] = useState(null);
  const [palabras, setPalabras] = useState([]);
  const [etiquetas, setEtiquetas] = useState([]);
  const [intenciones, setIntenciones] = useState([]);

  // Cargar modelo y datos necesarios
  useEffect(() => {
    const cargarDatos = async () => {
        try {
            // Carga el modelo como un modelo de gráfico
            const modeloCargado = await tf.loadGraphModel('/model.json');
            const intentsRes = await fetch('/intents.json').then(res => res.json());
            setModelo(modeloCargado);
            setIntents(intentsRes);


            // Agrega manualmente el vocabulario
            const palabrasDelModelo = []; 
            setPalabras(palabrasDelModelo);
        } catch (error) {
            console.error('Error al cargar los datos:', error);
        }
    };
    cargarDatos();
}, []);


  const manejarEnvio = async () => {
    if (!mensaje || !modelo) return;

    // Procesar entrada del usuario
    const tokens = dividirEnPalabras(mensaje);
    const vector = vectorBolsaDePalabras(tokens, palabras);
    const tensor = tf.tensor([vector]);

    // Hacer predicción con el modelo
    const prediccion = modelo.predict(tensor);
    const etiquetaIndex = prediccion.argMax(1).dataSync()[0];
    const confianza = prediccion.dataSync()[etiquetaIndex];

    if (confianza > 0.75) {
      const etiqueta = etiquetas[etiquetaIndex];
      const respuesta = intenciones.find((intencion) => intencion.tag === etiqueta).responses;

      // Procesar respuestas personalizadas como operaciones matemáticas
      if (["suma", "resta", "multiplicacion", "division"].includes(etiqueta)) {
        const numeros = mensaje.match(/\d+/g)?.map(Number) || [];
        if (numeros.length >= 2) {
          const [num1, num2] = numeros;
          let resultado = 0;
          switch (etiqueta) {
            case "suma":
              resultado = num1 + num2;
              break;
            case "resta":
              resultado = num1 - num2;
              break;
            case "multiplicacion":
              resultado = num1 * num2;
              break;
            case "division":
              resultado = num2 !== 0 ? num1 / num2 : "No puedo dividir entre cero.";
              break;
            default:
              break;
          }
          setChat([...chat, { usuario: mensaje, bot: `El resultado es ${resultado}` }]);
        } else {
          setChat([...chat, { usuario: mensaje, bot: "Faltan números para calcular." }]);
        }
      } else {
        setChat([...chat, { usuario: mensaje, bot: respuesta[0] }]);
      }
    } else {
      setChat([...chat, { usuario: mensaje, bot: "Lo siento, no entiendo eso." }]);
    }

    setMensaje('');
  };

  return (
    <div>
      <div>
        {chat.map((linea, idx) => (
          <div key={idx}>
            <b>Usuario:</b> {linea.usuario} <br />
            <b>Bot:</b> {linea.bot}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={mensaje}
        onChange={(e) => setMensaje(e.target.value)}
        placeholder="Escribe tu mensaje..."
      />
      <button onClick={manejarEnvio}>Enviar</button>
    </div>
  );
};

export default ChatBot;
