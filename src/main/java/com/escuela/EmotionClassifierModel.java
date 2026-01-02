package com.escuela;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
  Clase que define y configura la red neuronal LSTM para clasificación de emociones
 */
public class EmotionClassifierModel {

    private MultiLayerNetwork model;
    private int inputSize;      // Tamaño del vector de entrada (dimensiones de Word2Vec)
    private int numClasses;     // Número de emociones (5 en este caso)

    /**
      Constructor
      @param inputSize tamaño del vector de Word2Vec (ej.: 100)
      @param numClasses número de clases/emociones (5)
     */
    public EmotionClassifierModel(int inputSize, int numClasses) {
        this.inputSize = inputSize;
        this.numClasses = numClasses;
    }

    /**
      Construye la arquitectura de la red neuronal
     */
    public void construirModelo() {
        System.out.println("\n=== CONSTRUYENDO MODELO LSTM ===");
        System.out.println("Tamaño de entrada: " + inputSize);
        System.out.println("Número de clases: " + numClasses);

        // Configuración de la red neuronal
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)

                .updater(new Adam(0.01)) // Optimizador Adam: algoritmo inteligente para ajustar pesos
                                                    // learningRate = qué tan rápido aprende (0.01 es un buen valor)

                .weightInit(WeightInit.XAVIER)      // Inicialización de pesos (valores iniciales aleatorios)

                .list()                    // Lista de capas de la red

                // CAPA 1: Primera capa LSTM con 128 neuronas patrones simples
                .layer(0, new LSTM.Builder()
                        .nIn(inputSize)           // Entrada: vectores de Word2Vec
                        .nOut(128)                // Salida: 128 neuronas
                        .activation(Activation.TANH)  // Función de activación aprende patrones no lineales
                        .build())

                // CAPA 2: Segunda capa LSTM con 64 neuronas patrones complejos
                .layer(1, new LSTM.Builder()
                        .nIn(128)                 // Entrada: salida de capa anterior
                        .nOut(64)                 // Salida: 64 neuronas
                        .activation(Activation.TANH)
                        .build())

                // CAPA 3 densa: Capa de salida (clasificación)
                .layer(2, new RnnOutputLayer.Builder()
                        .nIn(64)                  // Entrada: salida de capa anterior
                        .nOut(numClasses)         // Salida: 5 emociones
                        .activation(Activation.SOFTMAX)  // Softmax: convierte a probabilidades
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  // Función de pérdida Si predice correctamente: pérdida baja (cercana a 0)
                        .build())

                .build();

        // Crea modelo con la configuración
        model = new MultiLayerNetwork(conf);
        model.init();

        // Listener para ver el progreso cada 10 iteraciones
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("✅ Modelo construido");
        System.out.println("Total de parámetros: " + model.numParams());
        System.out.println("\n--- ARQUITECTURA DEL MODELO ---");
        System.out.println(model.summary());
    }

    /**
      Obtiene el modelo (para entrenamiento y predicción)
     */
    public MultiLayerNetwork getModel() {
        return model;
    }

    /**
      Guarda el modelo entrenado en un archivo
     */
    public void guardar(String rutaArchivo) {
        try {
            File archivo = new File(rutaArchivo);
            model.save(archivo, true);  // true = guardar también el updater
            System.out.println("✅ Modelo guardado en: " + rutaArchivo);
        } catch (IOException e) {
            System.err.println("❌ Error al guardar modelo: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
      Carga un modelo entrenado
     */
    public void cargar(String rutaArchivo) {
        try {
            File archivo = new File(rutaArchivo);
            model = MultiLayerNetwork.load(archivo, true);
            System.out.println("✅ Modelo cargado desde: " + rutaArchivo);
        } catch (IOException e) {
            System.err.println("❌ Error al cargar modelo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
