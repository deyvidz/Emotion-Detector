package com.escuela;

/**
 * Prueba de construcción del modelo LSTM
 */
public class TestModel {

    public static void main(String[] args) {
        System.out.println("=== PROBANDO CONSTRUCCIÓN DEL MODELO ===\n");

        // Parámetros del modelo
        int vectorSize = 50;    // Tamaño de vectores de Word2Vec
        int numEmociones = 5;   // joy, sadness, anger, fear, surprise

        // Crear instancia del modelo
        EmotionClassifierModel classifierModel = new EmotionClassifierModel(
                vectorSize,
                numEmociones
        );

        // Construir la arquitectura
        classifierModel.construirModelo();

        System.out.println("\n✅ ¡Modelo creado exitosamente!");
        System.out.println("\nEl modelo está listo para ser entrenado.");
    }
}
