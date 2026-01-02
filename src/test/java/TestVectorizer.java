package com.escuela;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Prueba del vectorizador Word2Vec
 */
public class TestVectorizer {

    public static void main(String[] args) {
        System.out.println("=== PROBANDO WORD2VEC VECTORIZER ===\n");

        // Cargar datos
        DataLoader dataLoader = new DataLoader();
        dataLoader.cargarDesdeArchivo("emotions_train.csv");

        // Obtener longitud máxima de secuencia
        int maxLength = TextPreprocessor.obtenerLongitudMaxima(dataLoader.getTextos());
        System.out.println("Longitud máxima de secuencia: " + maxLength + " palabras");

        // Crear vectorizador
        int vectorSize = 100; // Usamos vectores de 50 dimensiones (más rápido para pruebas)
        Word2VecVectorizer vectorizer = new Word2VecVectorizer(vectorSize, maxLength);

        // Entrenar Word2Vec
        vectorizer.entrenar(dataLoader.getTextos());

        // Probar conversión de texto a vector
        System.out.println("\n--- PRUEBA DE VECTORIZACIÓN ---");
        String textoEjemplo = "Estoy muy feliz hoy";
        System.out.println("Texto original: " + textoEjemplo);

        INDArray vector = vectorizer.textoAVector(textoEjemplo);
        System.out.println("Forma del vector: " + java.util.Arrays.toString(vector.shape()));
        System.out.println("  [dimensiones]");
        System.out.println("  [" + vector.shape()[0] + ", " + vector.shape()[1] + ", " + vector.shape()[2] + "]");

        // Mostrar un fragmento del vector (primeras 5 dimensiones de la primera palabra)
        System.out.println("\nPrimeras 5 dimensiones de la primera palabra:");
        for (int i = 0; i < 5; i++) {
            double valor = vector.getDouble(0, i, 0);
            System.out.printf("  Dimensión %d: %.4f\n", i, valor);
        }

        // Probar palabras similares
        System.out.println("\n--- PALABRAS SIMILARES ---");
        String[] palabrasPrueba = {"feliz", "triste", "miedo"};

        for (String palabra : palabrasPrueba) {
            vectorizer.mostrarPalabrasSimilares(palabra, 5);
        }

        // Guardar el modelo
        String rutaModelo = "word2vec_emotions.zip";
        vectorizer.guardar(rutaModelo);

        System.out.println("\n✅ ¡Vectorizador funciona correctamente!");
    }
}
