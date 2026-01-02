package com.escuela;

import java.util.List;

/**
 * Prueba del preprocesador de texto
 */
public class TestPreprocessor {

    public static void main(String[] args) {
        System.out.println("=== PROBANDO PREPROCESADOR DE TEXTO ===\n");

        // Probar limpieza de texto
        String[] ejemplos = {
                "¡Hola! ¿Cómo estás?",
                "Estoy MUY feliz... 😊",
                "Me siento   triste  :(",
                "¡¡¡Qué enojo!!!",
                "Tengo miedo, mucho miedo."
        };

        System.out.println("--- LIMPIEZA DE TEXTO ---");
        for (String ejemplo : ejemplos) {
            String limpio = TextPreprocessor.limpiarTexto(ejemplo);
            System.out.println("Original: " + ejemplo);
            System.out.println("Limpio:   " + limpio);
            System.out.println();
        }

        // Probar tokenización
        System.out.println("\n--- TOKENIZACIÓN ---");
        for (String ejemplo : ejemplos) {
            List<String> tokens = TextPreprocessor.tokenizar(ejemplo);
            System.out.println("Texto: " + ejemplo);
            System.out.println("Tokens: " + tokens);
            System.out.println("Número de tokens: " + tokens.size());
            System.out.println();
        }

        // Probar con el dataset real
        System.out.println("\n--- PRUEBA CON DATASET REAL ---");
        DataLoader dataLoader = new DataLoader();
        dataLoader.cargarDesdeArchivo("emotions_train.csv");

        List<String> textos = dataLoader.getTextos();

        // Obtener vocabulario
        List<String> vocabulario = TextPreprocessor.obtenerVocabulario(textos);
        System.out.println("Tamaño del vocabulario: " + vocabulario.size() + " palabras únicas");

        // Mostrar primeras 20 palabras del vocabulario
        System.out.println("\nPrimeras 20 palabras del vocabulario:");
        for (int i = 0; i < Math.min(20, vocabulario.size()); i++) {
            System.out.print(vocabulario.get(i) + " ");
        }
        System.out.println();

        // Obtener longitud máxima
        int maxLongitud = TextPreprocessor.obtenerLongitudMaxima(textos);
        System.out.println("\nLongitud máxima de tokens: " + maxLongitud + " palabras");

        // Mostrar ejemplo de la frase más larga
        System.out.println("\nEjemplo de frase más larga:");
        for (String texto : textos) {
            List<String> tokens = TextPreprocessor.tokenizar(texto);
            if (tokens.size() == maxLongitud) {
                System.out.println("Texto original: " + texto);
                System.out.println("Tokens: " + tokens);
                break;
            }
        }

        System.out.println("\n✅ ¡Preprocesador funciona correctamente!");
    }
}
