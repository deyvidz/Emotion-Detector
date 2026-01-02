package com.escuela;

import java.util.Scanner;

/**
  Programa principal para usar el clasificador de emociones para hacer predicciones
 */
public class EmotionDetectorMain {

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════╗");
        System.out.println("║       DETECTOR DE EMOCIONES EN TEXTO 😊😢😠😨😲    ║");
        System.out.println("║              Powered by DeepLearning4J            ║");
        System.out.println("╚═══════════════════════════════════════════════════╝\n");

        // Crear el predictor
        EmotionPredictor predictor = new EmotionPredictor();

        // Cargar el modelo entrenado
        System.out.println("Cargando modelo entrenado...\n");
        boolean exito = predictor.cargarModelo(
                "emotion_classifier_model.zip",
                "word2vec_emotions.zip"
        );

        if (!exito) {
            System.err.println("\n❌ No se pudo cargar el modelo.");
            System.err.println("Asegúrate de haber ejecutado TrainEmotionClassifier primero.");
            return;
        }

        System.out.println("\n✅ Modelo cargado y listo para usar!\n");

        // Modo interactivo
        modoInteractivo(predictor);
    }

    /**
      Modo interactivo: el usuario puede escribir frases y ver predicciones
     */
    private static void modoInteractivo(EmotionPredictor predictor) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("═══════════════════════════════════════════════════");
        System.out.println("         MODO INTERACTIVO ACTIVADO");
        System.out.println("═══════════════════════════════════════════════════");
        System.out.println("\nEscribe una frase y te diré qué emoción expresa.");
        System.out.println("Escribe 'ejemplos' para ver ejemplos predefinidos.");
        System.out.println("Escribe 'salir' para terminar.\n");

        while (true) {
            System.out.print("💬 Escribe una frase: ");
            String input = scanner.nextLine().trim();

            if (input.isEmpty()) {
                continue;
            }

            if (input.equalsIgnoreCase("salir") || input.equalsIgnoreCase("exit")) {
                System.out.println("\n👋 ¡Hasta luego! Gracias por usar el detector de emociones.\n");
                break;
            }

            if (input.equalsIgnoreCase("ejemplos")) {
                probarEjemplos(predictor);
                continue;
            }

            // Hacer predicción
            EmotionPredictor.ResultadoPrediccion resultado = predictor.predecir(input);
            predictor.mostrarResultado(resultado);
        }

        scanner.close();
    }

    /**
      Prueba el modelo con ejemplos predefinidos
     */
    private static void probarEjemplos(EmotionPredictor predictor) {
        System.out.println("\n╔════════════════════════════════════════════════╗");
        System.out.println("║              EJEMPLOS PREDEFINIDOS             ║");
        System.out.println("╚════════════════════════════════════════════════╝\n");

        String[] ejemplos = {
                "Estoy muy feliz con mi nuevo trabajo",
                "Me siento triste y solo",
                "Esto me pone muy furioso",
                "Tengo mucho miedo de fracasar",
                "No puedo creer lo que acaba de pasar",
                "Hoy es el mejor día de mi vida",
                "Extraño mucho a mi familia",
                "Tu actitud me molesta",
                "Me da terror esa situación",
                "Qué sorpresa tan increíble"
        };

        for (int i = 0; i < ejemplos.length; i++) {
            System.out.println("─────────────────────────────────────────────────");
            System.out.println("Ejemplo " + (i + 1) + " de " + ejemplos.length);

            EmotionPredictor.ResultadoPrediccion resultado = predictor.predecir(ejemplos[i]);
            predictor.mostrarResultado(resultado);

            // Pausa pequeña para leer
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // Ignorar
            }
        }

        System.out.println("─────────────────────────────────────────────────\n");
    }
}
