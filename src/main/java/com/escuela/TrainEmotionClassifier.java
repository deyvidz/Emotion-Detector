package com.escuela;

/**
  Programa principal para entrenar el clasificador de emociones
 */
public class TrainEmotionClassifier {

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════╗");
        System.out.println("║  CLASIFICADOR DE EMOCIONES CON DEEPLEARNING4J   ║");
        System.out.println("║           Proyecto de Machine Learning            ║");
        System.out.println("╚═══════════════════════════════════════════════════╝\n");

        // ============================================
        // PASO 1: CARGAR DATOS
        // ============================================
        System.out.println("📂 PASO 1: Cargando datos...\n");

        DataLoader trainData = new DataLoader();
        DataLoader testData = new DataLoader();

        if (!trainData.cargarDesdeArchivo("emotions_train.csv")) {
            System.err.println("❌ Error al cargar datos de entrenamiento");
            return;
        }

        if (!testData.cargarDesdeArchivo("emotions_test.csv")) {
            System.err.println("❌ Error al cargar datos de prueba");
            return;
        }

        trainData.mostrarEstadisticas();

        // ============================================
        // PASO 2: ENTRENAR WORD2VEC
        // ============================================
        System.out.println("\n📚 PASO 2: Entrenando Word2Vec...\n");

        int maxLength = TextPreprocessor.obtenerLongitudMaxima(trainData.getTextos());
        int vectorSize = 100; // Tamaño de vectores de palabras

        Word2VecVectorizer vectorizer = new Word2VecVectorizer(vectorSize, maxLength);
        vectorizer.entrenar(trainData.getTextos());

        // Guardar Word2Vec
        vectorizer.guardar("word2vec_emotions.zip");

        // ============================================
        // PASO 3: CONSTRUIR MODELO LSTM
        // ============================================
        System.out.println("\n🧠 PASO 3: Construyendo modelo LSTM...\n");

        EmotionClassifierModel classifierModel = new EmotionClassifierModel(
                vectorSize,
                trainData.getNumeroClases()
        );
        classifierModel.construirModelo();

        // ============================================
        // PASO 4: ENTRENAR MODELO
        // ============================================
        System.out.println("\n🏋️ PASO 4: Entrenando modelo...\n");

        ModelTrainer trainer = new ModelTrainer(
                classifierModel.getModel(),
                vectorizer,
                trainData,
                testData
        );

        // Configuración de entrenamiento
        int numEpochs = 150;      // Número de epochs
        int batchSize = 20;       // Tamaño de batch

        trainer.entrenar(numEpochs, batchSize);

        // ============================================
        // PASO 5: GUARDAR MODELO
        // ============================================
        System.out.println("\n💾 PASO 5: Guardando modelo entrenado...\n");

        classifierModel.guardar("emotion_classifier_model.zip");

        // ============================================
        // FINALIZACIÓN
        // ============================================
        System.out.println("\n╔═══════════════════════════════════════════════════╗");
        System.out.println("║            ✅ ENTRENAMIENTO COMPLETADO            ║");
        System.out.println("╚═══════════════════════════════════════════════════╝\n");

        System.out.println("Archivos generados:");
        System.out.println("  📄 word2vec_emotions.zip - Modelo Word2Vec");
        System.out.println("  📄 emotion_classifier_model.zip - Modelo clasificador");
        System.out.println("\n¡Tu modelo está listo para hacer predicciones! 🎉\n");
    }
}
