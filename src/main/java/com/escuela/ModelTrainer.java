package com.escuela;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
  Clase para entrenar el modelo de clasificación de emociones
 */
public class ModelTrainer {

    private MultiLayerNetwork model;
    private Word2VecVectorizer vectorizer;
    private DataLoader trainData;
    private DataLoader testData;

    /**
      Constructor
     */
    public ModelTrainer(MultiLayerNetwork model,
                        Word2VecVectorizer vectorizer,
                        DataLoader trainData,
                        DataLoader testData) {
        this.model = model;
        this.vectorizer = vectorizer;
        this.trainData = trainData;
        this.testData = testData;
    }

    /**
      Prepara las etiquetas (labels) en formato one-hot encoding
      One-hot encoding convierte:
     "joy" (índice 0) → [1, 0, 0, 0, 0]
     "sadness" (índice 1) → [0, 1, 0, 0, 0]
     etc.
     */
    private INDArray prepararLabels(List<String> emociones, int numClases) {
        int numEjemplos = emociones.size();
        INDArray labels = Nd4j.zeros(numEjemplos, numClases, vectorizer.getMaxSequenceLength());

        for (int i = 0; i < numEjemplos; i++) {
            String emocion = emociones.get(i);
            int claseIndex = trainData.emocionANumero(emocion);

            // Marca la clase correcta con 1 en las posiciones de tiempo
            for (int t = 0; t < vectorizer.getMaxSequenceLength(); t++) {
                labels.putScalar(new int[]{i, claseIndex, t}, 1.0);
            }
        }

        return labels;
    }

    /**
      Entrena el modelo
      @param numEpochs número de veces que verá todos los datos
      @param batchSize número de ejemplos a procesar simultáneamente
     */
    public void entrenar(int numEpochs, int batchSize) {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║   INICIANDO ENTRENAMIENTO DEL MODELO   ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        System.out.println("Configuración:");
        System.out.println("  • Epochs: " + numEpochs);
        System.out.println("  • Batch size: " + batchSize);
        System.out.println("  • Ejemplos de entrenamiento: " + trainData.getNumeroEjemplos());
        System.out.println("  • Ejemplos de prueba: " + testData.getNumeroEjemplos());
        System.out.println();

        // Preparar datos de entrenamiento
        System.out.println("Preparando datos...");
        List<String> textos = trainData.getTextos();
        List<String> emociones = trainData.getEmociones();

        // Crear índices para mezclar los datos
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < textos.size(); i++) {
            indices.add(i);
        }

        // Entrenar por cada epoch
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Mezclar datos al inicio de cada epoch (importante para aprendizaje)
            Collections.shuffle(indices);

            double totalLoss = 0;
            int numBatches = 0;

            // Procesar en batches
            for (int i = 0; i < textos.size(); i += batchSize) {
                int endIdx = Math.min(i + batchSize, textos.size());

                // Crear batch
                List<String> batchTextos = new ArrayList<>();
                List<String> batchEmociones = new ArrayList<>();

                for (int j = i; j < endIdx; j++) {
                    int idx = indices.get(j);
                    batchTextos.add(textos.get(idx));
                    batchEmociones.add(emociones.get(idx));
                }

                // Vectorizar el batch
                INDArray features = vectorizer.textosAVectores(batchTextos);
                INDArray labels = prepararLabels(batchEmociones, trainData.getNumeroClases());

                // Entrenar con este batch
                model.fit(features, labels);

                // Acumular pérdida para estadísticas
                totalLoss += model.score();
                numBatches++;
            }

            // Calcular pérdida promedio del epoch
            double avgLoss = totalLoss / numBatches;

            // Evaluar en datos de prueba cada 10 epochs
            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                double accuracy = evaluarPrecision();
                System.out.printf("Epoch %3d/%d - Loss: %.4f - Precisión Test: %.2f%%\n",
                        epoch + 1, numEpochs, avgLoss, accuracy * 100);
            } else {
                System.out.printf("Epoch %3d/%d - Loss: %.4f\n",
                        epoch + 1, numEpochs, avgLoss);
            }
        }

        System.out.println("\n✅ ¡Entrenamiento completado!\n");

    }

    /**
      Evalúa la precisión del modelo en el conjunto de prueba
     */
    private double evaluarPrecision() {
        List<String> textosTest = testData.getTextos();
        List<String> emocionesTest = testData.getEmociones();

        int correctos = 0;

        for (int i = 0; i < textosTest.size(); i++) {
            String texto = textosTest.get(i);
            String emocionReal = emocionesTest.get(i);
            INDArray features = vectorizer.textoAVector(texto);
            INDArray output = model.output(features);

            // Obtener la clase con mayor probabilidad
            // Promedio de aciertos
            INDArray promedioTemporal = output.mean(2); // [1, numClases]
            int prediccion = Nd4j.argMax(promedioTemporal, 1).getInt(0);

            String emocionPredicha = testData.numeroAEmocion(prediccion);

            if (emocionReal.equals(emocionPredicha)) {
                correctos++;
            }
        }

        return (double) correctos / textosTest.size();
    }
}