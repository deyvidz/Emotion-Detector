package com.escuela;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
  Clase para cargar el modelo entrenado y predecir emociones en frases nuevas
 */
public class EmotionPredictor {

    private MultiLayerNetwork model;
    private Word2VecVectorizer vectorizer;
    private Map<Integer, String> indiceAEmocion;
    private Map<Integer, String> emocionAEmoji;

    /**
      Constructor
     */
    public EmotionPredictor() {
        // Mapeo de índices a emociones
        indiceAEmocion = new HashMap<>();
        indiceAEmocion.put(0, "joy");
        indiceAEmocion.put(1, "sadness");
        indiceAEmocion.put(2, "anger");
        indiceAEmocion.put(3, "fear");
        indiceAEmocion.put(4, "surprise");

        // Mapeo de emociones a emojis (para mejor visualización)
        emocionAEmoji = new HashMap<>();
        emocionAEmoji.put(0, "😊");
        emocionAEmoji.put(1, "😢");
        emocionAEmoji.put(2, "😠");
        emocionAEmoji.put(3, "😨");
        emocionAEmoji.put(4, "😲");
    }

    /**
      Carga el modelo y Word2Vec previamente entrenados
     */
    public boolean cargarModelo(String rutaModelo, String rutaWord2Vec) {
        try {
            System.out.println("📦 Cargando modelo entrenado...");

            // Cargar el modelo LSTM
            File archivoModelo = new File(rutaModelo);
            if (!archivoModelo.exists()) {
                System.err.println("❌ No se encontró el archivo del modelo: " + rutaModelo);
                return false;
            }
            model = MultiLayerNetwork.load(archivoModelo, true);
            System.out.println("✅ Modelo LSTM cargado");

            // Cargar Word2Vec
            File archivoWord2Vec = new File(rutaWord2Vec);
            if (!archivoWord2Vec.exists()) {
                System.err.println("❌ No se encontró el archivo Word2Vec: " + rutaWord2Vec);
                return false;
            }

            // El vectorSize y maxSequenceLength deben coincidir con los usados en entrenamiento
            int vectorSize = 100;
            int maxSequenceLength = 8;

            vectorizer = new Word2VecVectorizer(vectorSize, maxSequenceLength);
            vectorizer.cargar(rutaWord2Vec);
            System.out.println("✅ Word2Vec cargado");

            return true;

        } catch (Exception e) {
            System.err.println("❌ Error al cargar modelo: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
      Predice la emoción de un texto
      @param texto la frase a analizar
      @return resultado de la predicción con probabilidades
     */
    public ResultadoPrediccion predecir(String texto) {
        if (model == null || vectorizer == null) {
            System.err.println("❌ El modelo no está cargado. Llama a cargarModelo() primero.");
            return null;
        }

        // Vectorizar el texto
        INDArray features = vectorizer.textoAVector(texto);

        // Hacer predicción
        INDArray output = model.output(features);
        INDArray promedioTemporal = output.mean(2); // [1, numClases]

        // Obtener probabilidades para cada emoción
        double[] probabilidades = new double[5];
        for (int i = 0; i < 5; i++) {
            probabilidades[i] = promedioTemporal.getDouble(0, i);
        }

        // Obtener la emoción con mayor probabilidad
        int indiceMaximo = Nd4j.argMax(promedioTemporal, 1).getInt(0);
        String emocionPredicha = indiceAEmocion.get(indiceMaximo);
        String emoji = emocionAEmoji.get(indiceMaximo);
        double confianza = probabilidades[indiceMaximo];

        return new ResultadoPrediccion(
                texto,
                emocionPredicha,
                emoji,
                confianza,
                probabilidades
        );
    }

    /**
      Muestra el resultado de la predicción
     */
    public void mostrarResultado(ResultadoPrediccion resultado) {
        System.out.println("\n╔════════════════════════════════════════════════╗");
        System.out.println("║           RESULTADO DE LA PREDICCIÓN          ║");
        System.out.println("╚════════════════════════════════════════════════╝");
        System.out.println("\n📝 Texto: \"" + resultado.getTexto() + "\"");
        System.out.println("\n🎯 Emoción detectada: " + resultado.getEmocionPredicha().toUpperCase() + " " + resultado.getEmoji());
        System.out.printf("   Confianza: %.1f%%\n", resultado.getConfianza() * 100);

        System.out.println("\n📊 Probabilidades por emoción:");
        String[] emociones = {"Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa"};
        String[] emojis = {"😊", "😢", "😠", "😨", "😲"};

        for (int i = 0; i < 5; i++) {
            double prob = resultado.getProbabilidades()[i];
            String barra = generarBarraProgreso(prob);
            System.out.printf("   %s %-10s %s %.1f%%\n",
                    emojis[i], emociones[i], barra, prob * 100);
        }
        System.out.println();
    }

    /**
      Genera una barra de progreso visual
     */
    private String generarBarraProgreso(double porcentaje) {
        int barras = (int) (porcentaje * 20); // 20 caracteres máximo
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 20; i++) {
            if (i < barras) {
                sb.append("█");
            } else {
                sb.append("░");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
      Clase interna para almacenar resultados de predicción
     */
    public static class ResultadoPrediccion {
        private String texto;
        private String emocionPredicha;
        private String emoji;
        private double confianza;
        private double[] probabilidades;

        public ResultadoPrediccion(String texto, String emocionPredicha, String emoji,
                                   double confianza, double[] probabilidades) {
            this.texto = texto;
            this.emocionPredicha = emocionPredicha;
            this.emoji = emoji;
            this.confianza = confianza;
            this.probabilidades = probabilidades;
        }

        // Getters
        public String getTexto() { return texto; }
        public String getEmocionPredicha() { return emocionPredicha; }
        public String getEmoji() { return emoji; }
        public double getConfianza() { return confianza; }
        public double[] getProbabilidades() { return probabilidades; }
    }
}