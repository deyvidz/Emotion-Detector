package com.escuela;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
  Clase para entrenar Word2Vec y convertir texto a vectores numéricos
 */
public class Word2VecVectorizer {

    private Word2Vec word2Vec;
    private int vectorSize;        // Tamaño de cada vector (ej: 100 dimensiones)
    private int maxSequenceLength; // Longitud máxima de la secuencia (número de palabras)

    /**
      Constructor
      @param vectorSize tamaño del vector de cada palabra
      @param maxSequenceLength número máximo de palabras a considerar por frase
     */
    public Word2VecVectorizer(int vectorSize, int maxSequenceLength) {
        this.vectorSize = vectorSize;
        this.maxSequenceLength = maxSequenceLength;
    }

    /**
      Entrena el modelo Word2Vec con las frases proporcionadas
      @param textos lista de frases para entrenar
     */
    public void entrenar(List<String> textos) {
        System.out.println("\n=== ENTRENANDO WORD2VEC ===");
        System.out.println("Tamaño del vector: " + vectorSize);
        System.out.println("Número de frases: " + textos.size());

        // Preprocesar textos
        List<String> textosLimpios = new ArrayList<>();
        for (String texto : textos) {
            String limpio = TextPreprocessor.limpiarTexto(texto);
            if (!limpio.isEmpty()) {
                textosLimpios.add(limpio);
            }
        }

        // Crear iterador de frases (Word2Vec necesita esto)
        SentenceIterator iter = new CollectionSentenceIterator(textosLimpios);

        // Crear tokenizador (divide frases en palabras)
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        System.out.println("Construyendo modelo Word2Vec...");

        // Configuracion Word2Vec
        word2Vec = new Word2Vec.Builder()
                .minWordFrequency(1)           // Mínimo: una palabra debe aparecer 1 vez
                .iterations(20)                 // Número de iteraciones de entrenamiento
                .layerSize(vectorSize)         // Tamaño del vector de salida
                .seed(42)                      // Semilla para reproducibilidad
                .windowSize(2)                 // Ventana de contexto (palabras alrededor)
                .iterate(iter)                 // Datos de entrenamiento
                .tokenizerFactory(tokenizerFactory)
                .build();

        System.out.println("Entrenando...");
        word2Vec.fit();                        //se inicializa el modelo

        System.out.println("✅ Word2Vec entrenado");
        System.out.println("Vocabulario: " + word2Vec.getVocab().numWords() + " palabras");
    }

    /**
      Convierte un texto a una matriz numérica (tensor)
      Esta es la representación que la red neuronal puede procesar
      @param texto la frase a convertir
      @return matriz de tamaño [1, vectorSize, maxSequenceLength]
     */
    public INDArray textoAVector(String texto) {
        // Tokeniza: "estoy muy feliz" → ["estoy", "muy", "feliz"]
        List<String> tokens = TextPreprocessor.tokenizar(texto);

        // Crear matriz: [1 ejemplo, vectorSize dimensiones, maxSequenceLength palabras]
        INDArray features = Nd4j.zeros(1, vectorSize, maxSequenceLength);

        // Llenar la matriz con los vectores de cada palabra
        for (int i = 0; i < Math.min(tokens.size(), maxSequenceLength); i++) {
            String palabra = tokens.get(i);

            // Obtener el vector de la palabra desde Word2Vec
            if (word2Vec.hasWord(palabra)) {
                INDArray wordVector = word2Vec.getWordVectorMatrix(palabra);

                // Copiar el vector a la posición correspondiente
                for (int j = 0; j < vectorSize; j++) {
                    features.putScalar(new int[]{0, j, i}, wordVector.getDouble(j));
                }
            }
            // Si la palabra no existe en el vocabulario, se queda en ceros
        }

        return features;
    }

    /**
      Convierte múltiples textos a una matriz
      @param textos lista de frases
      @return matriz de tamaño [numTextos, vectorSize, maxSequenceLength]
     */
    public INDArray textosAVectores(List<String> textos) {
        INDArray features = Nd4j.zeros(textos.size(), vectorSize, maxSequenceLength);

        for (int i = 0; i < textos.size(); i++) {
            INDArray vectorTexto = textoAVector(textos.get(i));

            // Copiar cada vector de texto en la posición correcta
            // vectorTexto tiene forma [1, vectorSize, maxSequenceLength]
            // Necesitamos copiarlo en features[i, :, :]
            for (int j = 0; j < vectorSize; j++) {
                for (int k = 0; k < maxSequenceLength; k++) {
                    double valor = vectorTexto.getDouble(0, j, k);
                    features.putScalar(new int[]{i, j, k}, valor);
                }
            }
        }

        return features;
    }

    /**
      Guarda el modelo Word2Vec en un archivo
     */
    public void guardar(String rutaArchivo) {
        try {
            File archivo = new File(rutaArchivo);
            WordVectorSerializer.writeWord2VecModel(word2Vec, archivo);
            System.out.println("✅ Modelo Word2Vec guardado en: " + rutaArchivo);
        } catch (Exception e) {
            System.err.println("❌ Error al guardar Word2Vec: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
      Carga un modelo Word2Vec entrenado
     */
    public void cargar(String rutaArchivo) {
        try {
            File archivo = new File(rutaArchivo);
            word2Vec = WordVectorSerializer.readWord2VecModel(archivo);
            System.out.println("✅ Modelo Word2Vec cargado desde: " + rutaArchivo);
        } catch (Exception e) {
            System.err.println("❌ Error al cargar Word2Vec: " + e.getMessage());
            e.printStackTrace();
        }
    }



    // Getters
    public int getVectorSize() {
        return vectorSize;
    }

    public int getMaxSequenceLength() {
        return maxSequenceLength;
    }

    public Word2Vec getWord2Vec() {
        return word2Vec;
    }
}