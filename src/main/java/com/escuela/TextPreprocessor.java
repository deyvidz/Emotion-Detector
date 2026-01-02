package com.escuela;

import java.text.Normalizer;
import java.util.ArrayList;

import java.util.List;

/**
  Clase para preprocesar y tokenizar texto
  Tokenizar = dividir texto en palabras individuales
 */
public class TextPreprocessor {

    /**
      Limpia y normaliza el texto
      - Convierte a minúsculas
      - Elimina acentos
      - Elimina puntuación
      - Elimina espacios extra
     */
    public static String limpiarTexto(String texto) {
        if (texto == null || texto.isEmpty()) {
            return "";
        }

        // Convertir a minúsculas
        texto = texto.toLowerCase();

        // Eliminar acentos (á → a, é → e, etc.)
        texto = Normalizer.normalize(texto, Normalizer.Form.NFD);
        texto = texto.replaceAll("\\p{M}", "");

        // Eliminar caracteres especiales y puntuación, mantener solo letras y espacios
        texto = texto.replaceAll("[^a-z\\s]", " ");

        // Eliminar espacios múltiples
        texto = texto.replaceAll("\\s+", " ");

        // Eliminar espacios al inicio y final
        texto = texto.trim();

        return texto;
    }

    /**
      Divide el texto en palabras (tokens)
      @param texto el texto a tokenizar
      @return lista de palabras
     */
    public static List<String> tokenizar(String texto) {
        String textoLimpio = limpiarTexto(texto);

        if (textoLimpio.isEmpty()) {
            return new ArrayList<>();
        }

        // Dividir por espacios
        String[] palabras = textoLimpio.split("\\s+");

        // Filtrar palabras muy cortas (menos de 2 caracteres)
        List<String> tokens = new ArrayList<>();
        for (String palabra : palabras) {
            if (palabra.length() >= 2) {
                tokens.add(palabra);
            }
        }

        return tokens;
    }

    /**
      Obtiene todas las palabras únicas de una lista de textos
      Esto se usa para construir el vocabulario
     */
    public static List<String> obtenerVocabulario(List<String> textos) {
        List<String> vocabulario = new ArrayList<>();

        for (String texto : textos) {
            List<String> tokens = tokenizar(texto);
            for (String token : tokens) {
                if (!vocabulario.contains(token)) {
                    vocabulario.add(token);
                }
            }
        }

        return vocabulario;
    }

    /**
      Calcula la longitud máxima de tokens en una lista de textos
      Necesario para hacer que todas las frases tengan el mismo largo (padding)
     */
    public static int obtenerLongitudMaxima(List<String> textos) {
        int maxLongitud = 0;

        for (String texto : textos) {
            List<String> tokens = tokenizar(texto);
            if (tokens.size() > maxLongitud) {
                maxLongitud = tokens.size();
            }
        }

        return maxLongitud;
    }
}