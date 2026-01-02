package com.escuela;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
  Clase para cargar y procesar el dataset de emociones desde archivos CSV
 */
public class DataLoader {

    // Lista que almacena todas las frases del dataset
    private List<String> textos;

    // Lista que almacena las emociones correspondientes a cada frase
    private List<String> emociones;

    // Mapeo de emociones a números (la red neuronal trabaja con números, no con palabras)
    private Map<String, Integer> emocionAIndice;

    // Mapeo inverso: de números a emociones (para convertir predicciones a texto)
    private Map<Integer, String> indiceAEmocion;

    /**
      Constructor: inicializa las listas
     */
    public DataLoader() {
        this.textos = new ArrayList<>();
        this.emociones = new ArrayList<>();
        this.emocionAIndice = new HashMap<>();
        this.indiceAEmocion = new HashMap<>();

        // Definir el mapeo de emociones a índices numéricos
        // Cada emoción tiene un número único
        emocionAIndice.put("joy", 0);
        emocionAIndice.put("sadness", 1);
        emocionAIndice.put("anger", 2);
        emocionAIndice.put("fear", 3);
        emocionAIndice.put("surprise", 4);

        // Crear el mapeo inverso
        for (Map.Entry<String, Integer> entry : emocionAIndice.entrySet()) {
            indiceAEmocion.put(entry.getValue(), entry.getKey());
        }
    }

    /**
      Carga los datos desde un archivo CSV
      @param nombreArchivo nombre del archivo dentro de resources/data/emotions/
      @return true si se cargó correctamente, false si hubo error
     */
    public boolean cargarDesdeArchivo(String nombreArchivo) {
        try {
            // Obtener el archivo desde resources
            InputStream inputStream = getClass().getClassLoader()
                    .getResourceAsStream("data/emotions/" + nombreArchivo);

            if (inputStream == null) {
                System.err.println("❌ No se encontró el archivo: " + nombreArchivo);
                return false;
            }

            // Leer el archivo línea por línea
            BufferedReader reader = new BufferedReader(
            new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            String linea;
            boolean primeraLinea = true;
            int lineaNumero = 0;

            while ((linea = reader.readLine()) != null) {
                lineaNumero++;

                // Saltar la primera línea (encabezado: "texto,emocion")
                if (primeraLinea) {
                    primeraLinea = false;
                    continue;
                }

                // Dividir la línea por comas
                String[] partes = linea.split(",", 2);

                if (partes.length != 2) {
                    System.err.println("⚠️ Línea " + lineaNumero + " mal formateada: " + linea);
                    continue;
                }

                String texto = partes[0].trim();
                String emocion = partes[1].trim().toLowerCase();

                // Verificar que la emoción sea válida
                if (!emocionAIndice.containsKey(emocion)) {
                    System.err.println("⚠️ Emoción desconocida en línea " + lineaNumero + ": " + emocion);
                    continue;
                }

                // Agregar a las listas
                textos.add(texto);
                emociones.add(emocion);
            }

            reader.close();

            System.out.println("✅ Cargados " + textos.size() + " ejemplos desde " + nombreArchivo);
            return true;

        } catch (IOException e) {
            System.err.println("❌ Error al leer archivo: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
      Obtiene el número total de ejemplos cargados
     */
    public int getNumeroEjemplos() {
        return textos.size();
    }

    /**
      Obtiene un texto específico por su índice
     */
    public String getTexto(int indice) {
        return textos.get(indice);
    }

    /**
      Obtiene una emoción específica por su índice
     */
    public String getEmocion(int indice) {
        return emociones.get(indice);
    }

    /**
      Convierte una emoción (texto) a su índice numérico
     */
    public int emocionANumero(String emocion) {
        return emocionAIndice.get(emocion.toLowerCase());
    }

    /**
      Convierte un índice numérico a su emoción (texto)
     */
    public String numeroAEmocion(int indice) {
        return indiceAEmocion.get(indice);
    }

    /**
      Obtiene el número total de clases/emociones diferentes
     */
    public int getNumeroClases() {
        return emocionAIndice.size();
    }

    /**
      Muestra estadísticas del dataset cargado
     */
    public void mostrarEstadisticas() {
        System.out.println("\n=== ESTADÍSTICAS DEL DATASET ===");
        System.out.println("Total de ejemplos: " + textos.size());
        System.out.println("Número de clases: " + emocionAIndice.size());

        // Contar cuántos ejemplos hay de cada emoción
        Map<String, Integer> conteo = new HashMap<>();
        for (String emocion : emociones) {
            conteo.put(emocion, conteo.getOrDefault(emocion, 0) + 1);
        }

        System.out.println("\nDistribución por emoción:");
        for (Map.Entry<String, Integer> entry : conteo.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " ejemplos");
        }
        System.out.println("================================\n");
    }

    // Getters para acceder a las listas completas
    public List<String> getTextos() {
        return textos;
    }

    public List<String> getEmociones() {
        return emociones;
    }

    public Map<String, Integer> getEmocionAIndice() {
        return emocionAIndice;
    }
}