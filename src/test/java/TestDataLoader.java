package com.escuela;

/**
 * Clase para probar que el DataLoader funciona correctamente
 */
public class TestDataLoader {

    public static void main(String[] args) {
        System.out.println("=== PROBANDO CARGA DE DATOS ===\n");

        // Crear instancia del cargador de datos
        DataLoader dataLoader = new DataLoader();

        // Intentar cargar el archivo de entrenamiento
        boolean exito = dataLoader.cargarDesdeArchivo("emotions_train.csv");

        if (!exito) {
            System.err.println("❌ No se pudo cargar el archivo");
            return;
        }

        // Mostrar estadísticas
        dataLoader.mostrarEstadisticas();

        // Mostrar algunos ejemplos
        System.out.println("=== EJEMPLOS DEL DATASET ===");
        int numEjemplos = Math.min(5, dataLoader.getNumeroEjemplos());

        for (int i = 0; i < numEjemplos; i++) {
            String texto = dataLoader.getTexto(i);
            String emocion = dataLoader.getEmocion(i);
            int indice = dataLoader.emocionANumero(emocion);

            System.out.println("\nEjemplo " + (i + 1) + ":");
            System.out.println("  Texto: " + texto);
            System.out.println("  Emoción: " + emocion);
            System.out.println("  Índice numérico: " + indice);
        }

        // Probar conversión de número a emoción
        System.out.println("\n=== PRUEBA DE CONVERSIONES ===");
        for (int i = 0; i < 5; i++) {
            System.out.println("Índice " + i + " → " + dataLoader.numeroAEmocion(i));
        }

        System.out.println("\n✅ ¡DataLoader funciona correctamente!");
    }
}
