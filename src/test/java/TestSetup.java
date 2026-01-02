package com.escuela;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Clase simple para verificar que DeepLearning4j está configurado correctamente
 */
public class TestSetup {

    public static void main(String[] args) {
        System.out.println("=== VERIFICANDO INSTALACIÓN ===");
        System.out.println("Versión de Java: " + System.getProperty("java.version"));

        try {
            // Crear una matriz simple con ND4J (backend de DL4J)
            INDArray matriz = Nd4j.create(new double[]{1, 2, 3, 4, 5});
            System.out.println("✅ ND4J funciona correctamente");
            System.out.println("Matriz de prueba: " + matriz);

            // Probar una operación matemática simple
            INDArray resultado = matriz.mul(2); // Multiplicar por 2
            System.out.println("Resultado de multiplicar por 2: " + resultado);

            System.out.println("\n🎉 ¡DeepLearning4j está listo para usar!");

        } catch (Exception e) {
            System.err.println("❌ ERROR: Algo salió mal");
            e.printStackTrace();
        }
    }
}
