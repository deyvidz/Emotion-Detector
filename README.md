# 🎭 Emotion Detector

[![Java](https://img.shields.io/badge/Java-17+-orange.svg)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.8+-red.svg)](https://maven.apache.org/)
[![DeepLearning4J](https://img.shields.io/badge/DeepLearning4J-1.0.0--M2.1-blue.svg)](https://deeplearning4j.konduit.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Clasificador de emociones en texto español utilizando redes neuronales LSTM con DeepLearning4J**

## 📋 Descripción

**Emotion Detector** es un sistema de Machine Learning que analiza texto en español y clasifica las emociones expresadas utilizando redes neuronales recurrentes LSTM (Long Short-Term Memory). El proyecto implementa un pipeline completo de procesamiento de lenguaje natural, desde la vectorización de palabras con Word2Vec hasta la clasificación final con una arquitectura LSTM optimizada.

### 🎯 Emociones Detectadas

- **😊 Joy** (Alegría)
- **😢 Sadness** (Tristeza) 
- **😠 Anger** (Enojo)
- **😨 Fear** (Miedo)
- **😲 Surprise** (Sorpresa)

## 🚀 Características Principales

- **🧠 Red Neuronal LSTM**: Arquitectura de 2 capas LSTM con 128 y 64 neuronas
- **📚 Word2Vec**: Vectorización de palabras con 50 dimensiones
- **💬 Modo Interactivo**: Interfaz de línea de comandos para predicciones en tiempo real
- **📊 Métricas de Confianza**: Muestra probabilidades para cada emoción
- **🎯 Alta Precisión**: Entrenado con 200 ejemplos y validado con 40 casos de prueba
- **⚡ Optimizado**: Procesamiento eficiente con batch size configurable

## 📁 Estructura del Proyecto

```
EmotionDetector/
├── 📄 README.md                          # Este archivo
├── 📄 pom.xml                            # Configuración Maven
├── 📄 MANUAL_TECNICO.md                  # Documentación técnica
├── 📄 MANUAL_USUARIO.md                  # Manual de usuario
├── 📄 .gitignore                         # Archivos ignorados por Git
├── 📁 src/
│   ├── 📁 main/
│   │   ├── 📁 java/com/escuela/
│   │   │   ├── 🎯 EmotionDetectorMain.java      # Clase principal (interfaz)
│   │   │   ├── 🧠 EmotionClassifierModel.java   # Arquitectura LSTM
│   │   │   ├── 🏋️ ModelTrainer.java             # Lógica de entrenamiento
│   │   │   ├── 🔮 EmotionPredictor.java         # Motor de predicciones
│   │   │   ├── 📊 Word2VecVectorizer.java       # Vectorización Word2Vec
│   │   │   ├── 🧹 TextPreprocessor.java         # Preprocesamiento de texto
│   │   │   ├── 📂 DataLoader.java               # Carga de datasets
│   │   │   └── 🚀 TrainEmotionClassifier.java   # Script de entrenamiento
│   │   └── 📁 resources/
│   │       └── 📁 data/emotions/
│   │           ├── 📊 emotions_train.csv        # Datos de entrenamiento (200 ejemplos)
│   │           └── 📊 emotions_test.csv         # Datos de prueba (40 ejemplos)
│   └── 📁 test/                                 # Pruebas unitarias
├── 📁 target/                                # JAR compilado y dependencias
├── 📄 emotion_classifier_model.zip          # Modelo LSTM entrenado
└── 📄 word2vec_emotions.zip                  # Modelo Word2Vec entrenado
```

## 🛠️ Requisitos Previos

### Hardware Mínimo
- **RAM**: 4GB (8GB recomendado para entrenamiento)
- **CPU**: Procesador de 2 núcleos o superior
- **Almacenamiento**: 500MB de espacio libre

### Software Requerido
- **Java Development Kit (JDK) 17** o superior
- **Apache Maven 3.8+** para gestión de dependencias
- **Git** para control de versiones

### Librerías Principales
- **DeepLearning4J 1.0.0-M2.1** - Framework principal de deep learning
- **ND4J 1.0.0-M2.1** - Backend de cálculo numérico (CPU)
- **DeepLearning4J NLP** - Procesamiento de lenguaje natural
- **SLF4J** - Sistema de logging
- **Apache Commons IO** - Utilidades de I/O
- **JUnit 4.13.2** - Pruebas unitarias


### 3. Entrenar el Modelo (Primera Vez)

# Entrenar el modelo LSTM y Word2Vec
mvn exec:java -Dexec.mainClass="com.escuela.TrainEmotionClassifier"


> **⚠️ Nota**: El entrenamiento puede tardar varios minutos dependiendo del hardware. Se generarán dos archivos: `emotion_classifier_model.zip` y `word2vec_emotions.zip`.

## 🎮 Uso del Programa

### Modo Interactivo (Recomendado)

# Ejecutar el programa principal
mvn exec:java -Dexec.mainClass="com.escuela.EmotionDetectorMain"


### Ejemplos de Uso

#### 🎉 Alegría
```
💬 Escribe una frase: Estoy muy feliz con mi nuevo trabajo

🎯 EMOCIÓN DETECTADA: JOY
📊 Confianza: 92.3%
📈 Distribución:
   😊 Joy:       92.3%
   😢 Sadness:    2.1%
   😠 Anger:      1.8%
   😨 Fear:       1.5%
   😲 Surprise:   2.3%
```

#### 😢 Tristeza
```
💬 Escribe una frase: Me siento solo y abandonado

🎯 EMOCIÓN DETECTADA: SADNESS
📊 Confianza: 87.6%
📈 Distribución:
   😊 Joy:       3.2%
   😢 Sadness:   87.6%
   😠 Anger:      2.8%
   😨 Fear:       4.1%
   😲 Surprise:   2.3%
```

#### 😠 Enojo
```
💬 Escribe una frase: Tu actitud me molesta mucho

🎯 EMOCIÓN DETECTADA: ANGER
📊 Confianza: 79.4%
📈 Distribución:
   😊 Joy:       5.1%
   😢 Sadness:   8.2%
   😠 Anger:     79.4%
   😨 Fear:       4.3%
   😲 Surprise:   3.0%
```

### Comandos Especiales
| Comando |                        Descripción                       |
|---------|----------------------------------------------------------|
| `ejemplos` | Muestra 10 ejemplos predefinidos con sus predicciones |
| `salir` o `exit` | Termina el programa                             |

## 🧠 Arquitectura del Modelo

### Pipeline de Procesamiento
```
TEXTO ENTRADA → PREPROCESAMIENTO → WORD2VEC → LSTM → SOFTMAX → EMOCIÓN
     ↓               ↓              ↓        ↓        ↓         ↓
"Estoy feliz" → Tokenización → Vectores → 128 → 64 → 5 clases → JOY
```

### Especificaciones Técnicas
- **Word2Vec**: 50 dimensiones, ventana de 5 palabras, frecuencia mínima: 5
- **LSTM Layer 1**: 128 neuronas, activación TANH
- **LSTM Layer 2**: 64 neuronas, activación TANH  
- **Output Layer**: 5 neuronas con activación SOFTMAX
- **Optimizador**: Adam con learning rate 0.01
- **Inicialización**: Xavier/Glorot
- **Entrenamiento**: 150 epochs, batch size 20

## 📊 Dataset

### Datos de Entrenamiento
- **Archivo**: `emotions_train.csv`
- **Ejemplos**: 200 frases 
- **Formato**: `texto,emocion`
- **Distribución**: 40 frases por emocion

### Datos de Prueba
- **Archivo**: `emotions_test.csv`
- **Ejemplos**: 40 frases
- **Propósito**: Validación del modelo entrenado

### Ejemplo del Dataset
```csv
texto,emocion
Estoy muy feliz con mis resultados,joy
Hoy es un día maravilloso,joy
Me siento triste y solo,sadness
Tu actitud me molesta,anger
Tengo miedo de fracasar,fear
No puedo creer lo que pasó,surprise
```

## 🔧 Uso Programático (API)

```java
// Crear predictor
EmotionPredictor predictor = new EmotionPredictor();

// Cargar modelos entrenados
boolean exito = predictor.cargarModelo(
    "emotion_classifier_model.zip",
    "word2vec_emotions.zip"
);

if (exito) {
    // Hacer predicción
    String texto = "Hoy es un día maravilloso";
    EmotionPredictor.ResultadoPrediccion resultado = predictor.predecir(texto);
    
    // Obtener resultados
    String emocion = resultado.getEmocionPredicha();
    double confianza = resultado.getConfianza();
    double[] probabilidades = resultado.getProbabilidades();
    
    System.out.println("Emoción: " + emocion + " (" + (confianza*100) + "%)");
}
```

## 🧪 Pruebas Unitarias

El proyecto incluye pruebas unitarias completas:


# Ejecutar todas las pruebas
mvn test

# Ejecutar prueba específica
mvn test -Dtest=TestModel


### Pruebas Disponibles
- **TestModel**: Validación de la arquitectura LSTM
- **TestVectorizer**: Verificación de vectorización Word2Vec
- **TestPreprocessor**: Pruebas de preprocesamiento de texto
- **TestSetup**: Configuración inicial y carga de datos
- **TestDataLoader**: Validación de carga de datasets


## 📚 Referencias y Recursos

### Documentación Oficial
- **DeepLearning4J**: https://deeplearning4j.konduit.ai/
- **ND4J**: https://nd4j.org/
- **Word2Vec**: https://code.google.com/archive/p/word2vec/

### Papers Académicos
1. **Long Short-Term Memory** - Hochreiter & Schmidhuber (1997)
2. **Efficient Estimation of Word Representations** - Mikolov et al. (2013)
3. **Attention Is All You Need** - Vaswani et al. (2017)

## 👥 Contribución

¡Las contribuciones son bienvenidas!

## 👤 Autor

**David Luchetta**  
*Tecnico en computacion*

---
