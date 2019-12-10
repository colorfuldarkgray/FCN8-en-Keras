# FCN8-en-Keras
Redes totalmente convolucionales para segmentación semántica.

Un archivo es para entrenar y hace una evaluación con los datos de entrenamiento.

El otro (con sufijo tt) es para evaluar con datos de prueba.

Se requieren 2 carpetas para imágenes y etiquetas de entrenamiento, también dos para imágenes y etiquetas de prueba

Las etiquetas con tipo de archivo png pero tres canales repitiendo un mismo mapa de segmentación. Las clases deben corresponder a numero enteros consecutivos.  

Se ejecutan entrenamientos definidos en una lista de configuraciones. Las variables son los conjuntos de datos a entrenar (agregar el número de clases que corresponda), el tipo de codificador (vgg16 o 19), si hace transferencia de aprendizaje (requiere pesos preentrenados), se puede jugar también con el tamaño de la entrada, etc.

El crédito es para Yumi y para los desarrolladores de Keras y de Python.

https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html

