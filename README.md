# Eficiencia del Sector Público e Inversión: Un Análisis de DEA y Aprendizaje Automático

## Autores
- Zoila Llempén
- Oscar M. Valencia
- Jorge Guerra

## Descripción
Este repositorio contiene un análisis exhaustivo de la eficiencia del sector público utilizando el Análisis Envolvente de Datos (DEA) y técnicas de aprendizaje automático para evaluar la eficiencia de las inversiones del sector público en varios países de América Latina y el Caribe. El estudio se basa en el marco de DEA desarrollado por Afonso et al. (2007; 2013), con el objetivo de explorar la relación entre los inputs y outputs de las inversiones públicas y sus impactos en la eficiencia sectorial y agregada.

## Estructura del Documento
- **Introducción al DEA en la Eficiencia del Sector Público:** Introducción al concepto y aplicación del DEA en la evaluación del rendimiento del sector público.
- **Análisis de Eficiencia Sectorial vs. Agregada:** Examinación de la relación entre las eficiencias sectoriales individuales y la eficiencia global, enfatizando las interdependencias no lineales entre los sectores.

## Metodología
### Análisis Envolvente de Datos (DEA)
Aplicamos DEA para modelar la eficiencia de cada país, comparando el output real contra el output potencial. El modelo de DEA se especifica de la siguiente manera:
- Para el país \(i\), dado $y_i$ como el vector de output y $x_i$ como el vector de input, la relación se expresa como $y_i = f(x_i)$. Resolvemos el siguiente problema de programación lineal para encontrar la puntuación de eficiencia ($\theta$) y el vector de constantes ($\lambda$):

$$
\text{min} \ \theta,\lambda \ \text{sujeto a}
$$

$$
-y_i + Y\lambda \geq 0
$$

$$
\theta x_i - X\lambda \geq 0
$$

$$
\ell\lambda \geq 1
$$

$$
\lambda \geq 0
$$

### Aprendizaje Automático para Análisis No Lineal
Para capturar las complejas interdependencias sectoriales, utilizamos modelos basados en árboles de decisión (CATBoost, Random Forest y XGBoost). Estos modelos se seleccionan por su capacidad para manejar la no linealidad y prevenir el sobreajuste, con la elección del modelo basada en el Error Absoluto Medio (MAE).

#### Importancia de Características en Árboles de Decisión
Explica cómo la eficiencia de cada sector influye en la eficiencia general, calculada evaluando la contribución de cada característica a las predicciones del modelo. ## Importancia por Variable
La "Feature Importance" mide la contribución relativa de cada variable predictora a la eficiencia del modelo. Para un árbol de decisión, la importancia de una característica específica $\(f\)$ se calcula sumando la reducción ponderada de impureza que $\(f\)$ produce en todos los nodos donde se utiliza para dividir la muestra:

$$
\text{Importancia}(f) = \sum_{d \in D_f} n_d \times \Delta\text{Impureza}(d)
$$

Donde:
- $\(D_f\)$ es el conjunto de todos los nodos que utilizan la característica $\(f\)$ para dividir.
- $\(n_d\)$ es el número de muestras en el nodo $\(d\)$.
- $\(\Delta\text{Impureza}(d)\)$ es la mejora en la impureza lograda al usar la característica $\(f\)$ para dividir el nodo $\(d\)$.

La medida de error que utilizamos para evaluar el desempeño del modelo es el MAE, definido como:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} | \hat{y_i} - y_i |
$$

Donde $\hat{y_i}$ es la predicción del modelo y $y_i$ es el valor real.

## Datos y Fuentes
### Inputs
Los datos sobre la inversión pública provienen de la Base de Datos de Inversión Pública en América Latina (BBDD-GIPAL) de Aramendiz et al. (2019). Este conjunto de datos proporciona clasificaciones detalladas de los gastos de inversión pública en varios niveles de gobierno y sectores para países de América Latina y el Caribe desde 1999 hasta 2021.

### Outputs
Las fuentes de datos varían según la variable, asegurando un análisis comprensivo de los impactos de la inversión. Las fuentes incluyen:
- **WB:** Banco Mundial
- **IEA:** Agencia Internacional de Energía
- **WHO:** Organización Mundial de la Salud
- **UNICEF:** Fondo Internacional de Emergencia para la Infancia de las Naciones Unidas
- **WGI:** Indicadores de Gobernanza Mundial
- **WEO:** Perspectivas de la Economía Mundial (Fondo Monetario Internacional)
- **ILO:** Organización Internacional del Trabajo
- **FAO:** Organización de Alimentos y Agricultura
- **OECD:** Organización para la Cooperación y el Desarrollo Económicos

## Licencia
Este proyecto está licenciado bajo la Licencia MIT.

### Contacto
Para consultas o colaboraciones, por favor contactar a [Jorge Guerra](mailto:ja.guerrae@uniandes.edu.co).

