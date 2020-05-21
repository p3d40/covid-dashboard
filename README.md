# covid-dashboard
Este proyecto busca creat una interfaz para desplegar información acerca del brote de COVID-19 aplicado a Guatemala.

## Estadistícas

Con base a los datos reportados de Johns Hopkins para la COVID-19, se calculan y estiman:
- Tiempo de duplicación
- Letalidad
- Número de reproducción

Estos parámetros se calculan con sus intervalos de confianza.

## Proyección

Las proyecciones se realizan utilizando un modelo compartimentado generado por los parámetros obtenidos con base
a los datos. Este se utiliza para proyectar:
- Casos
- Muertes

## Uso
EL archivo `ml.py` genera las gráficas y las páginas de reportes.
