# Escaneo 3D multicámara con Intel RealSense

Proyecto para capturar y procesar nubes de puntos usando dos o tres cámaras Intel RealSense (D415/D435). El flujo general es:

1. Comprobar la conexión y el ancho de banda USB.
2. Calibrar las cámaras y fusionar sus nubes de puntos.
3. Limpiar la nube fusionada.
4. Reconstruir una malla 3D y visualizar los resultados.

## Scripts

- `multicam_preview_check.py`: muestra en vivo color y profundidad de todas las cámaras conectadas. Permite comprobar estabilidad, FPS y resolución según USB; `q`/`ESC` cierra y `s` guarda una captura en `snapshots/`.
- `multicam_charuco_scanner.py`: opción recomendada para calibrar con un tablero ChArUco, capturar el objeto y fusionar las nubes mediante la pose del tablero e ICP. Genera `scan_fusionado_charuco.ply` por defecto.
- `multicam_realsense_scanner.py`: versión base que usa un tablero de ajedrez para calibración, captura las nubes y las fusiona con ICP. Genera `scan_fusionado.ply`.
- `clean_pointcloud.py`: aplica recortes por coordenadas/centroide, reducción por vóxel y filtros de outliers a un `.ply`. Guarda una copia procesada en `scans/scans_proc/` sin sobrescribir el original.
- `reconstruct_mesh.py`: estima normales y convierte una nube limpia en una malla mediante Poisson, Ball Pivoting o Alpha Shape. Puede exportar `.ply`, `.obj` y `.stl`.
- `scans/view_pointcloud.py`: abre un visualizador 3D interactivo de Open3D y muestra estadísticas, ejes, color y opciones de limpieza temporal.

## Instalación

```bash
pip install pyrealsense2 opencv-python opencv-contrib-python open3d numpy
```

`opencv-contrib-python` es necesario para el script ChArUco. Se recomienda conectar las cámaras directamente a puertos USB 3.x; las cámaras detectadas como USB 2.x usan automáticamente una resolución y FPS menores.

## Uso básico

Ejecutar desde la carpeta del proyecto:

```bash
python multicam_preview_check.py
python multicam_charuco_scanner.py --generate-board
python multicam_charuco_scanner.py
python clean_pointcloud.py
python reconstruct_mesh.py
python scans/view_pointcloud.py
```

Los scripts de procesamiento usan rutas y parámetros definidos en su sección `CONFIGURACIÓN`. Ajusta el archivo de entrada, el archivo de salida y las dimensiones reales del tablero antes de ejecutar el escaneo.
