"""
Limpieza de nubes de puntos (.ply)
--------------------------------------------------------------
Aplica una cadena de filtros configurables sobre un point cloud y guarda
el resultado en un archivo NUEVO (nunca sobrescribe el original), para que
puedas comparar antes/después y ajustar parámetros por prueba y error.

Requisitos:
    pip install open3d numpy

Uso:
    1. Ajusta los valores de la sección CONFIGURACIÓN más abajo.
    2. Ejecuta: python clean_pointcloud.py

Todos los filtros son opcionales e independientes; se aplican en este orden:
    1. Recorte por distancia al centroide (elimina fondo/mesa lejana)
    2. Recorte por altura Z (por ejemplo, quitar el plano de la mesa)
    3. Downsample por voxel (reduce densidad/ruido uniforme)
    4. Filtro de outliers estadístico (elimina puntos dispersos aislados)
    5. Filtro de outliers por radio (elimina puntos sin suficientes vecinos cercanos)

Cada paso imprime cuántos puntos quedaron, para que veas el efecto de cada filtro.
"""

import numpy as np
import open3d as o3d


# ---------------------- CONFIGURACIÓN ----------------------
INPUT_PATH = "scans/scans_raw/scan_2.ply"
OUTPUT_PATH = "scans/scans_proc/scan_charger.ply"

# Usa None para desactivar cualquiera de estos recortes.
CENTROID_RADIUS = 0.25       # Metros desde el centroide, por ejemplo: 0.25
MIN_Z = None               # Altura Z mínima, por ejemplo: -0.05
MAX_Z = None                 # Altura Z máxima, por ejemplo: 1.0
MIN_X = -0.9999                 # Altura X mínima, por ejemplo: -0.05
MAX_X = 0.5                 # Altura X máxima, por ejemplo: 1.0
MIN_Y = None                 # Altura Y mínima, por ejemplo: -0.05
MAX_Y = 0.075                 # Altura Y máxima, por ejemplo: 1.0  # Tumba el piso
VOXEL_SIZE = None            # Metros, por ejemplo: 0.003

# Filtro estadístico de outliers.
USE_STATISTICAL_FILTER = True
STAT_NEIGHBORS = 20
STAT_STD = 2.0

# Filtro de outliers por radio. Usa None para desactivarlo.
RADIUS_NN = None             # Vecinos mínimos, por ejemplo: 15
RADIUS = 0.02                # Metros

SHOW_VIEW = False            # True para abrir el visualizador al final


def load_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"El archivo {path} no contiene puntos válidos.")
    return pcd


def report(pcd, step_name):
    n = len(pcd.points)
    print(f"[{step_name}] puntos restantes: {n}")
    return n


def crop_by_centroid_distance(pcd, radius):
    """Elimina todo punto más lejos de `radius` metros del centroide de la nube.
    Útil para cortar mesa/fondo/piso que quedó fuera del objeto principal."""
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mask = distances <= radius
    cropped = pcd.select_by_index(np.where(mask)[0])
    return cropped, centroid


def crop_by_height(pcd, min_z=None, max_z=None):
    """Recorta puntos por coordenada Z absoluta (útil para quitar el plano
    de la mesa/piso si conoces su altura aproximada en el sistema de la cámara 0)."""
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    if max_z is not None:
        mask &= points[:, 2] <= max_z
    return pcd.select_by_index(np.where(mask)[0])


def crop_by_width(pcd, min_x=None, max_x=None):
    """Recorta puntos por coordenada X absoluta."""
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    if min_x is not None:
        mask &= points[:, 0] >= min_x
    if max_x is not None:
        mask &= points[:, 0] <= max_x
    return pcd.select_by_index(np.where(mask)[0])


def crop_by_depth(pcd, min_y=None, max_y=None):
    """Recorta puntos por coordenada Y absoluta."""
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    if min_y is not None:
        mask &= points[:, 1] >= min_y
    if max_y is not None:
        mask &= points[:, 1] <= max_y
    return pcd.select_by_index(np.where(mask)[0])


def voxel_downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_statistical_outliers(pcd, nb_neighbors, std_ratio):
    """Elimina puntos cuya distancia promedio a sus vecinos se aleja
    demasiado (std_ratio desviaciones estándar) del promedio general.
    Bueno para ruido disperso típico de bordes de fusión entre cámaras."""
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean


def remove_radius_outliers(pcd, nb_points, radius):
    """Elimina puntos que no tienen al menos `nb_points` vecinos dentro de
    `radius` metros. Bueno para eliminar puntos totalmente aislados/flotantes."""
    clean, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return clean


def main():
    output_path = OUTPUT_PATH or INPUT_PATH.rsplit(".", 1)[0] + "_limpio.ply"

    print(f"Cargando: {INPUT_PATH}")
    pcd = load_pointcloud(INPUT_PATH)
    report(pcd, "original")

    if CENTROID_RADIUS is not None:
        pcd, centroid = crop_by_centroid_distance(pcd, CENTROID_RADIUS)
        print(f"  Centroide usado: {centroid}")
        report(pcd, f"recorte por centroide (radio {CENTROID_RADIUS}m)")

    if MIN_X is not None or MAX_X is not None:
        pcd = crop_by_width(pcd, MIN_X, MAX_X)
        report(pcd, f"recorte por ancho X (min={MIN_X}, max={MAX_X})")

    if MIN_Y is not None or MAX_Y is not None:
        pcd = crop_by_depth(pcd, MIN_Y, MAX_Y)
        report(pcd, f"recorte por profundidad Y (min={MIN_Y}, max={MAX_Y})")

    if MIN_Z is not None or MAX_Z is not None:
        pcd = crop_by_height(pcd, MIN_Z, MAX_Z)
        report(pcd, f"recorte por altura Z (min={MIN_Z}, max={MAX_Z})")

    if VOXEL_SIZE is not None:
        pcd = voxel_downsample(pcd, VOXEL_SIZE)
        report(pcd, f"downsample voxel ({VOXEL_SIZE}m)")

    if USE_STATISTICAL_FILTER:
        pcd = remove_statistical_outliers(pcd, STAT_NEIGHBORS, STAT_STD)
        report(pcd, f"filtro estadístico (vecinos={STAT_NEIGHBORS}, std={STAT_STD})")

    if RADIUS_NN is not None:
        pcd = remove_radius_outliers(pcd, RADIUS_NN, RADIUS)
        report(pcd, f"filtro por radio (min_vecinos={RADIUS_NN}, radio={RADIUS}m)")

    if len(pcd.points) == 0:
        print("\nADVERTENCIA: el resultado quedó vacío. Los filtros fueron demasiado agresivos, "
              "reduce STAT_STD, aumenta CENTROID_RADIUS, o quita RADIUS_NN.")
        return

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"\nGuardado: {output_path} ({len(pcd.points)} puntos)")

    if SHOW_VIEW:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axes], window_name=f"Limpio - {output_path}",
                                           width=1280, height=800)


if __name__ == "__main__":
    main()
