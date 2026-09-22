"""
Visualizador de nubes de puntos (.ply) generadas por multicam_realsense_scanner.py
------------------------------------------------------------------------------------
Abre una ventana interactiva 3D con Open3D para inspeccionar el point cloud
fusionado: rotar, hacer zoom, medir densidad de puntos y ver si la
calibración/fusión entre las 3 cámaras quedó bien alineada.

Requisitos:
    pip install open3d numpy

Uso:
    1. Ajusta los valores de la sección CONFIGURACIÓN más abajo.
    2. Ejecuta: python view_pointcloud.py

Controles en la ventana Open3D:
    Click + arrastrar   -> rotar
    Ctrl + arrastrar     -> desplazar (pan)
    Scroll / rueda       -> zoom
    N                    -> mostrar/ocultar normales
    +/-                  -> aumentar/disminuir tamaño de punto
    R                    -> resetear vista
    Q / ESC              -> cerrar ventana
"""

import numpy as np
import open3d as o3d


# ---------------------- CONFIGURACIÓN ----------------------
# Se conserva el archivo de entrada que estaba definido como valor por defecto.
INPUT_PATH = "scans/scans_proc/scan_charger.ply"
# INPUT_PATH = "scans/mesh/scan_cartera_mesh.ply"
# INPUT_PATH = "scans/scans_raw/scan_1.ply" 
DOWNSAMPLE = 0.0          # Tamaño de voxel en metros; 0 = sin downsample
REMOVE_OUTLIERS = False
COLOR_NORMALS = False
SHOW_AXES = True


def load_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"El archivo {path} no contiene puntos válidos.")
    return pcd


def print_stats(pcd, label=""):
    points = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"\n--- Estadísticas {label} ---")
    print(f"Puntos totales: {len(points)}")
    print(f"Tiene color RGB: {has_colors}")
    print(f"Bounding box (x,y,z) en metros: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")
    print(f"Centro: {bbox.get_center()}")


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Limpia puntos aislados/ruido, típico en los bordes de fusión entre cámaras."""
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    removed = len(pcd.points) - len(clean.points)
    print(f"Outliers removidos: {removed} ({100 * removed / len(pcd.points):.1f}%)")
    return clean


def main():
    print(f"Cargando: {INPUT_PATH}")
    pcd = load_pointcloud(INPUT_PATH)
    print_stats(pcd, label="(original)")

    if DOWNSAMPLE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE)
        print_stats(pcd, label=f"(downsample {DOWNSAMPLE}m)")

    if REMOVE_OUTLIERS:
        pcd = remove_outliers(pcd)
        print_stats(pcd, label="(sin outliers)")

    geometries = [pcd]

    if SHOW_AXES:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(axes)

    if COLOR_NORMALS:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        normals = np.asarray(pcd.normals)
        colors_from_normals = (normals + 1.0) / 2.0  # normaliza a rango [0,1] para visualizar como color
        pcd.colors = o3d.utility.Vector3dVector(colors_from_normals)

    print("\nAbriendo visualizador... (Q o ESC para cerrar)")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Point Cloud Viewer - {INPUT_PATH}",
        width=1280,
        height=800,
        point_show_normal=False
    )


if __name__ == "__main__":
    main()
