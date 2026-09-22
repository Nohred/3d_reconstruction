"""
Reconstrucción de malla 3D a partir de un point cloud limpio
--------------------------------------------------------------
Toma un .ply de puntos (idealmente ya procesado con clean_pointcloud.py)
y genera una malla 3D con superficie (triángulos), no solo puntos sueltos.
Exporta a .ply y opcionalmente a .obj/.stl para usar en otros programas
(Blender, MeshLab, slicers de impresión 3D, etc.)

Requisitos:
    pip install open3d numpy

Uso:
    1. Ajusta los valores de la sección CONFIGURACIÓN más abajo.
    2. Ejecuta: python reconstruct_mesh.py

Notas sobre los métodos:
    - Poisson: genera superficies suaves y cerradas (rellena huecos).
      Bueno si tu objeto es tipo "blob" sólido (botella, perfume, caja).
      Requiere normales bien orientadas; el script las calcula automáticamente.
    - Ball Pivoting (BPA): sigue más fielmente los puntos originales, no
      rellena huecos agresivamente. Mejor si tu nube tiene huecos reales
      (por ejemplo, zonas que ninguna cámara alcanzó a ver) y no quieres
      que el algoritmo los "invente".
    - Alpha Shape: alternativa rápida y simple, útil para pruebas iniciales.
"""

import numpy as np
import open3d as o3d


# ---------------------- CONFIGURACIÓN ----------------------
# Se conserva el archivo de entrada que estaba definido como valor por defecto.
INPUT_PATH = "scans/scans_proc/scan_cinta.ply"
OUTPUT_PATH = "scans/mesh/scan_cinta_mesh.ply"      

METHOD = "ballpivot"         # "poisson", "ballpivot" o "alpha"

NORMAL_RADIUS = 0.02
NORMAL_MAX_NN = 30

POISSON_DEPTH = 9
DENSITY_THRESHOLD = 0.02    # 0: no recortar por densidad

BPA_RADII = [0.005, 0.01, 0.02]
ALPHA = 0.03  # 

TRANSFER_COLOR = True
EXPORT_OBJ = False
EXPORT_STL = False
SHOW_VIEW = False


def load_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"El archivo {path} no contiene puntos válidos.")
    return pcd


def estimate_normals(pcd, radius, max_nn):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)
    return pcd


def reconstruct_poisson(pcd, depth, density_threshold):
    print(f"Reconstruyendo con Poisson (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)

    if density_threshold > 0:
        # Elimina triángulos de baja confianza (zonas donde Poisson "inventó"
        # superficie sin suficiente soporte real de puntos, típico en bordes)
        threshold_value = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < threshold_value
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"  Vértices de baja densidad removidos (percentil {density_threshold})")

    return mesh


def reconstruct_ball_pivoting(pcd, radii):
    print(f"Reconstruyendo con Ball Pivoting (radios={radii})...")
    radii_vector = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii_vector)
    return mesh


def reconstruct_alpha_shape(pcd, alpha):
    print(f"Reconstruyendo con Alpha Shape (alpha={alpha})...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def transfer_colors_from_pointcloud(mesh, pcd, max_distance=0.01):
    """Si el point cloud tenía color RGB, lo transfiere a los vértices de la
    malla buscando el punto más cercano de la nube original para cada vértice."""
    if not pcd.has_colors():
        print("  El point cloud original no tiene color RGB, la malla quedará sin color.")
        return mesh

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros_like(mesh_vertices)

    for i, vertex in enumerate(mesh_vertices):
        _, idx, dist = pcd_tree.search_knn_vector_3d(vertex, 1)
        if len(idx) > 0 and np.sqrt(dist[0]) <= max_distance:
            mesh_colors[i] = pcd_colors[idx[0]]
        else:
            mesh_colors[i] = [0.6, 0.6, 0.6]  # gris para vértices sin color cercano

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    return mesh


def main():
    output_path = OUTPUT_PATH or INPUT_PATH.rsplit(".", 1)[0] + "_mesh.ply"

    print(f"Cargando: {INPUT_PATH}")
    pcd = load_pointcloud(INPUT_PATH)
    print(f"Puntos de entrada: {len(pcd.points)}")

    pcd_for_normals = o3d.geometry.PointCloud(pcd)
    estimate_normals(pcd_for_normals, NORMAL_RADIUS, NORMAL_MAX_NN)

    if METHOD == "poisson":
        mesh = reconstruct_poisson(pcd_for_normals, POISSON_DEPTH, DENSITY_THRESHOLD)
    elif METHOD == "ballpivot":
        mesh = reconstruct_ball_pivoting(pcd_for_normals, BPA_RADII)
    else:
        mesh = reconstruct_alpha_shape(pcd_for_normals, ALPHA)

    print(f"Malla generada: {len(mesh.vertices)} vértices, {len(mesh.triangles)} triángulos")

    mesh = clean_mesh(mesh)
    print(f"Malla limpiada: {len(mesh.vertices)} vértices, {len(mesh.triangles)} triángulos")

    if TRANSFER_COLOR:
        mesh = transfer_colors_from_pointcloud(mesh, pcd)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"\nGuardado: {output_path}")

    if EXPORT_OBJ:
        obj_path = output_path.rsplit(".", 1)[0] + ".obj"
        o3d.io.write_triangle_mesh(obj_path, mesh)
        print(f"Guardado: {obj_path}")

    if EXPORT_STL:
        stl_path = output_path.rsplit(".", 1)[0] + ".stl"
        o3d.io.write_triangle_mesh(stl_path, mesh)
        print(f"Guardado: {stl_path} (sin color, formato STL no soporta color)")

    if SHOW_VIEW:
        o3d.visualization.draw_geometries([mesh], window_name=f"Malla - {output_path}",
                                           width=1280, height=800, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
