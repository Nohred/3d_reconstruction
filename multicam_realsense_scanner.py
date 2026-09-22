"""
Multi-camera RealSense 3D object scanner (base script)
--------------------------------------------------------
Captura point clouds simultáneos de 2-3 cámaras Intel RealSense, calibra sus
posiciones relativas usando un tablero de ajedrez visible por todas, y fusiona
las nubes de puntos en un sistema de coordenadas común usando ICP (Open3D).

Requisitos:
    pip install pyrealsense2 opencv-python open3d numpy

Setup detectado en este equipo (ver multicam_preview_check.py):
    - 2x D435 en USB 2.1 (serials 317622073568, 242622071150)
    - 1x D415 en USB 3.2 (serial 145522066005)
    Por eso este script usa resolución/FPS reducidos automáticamente para
    las cámaras en USB 2.x, igual que el script de preview.

Flujo:
    1. Detecta todas las cámaras RealSense conectadas.
    2. Fija inter_cam_sync_mode ANTES de iniciar streaming (no se puede
       cambiar una vez que el pipeline ya está corriendo -> este era el bug).
    3. Captura un frame de color + depth de cada cámara con el tablero visible.
    4. Detecta el tablero en cada imagen (cv2.findChessboardCorners) y calcula
       la pose de cada cámara respecto al tablero (solvePnP).
    5. Calcula la transformación relativa entre cámaras a partir de esas poses.
    6. Genera point clouds individuales, los transforma al marco común,
       y aplica ICP para refinar la alineación.
    7. Exporta el point cloud fusionado a un archivo .ply.

Parámetros del tablero (default, igual que box_dimensioner_multicam de Intel):
    - Esquinas internas: 6 x 9  (equivale a un tablero de 7 x 10 cuadros)
    - Tamaño de cuadro: 2.40 cm (0.0240 m)
    Si imprimes un tablero de otro tamaño, actualiza CHESSBOARD_SIZE y SQUARE_SIZE_M.
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
import time
import os

# ---------------------- CONFIGURACIÓN ----------------------
CHESSBOARD_SIZE = (6, 9)      # (esquinas internas ancho, alto)
SQUARE_SIZE_M = 0.0240        # metros por cuadro (2.40 cm) -- AJUSTA si tu impresión difiere
DEPTH_TRUNC = 1             # metros, recorta el fondo más allá de esta distancia

FPS_USB3 = 30
FPS_USB2 = 15
RES_USB3 = (640, 480)
RES_USB2 = (424, 240)        # resolución reducida para no saturar USB 2.1
CHESSBOARD_LOG_DIR = os.path.join("logs", "chessboard")


def get_connected_devices():
    """Devuelve lista de dicts con serial, nombre y si está en USB3 o no."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No se detectó ninguna cámara RealSense conectada.")

    info = []
    for d in devices:
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        is_usb3 = usb.startswith("3")
        info.append({"serial": serial, "name": name, "usb": usb, "is_usb3": is_usb3})
        print(f"Detectada: {name} | Serial: {serial} | USB: {usb} "
              f"{'(USB3)' if is_usb3 else '(USB2, resolución reducida)'}")
    return info


def start_pipeline(device, master=False):
    """Inicia un pipeline para una cámara.
    IMPORTANTE: inter_cam_sync_mode se fija ANTES de pipeline.start(),
    usando el objeto device obtenido directamente del context. Esta opción
    no se puede cambiar una vez que el streaming ya está corriendo."""
    serial = device["serial"]
    resolution = RES_USB3 if device["is_usb3"] else RES_USB2
    fps = FPS_USB3 if device["is_usb3"] else FPS_USB2

    ctx = rs.context()
    rs_device = None
    for d in ctx.query_devices():
        if d.get_info(rs.camera_info.serial_number) == serial:
            rs_device = d
            break

    depth_sensor = rs_device.first_depth_sensor()
    if depth_sensor.supports(rs.option.inter_cam_sync_mode):
        try:
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1 if master else 2)
        except RuntimeError as e:
            print(f"Aviso: no se pudo fijar inter_cam_sync_mode en {serial}: {e}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, *resolution, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *resolution, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        raise RuntimeError(
            f"No se pudo iniciar la cámara {serial} ({device['name']}, {device['usb']}). "
            f"Error original: {e}"
        )

    align = rs.align(rs.stream.color)
    return pipeline, profile, align


# def capture_aligned_frames(pipeline, align, warmup=30):
#     """Descarta frames de calentamiento y captura un par depth+color alineado."""
#     for _ in range(warmup):
#         pipeline.wait_for_frames()
#     frames = pipeline.wait_for_frames()
#     aligned = align.process(frames)
#     depth_frame = aligned.get_depth_frame()
#     color_frame = aligned.get_color_frame()
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#     return depth_image, color_image, depth_frame.profile.as_video_stream_profile().get_intrinsics()

def capture_aligned_frames(pipeline, align, warmup=30, timeout_ms=10000, retries=3):
    """Descarta frames de calentamiento y captura un par depth+color alineado.
    Usa timeout largo y reintentos porque con 3 cámaras en el mismo bus/hub
    puede haber contención momentánea de ancho de banda al iniciar streaming."""
    last_error = None
    for attempt in range(retries):
        try:
            for _ in range(warmup):
                pipeline.wait_for_frames(timeout_ms=timeout_ms)
            frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return depth_image, color_image, depth_frame.profile.as_video_stream_profile().get_intrinsics()
        except RuntimeError as e:
            last_error = e
            print(f"  Reintento {attempt + 1}/{retries} tras timeout: {e}")
            time.sleep(0.5)
    raise RuntimeError(f"No se pudieron obtener frames tras {retries} intentos: {last_error}")


def intrinsics_to_o3d(intr):
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )


def get_chessboard_pose(color_image, intr):
    """Detecta el tablero y devuelve (rvec, tvec) respecto a la cámara."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)
    if not found:
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not ok:
        return None, None
    return rvec, tvec


def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def depth_to_pointcloud(depth_image, color_image, intr_o3d):
    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)
    return pcd


def refine_with_icp(source, target, initial_transform, threshold=0.02):
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result.transformation


def main():
    devices = get_connected_devices()
    if len(devices) < 2:
        print("Advertencia: se recomienda usar 2 o 3 cámaras. Detectada(s):", len(devices))

    pipelines = []
    aligns = []
    for i, device in enumerate(devices):
        pipeline, profile, align = start_pipeline(device, master=(i == 0))
        pipelines.append(pipeline)
        aligns.append(align)

    print("\nColoca el tablero de ajedrez visible por TODAS las cámaras y presiona Enter para calibrar...")
    input()

    poses_cam_to_board = []

    # for i, (pipeline, align) in enumerate(zip(pipelines, aligns)):
    #     depth_image, color_image, intr = capture_aligned_frames(pipeline, align)
    #     rvec, tvec = get_chessboard_pose(color_image, intr)
    #     if rvec is None:
    #         raise RuntimeError(f"No se detectó el tablero en la cámara {i} (serial {devices[i]['serial']}). "
    #                             f"Ajusta el ángulo o la iluminación.")
    #     T_cam_to_board = pose_to_matrix(rvec, tvec)
    #     poses_cam_to_board.append(T_cam_to_board)

    #     intr_o3d = intrinsics_to_o3d(intr)
    #     pcd = depth_to_pointcloud(depth_image, color_image, intr_o3d)
    #     pointclouds.append(pcd)
    #     print(f"Cámara {i} (serial {devices[i]['serial']}): tablero detectado y pose calculada.")

    for i, (pipeline, align) in enumerate(zip(pipelines, aligns)):
        time.sleep(0.3)  # deja que el buffer de la cámara se estabilice antes de leer
        depth_image, color_image, intr = capture_aligned_frames(pipeline, align)

        os.makedirs(CHESSBOARD_LOG_DIR, exist_ok=True)
        image_path = os.path.join(
            CHESSBOARD_LOG_DIR,
            f"camera_{i}_serial_{devices[i]['serial']}.png"
        )
        if not cv2.imwrite(image_path, color_image):
            print(f"Aviso: no se pudo guardar la captura de calibración en {image_path}")
        else:
            print(f"Captura de calibración guardada en {image_path}")

        rvec, tvec = get_chessboard_pose(color_image, intr)
        if rvec is None:
            raise RuntimeError(f"No se detectó el tablero en la cámara {i} (serial {devices[i]['serial']}). "
                                f"Ajusta el ángulo o la iluminación.")
        T_cam_to_board = pose_to_matrix(rvec, tvec)
        poses_cam_to_board.append(T_cam_to_board)

        print(f"Cámara {i} (serial {devices[i]['serial']}): tablero detectado y pose calculada.")

    print("\nCalibración completada. Ahora retira el tablero (o colócalo a un lado, visible) y "
          "coloca el objeto a escanear en el área de captura.")
    print("Presiona Enter cuando el objeto esté listo para capturar la nube de puntos definitiva...")
    input()

    pointclouds = []
    for i, (pipeline, align) in enumerate(zip(pipelines, aligns)):
        time.sleep(0.3)
        depth_image, color_image, intr = capture_aligned_frames(pipeline, align)
        intr_o3d = intrinsics_to_o3d(intr)
        pcd = depth_to_pointcloud(depth_image, color_image, intr_o3d)
        pointclouds.append(pcd)
        print(f"Cámara {i} (serial {devices[i]['serial']}): point cloud del objeto capturado.")

    # Transformación de cada cámara al marco de la cámara 0 (referencia común)
    # T_ref_to_board = poses_cam_to_board[0]
    # combined = o3d.geometry.PointCloud()
    # combined += pointclouds[0]

    # for i in range(1, len(pointclouds)):
        # T_cam_to_board = poses_cam_to_board[i]
        # T_cam_to_ref = np.linalg.inv(T_ref_to_board) @ T_cam_to_board
        # pointclouds[i].transform(T_cam_to_ref)

        # pointclouds[i].estimate_normals()
        # combined.estimate_normals()
        # refined_T = refine_with_icp(pointclouds[i], combined, np.eye(4))
        # pointclouds[i].transform(refined_T)

        # combined += pointclouds[i]
        # print(f"Cámara {i} fusionada con ICP.")
###

# T_board_to_cam[i] es la pose que da solvePnP: transforma puntos del
# sistema del tablero al sistema de la cámara i (X_cam = R @ X_board + t)
    T_board_to_cam = poses_cam_to_board  # renombra conceptualmente; ya lo tienes calculado así

    combined = o3d.geometry.PointCloud()
    combined += pointclouds[0]

    # Cámara 0 es la referencia: cam_i -> board -> cam_0
    T_board_to_cam0 = T_board_to_cam[0]

    for i in range(1, len(pointclouds)):
        T_board_to_cam_i = T_board_to_cam[i]
        T_cam_i_to_board = np.linalg.inv(T_board_to_cam_i)
        T_cam_i_to_cam0 = T_board_to_cam0 @ T_cam_i_to_board

        pointclouds[i].transform(T_cam_i_to_cam0)

        pointclouds[i].estimate_normals()
        combined.estimate_normals()
        refined_T = refine_with_icp(pointclouds[i], combined, np.eye(4))
        pointclouds[i].transform(refined_T)

        combined += pointclouds[i]
        print(f"Cámara {i} fusionada con ICP.")


    
###
    combined = combined.voxel_down_sample(voxel_size=0.003)
    o3d.io.write_point_cloud("scan_fusionado.ply", combined)
    print("Point cloud fusionado guardado en scan_fusionado.ply")

    for pipeline in pipelines:
        pipeline.stop()


if __name__ == "__main__":
    main()
