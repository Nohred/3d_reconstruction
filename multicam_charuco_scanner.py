"""
Multi-camera RealSense 3D scanner con ChArUco (reemplaza chessboard puro)
----------------------------------------------------------------------------
Igual flujo que multicam_realsense_scanner.py, pero usa un tablero ChArUco
en vez de chessboard puro para eliminar la ambigüedad de pose de solvePnP
que causaba la desalineación entre las 3 cámaras (paredes "colapsadas").

Requisitos:
    pip install pyrealsense2 opencv-python opencv-contrib-python open3d numpy

IMPORTANTE: necesitas opencv-contrib-python (no solo opencv-python) para
tener el módulo cv2.aruco disponible. Si ya tienes opencv-python instalado
sin contrib, instala así:
    pip uninstall opencv-python
    pip install opencv-contrib-python

Paso 1 - Generar e imprimir el tablero:
    python multicam_charuco_scanner.py --generate-board
    Esto guarda charuco_board.png. Imprímelo a escala 100% (sin "ajustar a
    página") y mide con regla que un cuadro mida realmente SQUARE_SIZE_M.

Paso 2 - Correr la calibración + captura:
    python multicam_charuco_scanner.py

Parámetros del tablero (ajusta si cambias el tamaño de impresión):
    - Cuadros: 7 x 5
    - Tamaño de cuadro: 3.0 cm
    - Tamaño de marcador ArUco dentro de cada cuadro: 2.2 cm
    - Diccionario: DICT_5X5_100
"""

import argparse
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

# ---------------------- CONFIGURACIÓN DEL TABLERO ----------------------
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH_M = 0.036      # 3.6 cm por cuadro -- mide tu impresión real y ajusta
MARKER_LENGTH_M = 0.026      # 2.6 cm por marcador ArUco (debe ser menor al cuadro)
ARUCO_DICT = cv2.aruco.DICT_5X5_100

DEPTH_TRUNC = 1.5

FPS_USB3 = 30
FPS_USB2 = 15
RES_USB3 = (640, 480)
RES_USB2 = (424, 240)


def build_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary
    )
    return board, dictionary


def generate_board_image(path="charuco_board.png", dpi=300):
    board, _ = build_charuco_board()
    px_per_m = dpi / 0.0254
    width_px = int(CHARUCO_SQUARES_X * SQUARE_LENGTH_M * px_per_m)
    height_px = int(CHARUCO_SQUARES_Y * SQUARE_LENGTH_M * px_per_m)
    img = board.generateImage((width_px, height_px), marginSize=20, borderBits=1)
    cv2.imwrite(path, img)
    print(f"Tablero guardado en {path} ({width_px}x{height_px}px @ {dpi} dpi)")
    print(f"Imprime a escala 100%. Cada cuadro debe medir {SQUARE_LENGTH_M*100:.1f} cm real.")


def get_connected_devices():
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

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def capture_aligned_frames(pipeline, align, warmup=30, timeout_ms=10000, retries=3):
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
    return o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)


def get_charuco_pose(color_image, intr, board, dictionary):
    """Detecta el tablero ChArUco y devuelve la pose del TABLERO respecto a
    la CÁMARA (rvec, tvec), es decir T_board_to_cam, igual convención que
    solvePnP normal. Evita la ambigüedad de pose de chessboard puro porque
    cada marcador ArUco tiene identidad única."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, 0

    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 4:
        return None, None, 0

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 4:
        return None, None, 0

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not ok:
        return None, None, 0

    return rvec, tvec, len(charuco_corners)


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
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)


def refine_with_icp(source, target, initial_transform, threshold=0.02):
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result.transformation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-board", action="store_true",
                         help="Solo genera charuco_board.png para imprimir y termina.")
    parser.add_argument("--output", default="scan_fusionado_charuco.ply")
    args = parser.parse_args()

    if args.generate_board:
        generate_board_image()
        return

    board, dictionary = build_charuco_board()

    devices = get_connected_devices()
    if len(devices) < 2:
        print("Advertencia: se recomienda usar 2 o 3 cámaras. Detectada(s):", len(devices))

    pipelines = []
    aligns = []
    for i, device in enumerate(devices):
        pipeline, profile, align = start_pipeline(device, master=(i == 0))
        pipelines.append(pipeline)
        aligns.append(align)

    print("\nColoca el tablero ChArUco visible por TODAS las cámaras y presiona Enter para calibrar...")
    input()

    poses_board_to_cam = []
    for i, (pipeline, align) in enumerate(zip(pipelines, aligns)):
        time.sleep(0.3)
        depth_image, color_image, intr = capture_aligned_frames(pipeline, align)
        rvec, tvec, n_corners = get_charuco_pose(color_image, intr, board, dictionary)
        if rvec is None:
            raise RuntimeError(f"No se detectó suficiente tablero ChArUco en la cámara {i} "
                                f"(serial {devices[i]['serial']}). Ajusta ángulo/iluminación/distancia.")
        T_board_to_cam = pose_to_matrix(rvec, tvec)
        poses_board_to_cam.append(T_board_to_cam)
        print(f"Cámara {i} (serial {devices[i]['serial']}): {n_corners} esquinas ChArUco detectadas, pose calculada.")

    print("\nCalibración completada. Coloca el objeto a escanear en el área de captura.")
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

    # T_cam_i_to_cam0 = T_board_to_cam0 @ inv(T_board_to_cam_i)
    T_board_to_cam0 = poses_board_to_cam[0]
    combined = o3d.geometry.PointCloud()
    combined += pointclouds[0]

    for i in range(1, len(pointclouds)):
        T_board_to_cam_i = poses_board_to_cam[i]
        T_cam_i_to_board = np.linalg.inv(T_board_to_cam_i)
        T_cam_i_to_cam0 = T_board_to_cam0 @ T_cam_i_to_board

        pointclouds[i].transform(T_cam_i_to_cam0)

        pointclouds[i].estimate_normals()
        combined.estimate_normals()
        refined_T = refine_with_icp(pointclouds[i], combined, np.eye(4))
        pointclouds[i].transform(refined_T)

        combined += pointclouds[i]
        print(f"Cámara {i} fusionada con ICP.")

    combined = combined.voxel_down_sample(voxel_size=0.003)
    o3d.io.write_point_cloud(args.output, combined)
    print(f"Point cloud fusionado guardado en {args.output}")

    for pipeline in pipelines:
        pipeline.stop()


if __name__ == "__main__":
    main()
