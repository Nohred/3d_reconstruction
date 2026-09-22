"""
Verificación y vista previa multi-cámara: 2x D435 + 1x D415
--------------------------------------------------------------
Objetivo: comprobar que las 3 cámaras RealSense funcionan simultáneamente
sin conflictos de USB/ancho de banda, y mostrar en una sola ventana las
vistas de color + profundidad (colorizada) de cada una, en vivo, para poder
acomodarlas físicamente alrededor del objeto antes de calibrar.

Requisitos:
    pip install pyrealsense2 opencv-python numpy

Controles en la ventana:
    q / ESC  -> salir
    s        -> guardar snapshot de las 3 vistas actuales (color+depth) en ./snapshots/

Notas importantes (D415 vs D435, y USB detectado en este equipo):
    - Detectado: 2x D435 en USB 2.1 y 1x D415 en USB 3.2. Esto es una
      limitación real de ancho de banda: USB 2.1 da ~480 Mbps compartidos,
      mucho menos que USB 3.x (~5 Gbps). Con depth+color en 640x480 @30fps
      cada D435 en USB 2.1 puede fallar al iniciar o dar FPS muy bajo.
      Por eso este script baja automáticamente resolución/FPS para
      dispositivos detectados en USB 2.x (ver DEVICE_USB2_PROFILE abajo).
    - D415 FOV horizontal ~64°, D435 FOV horizontal ~86°. No esperes que
      "vean" exactamente lo mismo a la misma distancia; el D435 abarca más.
    - No es necesario cable de sync físico para evitar interferencia:
      las cámaras D4xx normalmente NO interfieren entre sí incluso en modo
      default. Este script configura inter_cam_sync_mode ANTES de iniciar
      el streaming (no se puede cambiar una vez que el pipeline ya corre).
    - Si tienes puertos USB 3.0 libres en la laptop/PC, conecta ahí los D435
      en vez de un hub USB 2.0 — vas a notar mejora inmediata en FPS y
      estabilidad al usar las 3 cámaras a la vez.
"""

import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs

FPS_USB3 = 30
FPS_USB2 = 15
RES_USB3 = (640, 480)
RES_USB2 = (424, 240)   # resolución más baja para no saturar USB 2.1
SNAPSHOT_DIR = "snapshots"


def get_device_info():
    """Lista las cámaras conectadas con su modelo, serial y tipo de USB,
    para identificar cuál es D415 y cuáles son D435 antes de armar los pipelines."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No se detectó ninguna cámara RealSense. Revisa conexiones USB.")

    info = []
    for d in devices:
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        is_usb3 = usb.startswith("3")
        info.append({"name": name, "serial": serial, "usb": usb, "is_usb3": is_usb3})
        print(f"Detectada: {name} | Serial: {serial} | USB: {usb} "
              f"{'(USB3, OK)' if is_usb3 else '(USB2, ancho de banda limitado)'}")
    return info


def start_pipeline(device):
    """Configura e inicia el pipeline. inter_cam_sync_mode se fija ANTES de
    llamar a pipeline.start(), usando el sensor obtenido vía device directo
    (no se puede tocar una vez iniciado el streaming)."""
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
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, 0)
        except RuntimeError as e:
            print(f"Aviso: no se pudo fijar inter_cam_sync_mode en {serial} antes de streaming: {e}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, *resolution, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, *resolution, rs.format.bgr8, fps)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        raise RuntimeError(
            f"No se pudo iniciar la cámara {serial} ({device['name']}, {device['usb']}). "
            f"Posible conflicto de USB/ancho de banda. Si está en un hub USB 2.0 compartido, "
            f"prueba conectarla directo a un puerto USB 3.0 de la laptop. Error original: {e}"
        )

    align = rs.align(rs.stream.color)
    return pipeline, align, resolution


def get_color_and_depth(pipeline, align, colorizer):
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return color_image, depth_colored


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def resize_to(img, target_size):
    """Reescala para que todas las vistas tengan el mismo tamaño en la
    cuadrícula final, incluso si cada cámara corre a resolución distinta."""
    return cv2.resize(img, target_size)


def main():
    devices = get_device_info()
    if len(devices) < 3:
        print(f"Advertencia: se esperaban 3 cámaras, se detectaron {len(devices)}. "
              f"Continuando con las disponibles.")

    n_usb2 = sum(1 for d in devices if not d["is_usb3"])
    if n_usb2 > 0:
        print(f"\nATENCION: {n_usb2} cámara(s) en USB 2.1 detectada(s). "
              f"Se usará resolución reducida ({RES_USB2[0]}x{RES_USB2[1]} @ {FPS_USB2}fps) "
              f"para esas cámaras para evitar errores de ancho de banda.\n"
              f"Recomendación: mover esas cámaras a puertos USB 3.0 si están disponibles.\n")

    pipelines = []
    aligns = []
    colorizers = []
    labels = []
    serials = []
    resolutions = []

    display_size = (480, 360)  # tamaño uniforme de cada celda en la cuadrícula final

    for d in devices:
        try:
            pipeline, align, resolution = start_pipeline(d)
        except RuntimeError as e:
            print(f"ERROR al iniciar {d['serial']}: {e}")
            continue
        pipelines.append(pipeline)
        aligns.append(align)
        colorizers.append(rs.colorizer())
        labels.append(f'{d["name"]} ({d["serial"][-4:]}) {d["usb"]}')
        serials.append(d["serial"])
        resolutions.append(resolution)
        time.sleep(0.5)  # arranque escalonado, reduce pico de consumo USB simultáneo

    if not pipelines:
        raise RuntimeError("Ninguna cámara pudo iniciarse. Revisa conexiones USB.")

    print(f"\n{len(pipelines)}/{len(devices)} cámaras iniciadas correctamente.")
    print("Presiona 'q' para salir, 's' para guardar snapshot.\n")

    frame_counts = [0] * len(pipelines)
    fail_counts = [0] * len(pipelines)
    t_start = time.time()

    try:
        while True:
            row_color = []
            row_depth = []

            for i, (pipeline, align, colorizer) in enumerate(zip(pipelines, aligns, colorizers)):
                try:
                    color_image, depth_colored = get_color_and_depth(pipeline, align, colorizer)
                except RuntimeError:
                    color_image, depth_colored = None, None

                if color_image is None:
                    fail_counts[i] += 1
                    placeholder = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                    row_color.append(label(placeholder, f"{labels[i]} SIN FRAME"))
                    row_depth.append(label(placeholder.copy(), "SIN FRAME"))
                    continue

                frame_counts[i] += 1
                color_r = resize_to(color_image, display_size)
                depth_r = resize_to(depth_colored, display_size)
                row_color.append(label(color_r, f"COLOR - {labels[i]}"))
                row_depth.append(label(depth_r, f"DEPTH - {serials[i]}"))

            color_row = np.hstack(row_color)
            depth_row = np.hstack(row_depth)
            combined = np.vstack([color_row, depth_row])

            elapsed = time.time() - t_start
            fps_text = " | ".join(
                f'{labels[i].split(" ")[0]}: {frame_counts[i] / elapsed:.1f} fps ({fail_counts[i]} fallos)'
                for i in range(len(pipelines))
            )
            print(f"\r{fps_text}", end="", flush=True)

            cv2.imshow("Multi-camara RealSense - Color (arriba) / Depth (abajo)", combined)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                ts = int(time.time())
                cv2.imwrite(f"{SNAPSHOT_DIR}/combined_{ts}.png", combined)
                print(f"\nSnapshot guardado: {SNAPSHOT_DIR}/combined_{ts}.png")

    finally:
        print("\nCerrando pipelines...")
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

        print("\nResumen de compatibilidad:")
        for i in range(len(pipelines)):
            print(f"  {labels[i]}: {frame_counts[i]} frames capturados, {fail_counts[i]} fallos/timeouts")


if __name__ == "__main__":
    main()
    