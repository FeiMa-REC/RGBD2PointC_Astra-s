import cv2
import threading
import time
import numpy as np
import os


# 定义Frame类，用于存储帧和时间戳
class Frame:
    def __init__(self, timestamp, frame):
        self.timestamp = timestamp
        self.frame = frame


def capture_frames(
    stream, frame_list, max_frames, mtx, data_ready, is_finish, mode=None
):
    while not is_finish[0]:
        # 抓取并解码新帧
        if stream.grab():
            timestamp = time.time()
            ret, frame = stream.retrieve(mode)
            if not ret:
                print(f"ERROR: Failed to decode frame from stream")
                break

            with mtx:
                if len(frame_list) >= max_frames:
                    frame_list.pop(0)
                frame_list.append(Frame(timestamp, frame))
            data_ready.notify()


def main():
    depth_stream = cv2.VideoCapture(1620)
    color_stream = cv2.VideoCapture(2, cv2.CAP_V4L2)

    if not color_stream.isOpened():
        print("ERROR: Unable to open color stream")
        return

    if not depth_stream.isOpened():
        print("ERROR: Unable to open depth stream")
        return

    color_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    color_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    depth_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    depth_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    depth_stream.set(cv2.CAP_PROP_OPENNI2_MIRROR, 0)

    print(
        f"Color stream: {int(color_stream.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(color_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(color_stream.get(cv2.CAP_PROP_FPS))} fps"
    )
    print(
        f"Depth stream: {int(depth_stream.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(depth_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(depth_stream.get(cv2.CAP_PROP_FPS))} fps"
    )

    depth_frames = []
    color_frames = []
    max_frames = 64

    mtx = threading.Lock()
    data_ready = threading.Condition(mtx)
    is_finish = [False]

    depth_thread = threading.Thread(
        target=capture_frames,
        args=(
            depth_stream,
            depth_frames,
            max_frames,
            mtx,
            data_ready,
            is_finish,
            cv2.CAP_OPENNI_DEPTH_MAP,
        ),
    )
    color_thread = threading.Thread(
        target=capture_frames,
        args=(color_stream, color_frames, max_frames, mtx, data_ready, is_finish),
    )

    depth_thread.start()
    color_thread.start()

    os.makedirs("./depth", exist_ok=True)
    os.makedirs("./rgb", exist_ok=True)

    with open("Association.txt", "w") as fout:
        while not is_finish[0]:
            with mtx:
                while not is_finish[0] and (
                    len(depth_frames) == 0 or len(color_frames) == 0
                ):
                    data_ready.wait()
                while len(depth_frames) > 0 and len(color_frames) > 0:
                    depth_frame = depth_frames.pop(0)
                    depth_t = depth_frame.timestamp

                    color_frame = color_frames.pop(0)
                    color_t = color_frame.timestamp

                    max_tdiff = 1 / (2 * color_stream.get(cv2.CAP_PROP_FPS))
                    if depth_t + max_tdiff < color_t:
                        continue
                    elif color_t + max_tdiff < depth_t:
                        continue

                    d8 = cv2.convertScaleAbs(depth_frame.frame, alpha=255.0 / 2000)
                    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_OCEAN)
                    cv2.imshow("Depth (colored)", d_color)
                    cv2.imshow("Color", color_frame.frame)

                    depth_img_name = f"./depth/{depth_frame.timestamp}.png"
                    color_img_name = f"./rgb/{color_frame.timestamp}.png"
                    cv2.imwrite(depth_img_name, depth_frame.frame)
                    cv2.imwrite(color_img_name, color_frame.frame)

                    fout.write(
                        f"{color_frame.timestamp} rgb/{color_frame.timestamp}.png {depth_frame.timestamp} depth/{depth_frame.timestamp}.png\n"
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        is_finish[0] = True
                        fout.close()
                        break

        data_ready.notify()
        depth_thread.join()
        color_thread.join()


if __name__ == "__main__":
    main()
