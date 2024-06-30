import cv2
import time
import numpy as np
import os
from openni import openni2
from threading import Thread, Lock
from queue import Queue
import signal


# 定义Frame类，用于存储帧和时间戳
class Frame:
    def __init__(self, timestamp, frame):
        self.timestamp = timestamp
        self.frame = frame


# 全局变量，指示程序是否应该退出
isFinish = False


def initialize_streams():
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    if not color_stream:
        raise Exception("ERROR: Unable to open color stream")

    if not depth_stream:
        raise Exception("ERROR: Unable to open depth stream")

    color_stream.set_video_mode(
        openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=640,
            resolutionY=480,
            fps=30,
        )
    )
    depth_stream.set_video_mode(
        openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX=640,
            resolutionY=480,
            fps=30,
        )
    )

    depth_stream.start()
    color_stream.start()

    print(f"Color stream: {640}x{480} @ {30} fps")
    print(f"Depth stream: {640}x{480} @ {30} fps")

    return depth_stream, color_stream


def read_depth_frames(depth_stream, depth_queue, lock):
    while not isFinish:
        depth_frame = depth_stream.read_frame()
        depth_timestamp = depth_frame.timestamp / 1000.0  # 转换为秒
        depth_data = np.array(depth_frame.get_buffer_as_triplet()).reshape(
            [480, 640, 2]
        )
        dpt1 = np.asarray(depth_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(depth_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        depth_frame_data = dpt1 + dpt2
        depth_frame_data = depth_frame_data.astype(np.uint16)
        depth_frame_obj = Frame(depth_timestamp, depth_frame_data)
        with lock:
            depth_queue.put(depth_frame_obj)


def read_color_frames(color_stream, color_queue, lock):
    while not isFinish:
        color_frame = color_stream.read_frame()
        color_timestamp = color_frame.timestamp / 1000.0  # 转换为秒
        color_data = np.array(color_frame.get_buffer_as_triplet()).reshape(
            [480, 640, 3]
        )
        R = color_data[:, :, 0]
        G = color_data[:, :, 1]
        B = color_data[:, :, 2]
        color_frame_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
        color_frame_obj = Frame(color_timestamp, color_frame_data)
        with lock:
            color_queue.put(color_frame_obj)


def is_synchronized(depth_frame_obj, color_frame_obj, threshold=0.03):
    """
    TODO:使用了多线程的方法打印出来的时间戳差值还是很大，不知道如何避免。此处先弃用该方法。
    """
    # print(
    #     f"Depth timestamp: {depth_frame_obj.timestamp:.6f}, Color timestamp: {color_frame_obj.timestamp:.6f}, Difference: {abs(depth_frame_obj.timestamp - color_frame_obj.timestamp):.6f}"
    # )
    # return abs(depth_frame_obj.timestamp - color_frame_obj.timestamp) < threshold
    return True


def process_frames(depth_queue, color_queue, lock, mode="record"):
    global isFinish
    while not isFinish:
        with lock:
            if not depth_queue.empty() and not color_queue.empty():
                depth_frame_obj = depth_queue.get()
                color_frame_obj = color_queue.get()
                if is_synchronized(depth_frame_obj, color_frame_obj):
                    d8 = cv2.convertScaleAbs(depth_frame_obj.frame, alpha=255.0 / 2000)
                    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_OCEAN)
                    cv2.imshow("Depth (colored)", d_color)
                    cv2.imshow("Color", color_frame_obj.frame)

                    if mode == "record":
                        depth_writer.write(
                            cv2.convertScaleAbs(
                                depth_frame_obj.frame, alpha=255.0 / 2000
                            )
                        )
                        color_writer.write(color_frame_obj.frame)
                    elif mode == "save":
                        depth_img_name = f"./depth/Depth_{i}.png"
                        color_img_name = f"./rgb/Color_{i}.png"
                        success_depth = cv2.imwrite(
                            depth_img_name, depth_frame_obj.frame
                        )
                        success_color = cv2.imwrite(
                            color_img_name, color_frame_obj.frame
                        )
                        if success_depth and success_color:
                            fout.write(
                                f"{color_frame_obj.timestamp:.6f} {color_img_name} {depth_frame_obj.timestamp:.6f} {depth_img_name}\n"
                            )
                            i += 1
                        else:
                            print(
                                f"保存图像失败：Depth成功={success_depth}, Color成功={success_color}"
                            )

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):  # 按下ESC或q键退出
                        isFinish = True
                        break

    cv2.destroyAllWindows()


def record_videos(depth_stream, color_stream, depth_queue, color_queue, lock):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    global depth_writer, color_writer
    depth_writer = cv2.VideoWriter("Depth.avi", fourcc, 30.0, (640, 480), isColor=False)
    color_writer = cv2.VideoWriter("Color.avi", fourcc, 30.0, (640, 480), isColor=True)

    process_frames(depth_queue, color_queue, lock, mode="record")

    depth_writer.release()
    color_writer.release()


def save_image_sequences(depth_stream, color_stream, depth_queue, color_queue, lock):
    os.makedirs("./depth", exist_ok=True)
    os.makedirs("./rgb", exist_ok=True)
    global fout, i
    fout = open("Association.txt", "w")
    i = 1

    process_frames(depth_queue, color_queue, lock, mode="save")

    fout.close()


def signal_handler(sig, frame):
    global isFinish
    print("You pressed Ctrl+C!")
    isFinish = True


def main():
    global isFinish
    signal.signal(signal.SIGINT, signal_handler)
    try:
        depth_stream, color_stream = initialize_streams()
        depth_queue = Queue()
        color_queue = Queue()
        lock = Lock()

        depth_thread = Thread(
            target=read_depth_frames, args=(depth_stream, depth_queue, lock)
        )
        color_thread = Thread(
            target=read_color_frames, args=(color_stream, color_queue, lock)
        )

        depth_thread.start()
        color_thread.start()

        mode = input("选择模式：1-录制视频 2-保存图像序列\n")
        if mode == "1":
            record_videos(depth_stream, color_stream, depth_queue, color_queue, lock)
        elif mode == "2":
            save_image_sequences(
                depth_stream, color_stream, depth_queue, color_queue, lock
            )
        else:
            print("无效的选择")

        depth_thread.join()
        color_thread.join()

    except Exception as e:
        print(str(e))
    finally:
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()


if __name__ == "__main__":
    main()
