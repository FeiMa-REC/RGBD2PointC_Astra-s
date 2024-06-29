import cv2
import time
import numpy as np
import os
from openni import openni2


# 定义Frame类，用于存储帧和时间戳
class Frame:
    def __init__(self, timestamp, frame):
        self.timestamp = timestamp
        self.frame = frame


def main():
    # 初始化OpenNI
    openni2.initialize()

    # 参数为设备ID
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    if not color_stream:
        print("ERROR: Unable to open color stream")
        return

    if not depth_stream:
        print("ERROR: Unable to open depth stream")
        return

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

    depth_frames = []
    color_frames = []
    max_frames = 64
    i = 1

    os.makedirs("./depth", exist_ok=True)
    os.makedirs("./rgb", exist_ok=True)

    with open("Association.txt", "w") as fout:
        while True:
            # 读取深度帧
            depth_frame = depth_stream.read_frame()
            depth_timestamp = time.time()
            depth_data = np.array(
                depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(depth_data[:, :, 0], dtype="float32")
            dpt2 = np.asarray(depth_data[:, :, 1], dtype="float32")
            dpt2 *= 255
            depth_frame_data = dpt1 + dpt2
            depth_frame_obj = Frame(depth_timestamp, depth_frame_data)

            # 读取彩色帧
            color_frame = color_stream.read_frame()
            color_timestamp = time.time()
            color_data = np.array(
                color_frame.get_buffer_as_triplet()).reshape([480, 640, 3])
            R = color_data[:, :, 0]
            G = color_data[:, :, 1]
            B = color_data[:, :, 2]
            color_frame_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
            color_frame_obj = Frame(color_timestamp, color_frame_data)

            # 显示深度帧和彩色帧
            # 最大深度为2000mm时对应255 （255/2000=1.275）
            # 0-255 : 黑-白
            d8 = cv2.convertScaleAbs(depth_frame_obj.frame, alpha=255.0 / 2000)
            d_color = cv2.applyColorMap(d8, cv2.COLORMAP_OCEAN)
            # d_color = cv2.applyColorMap(d8, 2)
            cv2.imshow("Depth (colored)", d_color)
            cv2.imshow("Color", color_frame_obj.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # 按下ESC或q键退出
                break
            elif key == ord(" "):  # 按下空格键保存图像
                depth_img_name = f"./depth/Depth_{i}.png"
                color_img_name = f"./rgb/Color_{i}.png"
                success_depth = cv2.imwrite(
                    depth_img_name, depth_frame_obj.frame)
                success_color = cv2.imwrite(
                    color_img_name, color_frame_obj.frame)
                if success_depth and success_color:
                    print(f"捕捉到第{i}组图像")
                    i += 1
                else:
                    print(
                        f"保存图像失败：Depth成功={success_depth}, Color成功={success_color}")

    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
