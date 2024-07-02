from openni import openni2
import numpy as np
import cv2

if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    color_stream = dev.create_color_stream()
    color_stream.start()

    cv2.namedWindow("depth")
    cv2.namedWindow("color")

    base_path = "./img/"
    count = 0

    while True:
        dframe = depth_stream.read_frame()
        dframe_data = np.array(dframe.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt.astype(np.uint16)
        dim_gray = cv2.convertScaleAbs(dpt, alpha=0.127)
        depth_colormap = cv2.applyColorMap(dim_gray, 2)
        cv2.imshow("depth", depth_colormap)

        cframe = color_stream.read_frame()
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
        cv2.imshow("color", cframe_data)

        key = cv2.waitKey(30)
        if int(key) == ord("q"):
            break

        if int(key) == ord(" "):  # Save image when spacebar is pressed
            count += 1
            depth_filename = base_path + f"depth_{count:08d}.jpg"
            color_filename = base_path + f"color_{count:08d}.jpg"
            cv2.imwrite(depth_filename, dpt)
            cv2.imwrite(color_filename, cframe_data)
            print(f"Images saved: {depth_filename}, {color_filename}")

    depth_stream.stop()
    color_stream.stop()
    dev.close()
