from openni import openni2
import numpy as np
import cv2
import os

if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    cv2.namedWindow('depth')

    base_path = './data'
    count = 1

    while True:
        dframe = depth_stream.read_frame()
        dframe_data = np.array(
            dframe.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt.astype(np.uint16)

        dim_gray = cv2.convertScaleAbs(dpt, alpha=0.127)
        depth_colormap = cv2.applyColorMap(dim_gray, 2)
        cv2.imshow('depth', depth_colormap)

        # Save depth data
        depth_filename = os.path.join(
            base_path, 'depth_frame_' + str(count).zfill(8) + '.txt')
        np.savetxt(depth_filename, dpt, fmt='%d')

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        count += 1

    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
    openni2.unload()
