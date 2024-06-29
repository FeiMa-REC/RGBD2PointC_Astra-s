import numpy as np
import cv2
import os

if __name__ == "__main__":
    base_path = './data'
    count = 1

    while True:
        # Load saved depth information
        depth_filename = os.path.join(
            base_path, 'depth_frame_' + str(count).zfill(8) + '.txt')
        if not os.path.exists(depth_filename):
            print(f"Depth file '{depth_filename}' not found.")
            break

        depth_data = np.loadtxt(depth_filename, dtype=np.uint16)

        # Display loaded depth information
        cv2.imshow('Saved Depth Information', depth_data.astype(
            np.float32) / 1000.0)  # Convert to meters for display

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        count += 1

    cv2.destroyAllWindows()
