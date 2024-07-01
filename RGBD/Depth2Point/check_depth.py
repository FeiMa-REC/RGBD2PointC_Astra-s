import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(rgb_path, depth_path, parameter_file):
    # 读取相机参数
    # K, D, cx, cy, fx, fy = get_parameter(parameter_file)

    # 读取图像并且读取所有通道
    rgb_src = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    depth_src = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 畸变校正
    # rectify_rgb = img_undistort(rgb_src, K, D)
    # rectify_depth = img_undistort(depth_src, K, D)

    rectify_rgb = rgb_src
    rectify_depth = depth_src

    # 显示深度图
    d8 = cv2.convertScaleAbs(depth_src, alpha=255.0 / 2000)
    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_OCEAN)
    cv2.imshow("Depth (colored)", d_color)
    cv2.imshow("RGB", rectify_rgb)

    # 打印深度图部分值进行检查
    print("深度图部分值：", rectify_depth[:10, :10])

    # 检查深度图统计信息
    print("深度图最大值：", np.max(rectify_depth))
    print("深度图最小值：", np.min(rectify_depth))
    print("深度图平均值：", np.mean(rectify_depth))

    # 绘制深度图直方图
    plt.figure()
    plt.hist(rectify_depth.ravel(), bins=256, range=(0, 2000), fc="k", ec="k")
    plt.title("Depth Image Histogram")
    plt.xlabel("Depth Value（mm）")
    plt.ylabel("Frequency")
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rgb_path = "./rgb/Color_1.png"
    depth_path = "./depth/Depth_1.png"
    parameter_file = "../CameraCalibra/CameraParams.yaml"

    main(rgb_path, depth_path, parameter_file)
