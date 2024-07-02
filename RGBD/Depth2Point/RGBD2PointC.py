import cv2
import numpy as np
import open3d as o3d
import yaml


def get_parameter(parameter_file):
    with open(parameter_file, "r", encoding="utf-8") as file:
        content = file.read()
        # print("File content:")
        # print(repr(content))  # 使用repr以显示隐藏字符
        params = yaml.safe_load(content)

    # 直接读取单独的参数
    cx = params["cx"]
    cy = params["cy"]
    fx = params["fx"]
    fy = params["fy"]

    # 解析K矩阵
    K_data = params["K"]["data"]
    K = np.array(K_data).reshape(3, 3)

    # 解析D矩阵
    D_data = params["D"]["data"]
    D = np.array(D_data).reshape(1, 5)

    if K is None or D is None or fy == 0 or fx == 0 or cx == 0 or cy == 0:
        raise ValueError("ERROR: Calibration parameters to rectify stereo are missing!")

    return K, D, cx, cy, fx, fy


# 图像去畸变函数
def img_undistort(origin, K, D):
    h, w = origin.shape[:2]
    new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(origin, K, D, None, new_cam_matrix)
    return dst


# 点云绘制函数
def show_point_cloud(pointcloud):
    if len(pointcloud) == 0:
        print("Point cloud is empty!")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)

    # 显示点云
    o3d.visualization.draw_geometries([pcd])


def main(rgb_path, depth_path, parameter_file):
    # 读取相机参数
    K, D, cx, cy, fx, fy = get_parameter(parameter_file)

    # 读取图像并且读取所有通道
    rgb_src = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    depth_src = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 畸变校正
    # rectify_rgb = img_undistort(rgb_src, K, D)
    # rectify_depth = img_undistort(depth_src, K, D)

    # 直接使用原图反而效果更好，暂时不知道为什么畸变矫正步骤会导致结果变差
    rectify_rgb = rgb_src
    rectify_depth = depth_src

    # 显示深度图
    d8 = cv2.convertScaleAbs(depth_src, alpha=255.0 / 2000)
    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    cv2.imshow("Depth (colored)", d_color)
    cv2.imshow("RGB", rectify_rgb)

    # 生成点云
    pointcloud = []
    for v in range(rectify_rgb.shape[0]):
        for u in range(rectify_rgb.shape[1]):
            d = rectify_depth[v, u]
            if d == 0:
                continue
            # 缩放到以m为单位的点云
            z = float(d) / 1000
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            r, g, b = rgb_src[v, u][::-1]
            pointcloud.append([x, y, z, r, g, b])

    pointcloud = np.array(pointcloud)
    print(f"有效点云数量为：{len(pointcloud)}")

    cv2.waitKey(0)
    show_point_cloud(pointcloud)


if __name__ == "__main__":
    rgb_path = "./rgb/Color_3.png"
    depth_path = "./depth/Depth_3.png"
    parameter_file = "../CameraCalibra/CameraParams.yaml"

    main(rgb_path, depth_path, parameter_file)
