import cv2
import numpy as np
import open3d as o3d
import yaml


def get_parameter(parameter_file):
    with open(parameter_file, "r") as file:
        params = yaml.safe_load(file)

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

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)

    # 添加点云到场景
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # 设置背景为白色
    opt.point_size = 2  # 设置点的大小

    # 设置相机参数
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    # 运行可视化循环
    vis.run()
    vis.destroy_window()


def main(rgb_path, depth_path, parameter_file):
    # 读取相机参数
    K, D, cx, cy, fx, fy = get_parameter(parameter_file)

    # 读取图像并且读取所有通道
    rgb_src = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    depth_src = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 畸变校正
    rectify_rgb = img_undistort(rgb_src, K, D)
    rectify_depth = img_undistort(depth_src, K, D)

    # 显示深度图
    d8 = cv2.convertScaleAbs(depth_src, alpha=255.0 / 2000)
    d_color = cv2.applyColorMap(d8, cv2.COLORMAP_OCEAN)
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
            r, g, b = rectify_rgb[v, u][::-1]
            pointcloud.append([x, y, z, r, g, b])

    pointcloud = np.array(pointcloud)
    print(f"有效点云数量为：{len(pointcloud)}")

    cv2.waitKey(0)
    show_point_cloud(pointcloud)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(
            "ERROR:格式错误！正确用法 python RGBD2PointC.py RGB图像路径 深度图像路径 参数文件路径"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
