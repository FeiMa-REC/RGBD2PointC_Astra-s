import cv2
import numpy as np
import glob


def output_camera_param(camera_matrix, dist_coeffs):
    """保存相机参数"""
    fs = cv2.FileStorage("mydepth_2.yaml", cv2.FILE_STORAGE_WRITE)
    if fs.isOpened():
        # 公共参数
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fs.write("CameraType", "PinHole")
        fs.write("fx", fx)
        fs.write("fy", fy)
        fs.write("cx", cx)
        fs.write("cy", cy)
        fs.write("K", camera_matrix)
        fs.write("D", dist_coeffs)
        fs.release()
        print("参数已经写入至：mydepth.yaml！")
    else:
        print("Error: 无法保存！")


def main():
    # 标定图像目录
    images = glob.glob("./data/color_*.jpg")

    # 棋盘格模板规格 ：8 一行黑白块数量-1 ；5 一列黑白块数量-1
    w = 8
    h = 5

    # 找棋盘格角点 迭代次数+阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.imshow("findCorners", img)
            cv2.waitKey(300)
    cv2.destroyAllWindows()

    # 标定：计算相机内外参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 保存相机参数
    output_camera_param(mtx, dist)

    # 去畸变
    img2 = cv2.imread("./data/color_00000009.jpg")
    h, w = img2.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 0, (w, h)
    )  # 自由比例参数
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    # 根据前面ROI区域裁剪图片
    # x, y, w, h = roi
    # dst = dst[y : y + h, x : x + w]
    cv2.imwrite("calibresult.png", dst)

    # 反投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print("total error: ", total_error / len(objpoints))


if __name__ == "__main__":
    main()
