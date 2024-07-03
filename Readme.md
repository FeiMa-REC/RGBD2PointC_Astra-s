## 项目介绍
本项目是基于Astra s单目结构光相机的标定和深度图采集的项目。

### 文件夹说明

- OpenNI2：存放OpenNI2的安装包和相关的库文件
- RAW：原始的采集代码，可作为调试简单使用
  - save_depth_txt: 保存深度数据为txt格式
  - view_depth_gray: 读取txt格式深度数据并以灰度显示
  - view_rgb_depth_ir: 读取rgb、深度图像或者IR图像并显示（注意IR图不能与深度图同时显示）
- RGBD：存放相机标定、数据采集以及点云转换脚本
  - CameraCalibra: 存放相机标定脚本
  - Capture_DepthRGB_images: 存放数据采集脚本（提供采集模式选择）
  - Depth2Point: 提供点云转换脚本
  - calibxxx.pdf: 6x9标定图

### 项目演示

输入RGB和深度图像(未进行染色)

<center class="half">
    <img src="./imgs/Color.png" width="400">
    <img src="./imgs/Depth.png" width="400">
</center>

输出点云

<center class="half">
    <img src="./imgs/test.gif">
</center>

### 标定注意事项

相机标定是进行视觉测量和定位的基础工作之一，标定参数准确与否直接关系到整个系统的精度，相机标定过程中标定图片数据的采集过程中需要注意以下问题:

标定板拍摄的张数要能覆盖整个测量空间及整个测量视场，把相机图像分成四个象限，应保证拍摄的标定板图像均匀分布在四个象限中，且在每个象限中的各个位置进行俯仰,翻滚,偏航摆放。

1、标定图片的数量通常在30张之间，图像数量太少，容易导致标定参数不准确。
2、圆或者圆环特征的像素数尽量大于20，标定板的成像尺寸应大致占整幅画面的1/4。
3、用辅助光源对标定板进行打光，保证标定板的亮度足够且均匀.(很重要)。
4、标定板成像不能过爆，过爆会导致特征轮廓的提取的偏移，从而导致圆心提取不准确。
5、标定板特征成像不能出现明显的离焦距，出现离焦时可通过调整调整标定板的距离、光圈的大小和像距（对于定焦镜头，通常说的调焦就是指调整像距。
6、标定过程，相机的光圈、焦距不能发生改变，改变需要重新标定。

> 目前标定的是640x480分辨率深度对应的相机内参，对于分辨率为1280x1024对应的内参计算公式为：
> fx_1280 = fx_640 * 2
> fy_1280 = fy_640 * 2
> cx_1280 = cx_640 * 2
> cy_1280 = cy_640 * 2 + 32

### 采集深度图注意事项

深度图像是一个单通道16bit二维无符整形数组，保存的是三维空间点的深度Z值，深度单位通常是毫米。
而Astras摄像头通过调用OPENNI2接口，可以获取到深度图像，但是在保存为深度图时需要特别注意转换为np.uint16格式，否则会出现深度数据只有0-255的情况。

这篇文章有对深度数据采集注意事项的详细介绍：https://developer.orbbec.com.cn/v/blog_detail?id=777

### TODO

- [ ] 找出矫正图畸变的原因
