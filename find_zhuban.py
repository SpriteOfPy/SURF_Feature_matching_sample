# 516030910172 叶健龙
import cv2
import numpy as np
from matplotlib import pyplot as plt

# SURF SIFT在图片里检测特征

fame = cv2.imread('new_m.png',cv2.IMREAD_GRAYSCALE)
fw = fame[:205, :330]
#fw =cv2.equalizeHist(fw)

num = "6"        # 在这里改录入图片
sam = cv2.imread(num+'.png', cv2.IMREAD_GRAYSCALE)
sam0 = cv2.imread(num+'.png')
sp= cv2.resize(sam, None, fx=0.75, fy=0.75)
sp0= cv2.resize(sam0, None, fx=0.75, fy=0.75)
sp = sp[300:620, 350:1080]

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(600)
key_f, desc_f = surf.detectAndCompute(fw, None)
key_s, desc_s = surf.detectAndCompute(sp, None)   # 关键点与信息符

# kdtree建立索引方式的常量参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # checks指定索引树要被遍历的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 进行匹配搜索
matches = flann.knnMatch(desc_f, desc_s, k=2)

# 寻找距离近的放入good列表
good = []
for m, n in matches:
    if m.distance < 0.83 * n.distance:
        good.append(m)
print(len(good))

# 单应性
# 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
src_pts = np.float32([key_f[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([key_s[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# findHomography 函数是计算变换矩阵
# 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
# 返回值：M 为变换矩阵，mask是掩模
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)#5
# ravel方法将数据降维处理，最后并转换成列表格式
matchesMask = mask.ravel().tolist()
# 获取图像尺寸
h,w = fw.shape
# pts是图像img1的四个顶点
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[315,0]]).reshape(-1,1,2)
# 计算变换后的四个顶点坐标位置
dst = cv2.perspectiveTransform(pts,M)

# 验证四个点组成的四边形面积
dd = dst.reshape(8)
area = -0.5*(dd[0]*dd[3]+dd[2]*dd[5]+dd[4]*dd[1]-dd[0]*dd[5]-dd[2]*dd[1]-dd[4]*dd[3]+
            dd[0]*dd[5]+dd[4]*dd[7]+dd[6]*dd[1]-dd[0]*dd[7]-dd[4]*dd[1]-dd[6]*dd[5])
# 验证顶点关系偏差
tr = dd[0]-dd[2]+dd[4]-dd[6]
print(area, tr)
# 验证是否取得正确
flag = 0
point4 = [315, 0]
while ((area > 69000) | (area < 55000)|(abs(tr) > 55)):
    # surf,sift都不行的时候用上下一张模板
    if flag == 1:
        fame = cv2.imread('new_m2.png', cv2.IMREAD_GRAYSCALE)
        fw = fame[:205, :330]
        point4 = [305, 15]

    key_f, desc_f = sift.detectAndCompute(fw, None)
    key_s, desc_s = sift.detectAndCompute(sp, None)  # 换sift

    matches = flann.knnMatch(desc_f, desc_s, k=2)

    # 寻找距离近的放入good列表
    good = []
    for m, n in matches:
        if m.distance < 0.83 * n.distance:
            good.append(m)
    print(len(good))

    # 单应性
    src_pts = np.float32([key_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    matchesMask = mask.ravel().tolist()

    h, w = fw.shape
    # pts是图像img1的四个顶点
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], point4]).reshape(-1, 1, 2)
    # 计算变换后的四个顶点坐标位置
    dst = cv2.perspectiveTransform(pts, M)

    # 验证四个点组成的四边形面积
    dd = dst.reshape(8)
    area = -0.5 * (dd[0] * dd[3] + dd[2] * dd[5] + dd[4] * dd[1] - dd[0] * dd[5] - dd[2] * dd[1] - dd[4] * dd[3] +
                   dd[0] * dd[5] + dd[4] * dd[7] + dd[6] * dd[1] - dd[0] * dd[7] - dd[4] * dd[1] - dd[6] * dd[5])
    tr = dd[0] - dd[2] + dd[4] - dd[6]
    print(flag, "******", area, tr)
    flag += 1


# 根据四个顶点坐标位置在小图像画出变换后的边框
sp = cv2.polylines(sp,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)


draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matchesMask,
    flags=2
 )
img3 = cv2.drawMatches(fw, key_f, sp, key_s, good, None, **draw_params)
cv2.imshow("", img3)

dst0 = dst+[350,300]

# 根据四个顶点坐标位置在img2图像画出变换后的边框
sp0 = cv2.polylines(sp0,[np.int32(dst0)],True,(255,250,0),3, cv2.LINE_AA)
cv2.imshow("DISCOVER", sp0)
cv2.waitKey()




#############################################################

# 以下是从摄像头中获取新模板的过程
'''
fame = cv2.imread('0.jpg',cv2.IMREAD_GRAYSCALE)
fw = cv2.resize(fame, None, fx=0.1, fy=0.1)
#fw =cv2.equalizeHist(fw)

sam = cv2.imread('8.png', cv2.IMREAD_GRAYSCALE)
sp= cv2.resize(sam, None, fx=0.75, fy=0.75)
sp = sp[300:620,350:1080]


sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(600)
key_f, desc_f = surf.detectAndCompute(fw, None)
key_s, desc_s = surf.detectAndCompute(sp, None)   # 关键点与信息符

# kdtree建立索引方式的常量参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # checks指定索引树要被遍历的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 进行匹配搜索
matches = flann.knnMatch(desc_f, desc_s, k=2)


# 寻找距离近的放入good列表
good = []
for m, n in matches:
    if m.distance < 0.85 * n.distance:
        good.append(m)
print(len(good))

# 单应性
# 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
src_pts = np.float32([key_f[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([key_s[m.trainIdx].pt for m in good]).reshape(-1,1,2)
# findHomography 函数是计算变换矩阵
# 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
# 返回值：M 为变换矩阵，mask是掩模
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5)#5
# ravel方法将数据降维处理，最后并转换成列表格式
matchesMask = mask.ravel().tolist()
# 获取图像尺寸
h,w = fw.shape
# pts是图像img1的四个顶点
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
# 计算变换后的四个顶点坐标位置
dst = cv2.perspectiveTransform(pts,M)
dd =dst.reshape(8)
area = -0.5*(dd[0]*dd[3]+dd[2]*dd[5]+dd[4]*dd[1]-dd[0]*dd[5]-dd[2]*dd[1]-dd[4]*dd[3]+
            dd[0]*dd[5]+dd[4]*dd[7]+dd[6]*dd[1]-dd[0]*dd[7]-dd[4]*dd[1]-dd[6]*dd[5])
print(area)


dst_size = (500,300)

Hinv = np.linalg.inv(M)
new_m = cv2.warpPerspective(sp, Hinv, dst_size)
cv2.imwrite("new_m2.png", new_m)
'''

