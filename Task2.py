import cv2
import numpy as np

from matplotlib import pyplot as plt

# Camera matrix from chessboard calibration
K = np.float32([[603.94694108, 0, 314.65318324], [0, 604.25733395, 257.92261967], [0, 0, 1]])
K_inv = np.linalg.inv(K)
K2 = np.float32([[605.17745219, 0, 315.54512136], [0, 606.1345342, 255.84516007], [0, 0, 1]])
K2_inv = np.linalg.inv(K2)
# distortion coefficients
d = np.array([-0.44063917, 0.18131714, 0.00086375, 0.00049087, 0.00000, 0.0, 0.0, 0.0]).reshape(1,
                                                                                                8)
# second distortion coefficients
d2 = np.array([-4.44077670e-01, 1.88024465e-01, 2.24788583e-03, -1.95033612e-05, 0.00000, 0.0, 0.0, 0.0]).reshape(1,
                                                                                                                  8)


def degeneracyCheckPass(first_points, second_points, rot, trans):
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True


def drawlines(img1, img2, lines, pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 5, color, -1)
        cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Read the images

img1 = cv2.imread("C:\\Users\\79260\\Desktop\\AprilCalib_orgframe_00007.jpg", 0)  # Query image
img2 = cv2.imread("C:\\Users\\79260\\Desktop\\AprilCalib_orgframe_00008.jpg", 0)  # Train image
img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
sift = cv2.xfeatures2d.SIFT_create()

# undistort the images first
img1d = cv2.undistort(img1, K, d)
img2d = cv2.undistort(img2, K2, d2)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
# Compute fundamental matrix using RANSAC with 0.1 threshold and 0.99 confidence
pts2 = np.float32(pts2)
pts1 = np.float32(pts1)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)

# Selecting only the inliers
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# drawing lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# drawing lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

pt1 = np.array([[pts1[0][0]], [pts1[0][1]], [1]])
pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])

print(pt1)
print(pt2)
print("The fundamental matrix is")
print(F)


# E = K2.T.dot(F).dot(K)
E = K.T.dot(F).dot(K)
# E = K.T.dot(F).dot(K)
# E= K2.T*F*K

print("The essential matrix is")
print(E)

# use singular value decomposition to find the R and T
U, S, Vt = np.linalg.svd(E)
W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
first_inliers = []
second_inliers = []
for i in range(len(pts1)):
    # normalize and homogenize the image coordinates
    first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
    second_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))

# First choice: R = U * W * Vt, T = u_3
R = U.dot(W).dot(Vt)
T = U[:, 2]

# Determine the correct choice of second camera matrix
# only in one of the four configurations will all the points be in front of both cameras
# First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
    # Second choice: R = U * W * Vt, T = -u_3
    T = - U[:, 2]
    if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
        if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

print("Translation matrix is")
print(T)
print("Rotation matrix is")
print(R)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()
