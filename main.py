import numpy as np
import cv2
import matplotlib.pyplot as plt

R2 = []
C2 = []
pts3D = []
z1 = []
z2 = []
pairs = []

def epipolarLines(pair1, pair2, F, Image_1, Image_2, file_name, rectified=False):
    img1 = Image_1.copy()
    img2 = Image_2.copy()
    epilines1, epilines2 = [], []

    for i in range(pair1.shape[0]):
        x1 = np.array([pair1[i,0], pair1[i,1], 1]).reshape(3,1)
        x2 = np.array([pair2[i,0], pair2[i,1], 1]).reshape(3,1)
        l1 = np.dot(F.T, x2)
        epilines1.append(l1)
        l2 = np.dot(F, x1)
        epilines2.append(l2)

        if not rectified:
            miny1 = 0
            minx1 = -(l1[1]*miny1 + l1[2])/l1[0]
            maxy1 = Image_1.shape[0]
            maxx1 = -(l1[1]*maxy1 + l1[2])/l1[0]
            miny2 = 0        
            minx2 = -(l2[1]*miny2 + l2[2])/l2[0]
            maxy2 = Image_2.shape[0] 
            maxx2 = -(l2[1]*maxy2 + l2[2])/l2[0]
               
        else:
            minx1 = 0
            maxx1 = Image_1.shape[1] -1
            minx2 = 0
            maxx2 = Image_2.shape[1] - 1
            miny1 = -l1[2]/l1[1]
            maxy1 = -l1[2]/l1[1]
            miny2 = -l2[2]/l2[1]
            maxy2 = -l2[2]/l2[1]           

        cv2.circle(img1, (int(pair2[i,0]),int(pair2[i,1])), 10, (0,0,255), -1)
        cv2.line(img1, (int(minx1), int(miny1)), (int(maxx1), int(maxy1)), (255, 0, 100), 2)
        cv2.circle(img2, (int(pair2[i,0]),int(pair2[i,1])), 10, (0,0,255), -1)
        cv2.line(img2, (int(minx2), int(miny2)), (int(maxx2), int(maxy2)), (255, 0, 100), 2)

    final = np.concatenate((img1, img2), axis = 1)
    cv2.imshow("", final)
    cv2.imwrite(file_name, final)
    return epilines1, epilines2

def fundamentalMat(features):
    x = features[:,0:2]
    x_ = features[:,2:4]

    xnorm, T1 = norm(x)
    x_norm, T2 = norm(x_)
        
    A = np.zeros((len(xnorm),9))

    for i in range(0, len(xnorm)):
        x_1 = xnorm[i][0]
        y_1 = xnorm[i][1]
        x_ = x_norm[i][0]
        y_= x_norm[i][1]
        A[i] = np.array([x_1*x_, x_*y_1, x_, y_*x_1, y_*y_1, y_, x_1, y_1, 1])

    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    F = Vt.T[:, -1]
    F = F.reshape(3,3)

    U, S, Vt = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    F = np.dot(U, np.dot(S,Vt))
    F = np.dot(T2.T, np.dot(F, T1))
    return F

def ransac(feat):
    n = 1000
    thresh = 0.02
    thresh_in = 0
    F = 0

    for count in range(n):
        idx = []
        idx_rand = np.random.choice(feat.shape[0], size=8)
        ft = feat[idx_rand, :] 
        f = fundamentalMat(ft)
        
        for num in range(feat.shape[0]):
            ft = feat[num]
            err = error(ft, f)
            if err < thresh:
                idx.append(num)

        if len(idx) > thresh_in:
            thresh_in = len(idx)
            choice = idx
            F = f

    return F, feat[choice, :]

def drawMatches(Image_1, Image_2, pairs, file_name):
    final = np.concatenate((Image_1, Image_2), axis = 1)
    x1 = pairs[:,0].astype(int)
    y1 = pairs[:,1].astype(int)
    x2 = pairs[:,2].astype(int)
    y2 = pairs[:,3].astype(int)
    x2 += Image_1.shape[1]
    for i in range(x1.shape[0]):
        cv2.line(final, (x1[i], y1[i]), (x2[i] ,y2[i]), (255,0,0), 2)    
    final_resized = cv2.resize(final, (int(final.shape[1]/2), int(final.shape[0]/2)))
    cv2.imshow(file_name, final_resized)
    cv2.waitKey() 
    cv2.destroyAllWindows()

def norm(pts):
    mean = np.mean(pts, axis=0)
    meanx = mean[0]
    meany = mean[1]
    xt = pts[:, 0] - meanx
    yt = pts[:, 1] - meany
    s = (2 / np.mean(xt ** 2 + yt ** 2)) ** 0.5
    scaleT = np.diag([s, s, 1])
    transT = np.array([[1, 0, -meanx], [0, 1, -meany], [0, 0, 1]])
    T = scaleT.dot(transT)
    homo_pts = np.column_stack((pts, np.ones(len(pts))))
    norm_pts = (T.dot(homo_pts.T)).T
    norm_pts = norm_pts[:, :-1]
    return norm_pts, T

def countPositive(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    count = 0
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        X_ = np.dot(P, X)
        X_ = X_ / X_[3]
        z = X_[2]
        if z > 0:
            count += 1
    return count

def error(feature, F): 
    x1 = feature[0:2]
    x2 = feature[2:4]
    newx1=np.array([x1[0], x1[1], 1]).T
    newx2=np.array([x2[0], x2[1], 1])
    err = np.dot(newx1, np.dot(F, newx2))
    return np.abs(err)

# MAIN CODE 

dataset = "3" # Select from (1. artroom), (2. chess), (3. ladder)

if dataset == "1":
    Image_1 = cv2.imread('data/artroom/im0.png')
    Image_2 = cv2.imread('data/artroom/im1.png')
    K1 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]])
    K2 = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]])
    baseline=536.62

elif dataset == "2":
    Image_1 = cv2.imread('data\chess\im0.png')
    Image_2 = cv2.imread('data\chess\im1.png')
    K1 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]])
    K2 = np.array([[1758.23, 0, 829.15], [0, 1758.23, 552.78], [0, 0, 1]])
    baseline=97.99

elif dataset == "3":
    Image_1 = cv2.imread('data\ladder\im0.png')
    Image_2 = cv2.imread('data\ladder\im1.png')
    K1 = np.array([[1734.16, 0, 333.49], [0, 1734.16, 958.05], [0, 0, 1]])
    K2 = np.array([[1734.16, 0, 333.49], [0, 1734.16, 958.05], [0, 0, 1]])
    baseline=228.38

else:
    print("Invalid Selection. Try Again!")
    quit()

img = Image_2.copy() 
sift = cv2.SIFT_create()

gray_1 = cv2.cvtColor(Image_1, cv2.COLOR_BGR2GRAY) 
gray_2 = cv2.cvtColor(Image_2, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray_1, None)
kp2, des2 = sift.detectAndCompute(gray_2, None)

bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
imp = matches[0:500]

feature_match = cv2.drawMatches(Image_1, kp1, Image_2, kp2, imp, Image_1, flags=2)
cv2.imwrite("result/matches.png", feature_match)

for m in imp:
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    pairs.append([p1[0], p1[1], p2[0], p2[1]])
pairs = np.array(pairs).reshape(-1, 4)

F, feature_match = ransac(pairs)   
print("Fundamental Matrix:\n", F,"\n")  
drawMatches(Image_1, Image_2, feature_match, "result/feature_match.png")

E = K2.T.dot(F).dot(K1)
U,s,V = np.linalg.svd(E)
s = [1,1,0]
E = np.dot(U,np.dot(np.diag(s),V))
print("Essential Matrix:\n", E,"\n")

U, _, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

R2.append(np.dot(U, np.dot(W, Vt)))
R2.append(np.dot(U, np.dot(W, Vt)))
R2.append(np.dot(U, np.dot(W.T, Vt)))
R2.append(np.dot(U, np.dot(W.T, Vt)))
C2.append(U[:, 2])
C2.append(-U[:, 2])
C2.append(U[:, 2])
C2.append(-U[:, 2])

for n in range(4):
    if (np.linalg.det(R2[n]) < 0):
        R2[n] = -R2[n]
        C2[n] = -C2[n]

R1 = I = np.identity(3)
k = 3
C1 = np.zeros((3,1))
P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
for i in range(len(C2)):
    P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))
    pts3D.append(cv2.triangulatePoints(P1, P2, pairs[:,0:2].T, pairs[:,2:4].T))

P = np.dot(K2, np.dot(R2[k], np.hstack((I, -C2[k].reshape(3,1)))))

X = pts3D[k]
Xn = np.dot(P, X)
Xn = Xn/Xn[2,:]

a = Xn[0, :].T
b = Xn[1, :].T

for i in range(feature_match.shape[0]):
    x1, y1 = a[i], b[i]
    x2, y2 = feature_match[i, 2], feature_match[i, 3]
    cv2.circle(img, (int(x1), int(y1)), 7, (0,100,10), 7)
    cv2.circle(img, (int(x2), int(y2)), 2, (10,200,10), 2)

cv2.imshow("Reprojected", img)
cv2.waitKey() 
cv2.destroyAllWindows()

for i in range(len(pts3D)):
    pt = pts3D[i]
    pt = pt/pt[3, :]
    x = pt[0,:]
    y = pt[1, :]
    z = pt[2, :]    
    z1.append(countPositive(pt, R1, C1))
    z2.append(countPositive(pt, R2[i], C2[i]))

count_thresh = int(pts3D[0].shape[1] / 2)
index = np.intersect1d(np.where(np.array(z1) > count_thresh), np.where(np.array(z2) > count_thresh))
R2 = R2[index[0]]
C2 = C2[index[0]]

print("Rotation: \n", R2, "\n")
print("Camera Center: \n", C2, "\n")

pair1, pair2 = feature_match[:,0:2], feature_match[:,2:4]
epilines1, epilines2 = epipolarLines(pair1, pair2, F, Image_1, Image_2, "result/epipolarlines.png", False)

h_img1, w_img1 = Image_1.shape[:2]
h_img2, w_img2 = Image_2.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pair1), np.float32(pair2), F, imgSize=(w_img1, h_img1))

img1rect = cv2.warpPerspective(Image_1, H1, (w_img1, h_img1))
img2rect = cv2.warpPerspective(Image_2, H2, (w_img2, h_img2))

pair1rect = cv2.perspectiveTransform(pair1.reshape(-1, 1, 2), H1).reshape(-1,2)
pair2rect = cv2.perspectiveTransform(pair2.reshape(-1, 1, 2), H2).reshape(-1,2)
    
cv2.imwrite("result/Rectified1.png", img1rect)
cv2.imwrite("result/Rectified2.png", img2rect)

print("Rectification Homography for img1: \n H1 \n", H1)
print("Rectification Homography for img2: \n H2 \n", H2)

rect_F = np.dot(np.linalg.inv(H2.T), np.dot(F, np.linalg.inv(H1)))
epilines1_rectified, epilines2_rectified = epipolarLines(pair1rect, pair2rect, rect_F, img1rect, img2rect, "result/RectifiedEpilines.png", True)

img1rect = cv2.resize(img1rect, (int(img1rect.shape[1] / 4), int(img1rect.shape[0] / 4)))
img2rect = cv2.resize(img2rect, (int(img2rect.shape[1] / 4), int(img2rect.shape[0] / 4)))

img1rect = cv2.cvtColor(img1rect, cv2.COLOR_BGR2GRAY)
img2rect = cv2.cvtColor(img2rect, cv2.COLOR_BGR2GRAY)

wsize = 3 

img1rect = img1rect.astype(int)
img2rect = img2rect.astype(int)

height, width = img1rect.shape
dispimg = np.zeros((height, width))

newx = width - (2 * wsize)

print("Computing Disparity. Please Wait.")
for y in range(wsize, height-wsize):
    img1b = []
    img2b = []
    for x in range(wsize, width-wsize):
        left = img1rect[y:y + wsize, x:x + wsize]
        img1b.append(left.flatten())

        right = img2rect[y:y + wsize, x:x + wsize]
        img2b.append(right.flatten())

    img1b = np.array(img1b)
    img1b = np.repeat(img1b[:, :, np.newaxis], newx, axis=2)

    img2b = np.array(img2b)
    img2b = np.repeat(img2b[:, :, np.newaxis], newx, axis=2)
    img2b = img2b.T

    dif = np.abs(img1b - img2b)
    totaldif = np.sum(dif, axis = 1)
    index = np.argmin(totaldif, axis = 0)
    disp = np.abs(index - np.linspace(0, newx, newx, dtype=int)).reshape(1, newx)
    dispimg[y, 0:newx] = disp 

normdispimg = np.uint8(dispimg * 255 / np.max(dispimg))
plt.imshow(normdispimg, cmap='gray', interpolation='nearest')
plt.savefig("result/disparity1.png")
plt.imshow(normdispimg, cmap='hot', interpolation='nearest')
plt.savefig("result/disparity2.png")

depth = (baseline * K1[0,0]) / (dispimg + 1e-10)
depth[depth > 200000] = 200000
depth_map = np.uint8(depth * 255 / np.max(depth))

plt.imshow(depth_map, cmap='hot', interpolation='nearest')
plt.savefig("result/depth1.png")
plt.show()
plt.imshow(depth_map, cmap='gray', interpolation='nearest')
plt.savefig("result/depth2.png")
plt.show()