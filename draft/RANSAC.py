
import concurrent.futures
import cv2

import math
import pyttsx3
import threading
import queue
import pyrealsense2 as rs
import numpy as np
import pyrealsense2 as rs
from sklearn.cluster import DBSCAN
import time
import threading
from sklearn.linear_model import LinearRegression
import cmath
import open3d as o3d

flag_process = 0
n = 3
k = 15
dist_ref = 0.03
d = 700
bestFit = None
bestErr = 10000
threshold_ransac = 35
model = LinearRegression()

kernel = np.ones((5, 5), np.uint8)

stop_event = threading.Event()
speech_queue = queue.LifoQueue()

text = ""
text_queue = queue.LifoQueue()

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 180)

ransac_value = 0
ransac_threshold = 4
db_value = 0
db_threshold = 5
len_ransac = 0
len_rs_threshold = 150

def dis_filter_ransac(current_value):
    global ransac_value

    # Nếu không có giá trị trước đó, lưu lại giá trị hiện tại và trả về True
    if (ransac_value == 0):
        ransac_value = current_value
        return True

    # Kiểm tra sự thay đổi giữa giá trị hiện tại và giá trị trước đó
    if abs(current_value - ransac_value) > ransac_threshold:
        ransac_value = current_value  # Cập nhật giá trị trước đó
        return True  # Sự thay đổi đủ lớn, thực hiện hành động

    # Nếu sự thay đổi quá nhỏ, bỏ qua hành động
    return False

class AppState:
    def __init__(self, *args, **kwargs):
        # self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

    def adjust_camera_position(self, delta_x, alpha_angle_Camera):
        """
        Điều chỉnh vị trí camera và góc quay alpha.

        :param delta_x: Khoảng cách di chuyển camera theo trục x (m).
        :param alpha_angle_Camera: Góc alpha (đơn vị độ) quay camera.
        """
        # Cập nhật vị trí camera theo trục X
        self.translation[0] += delta_x  # Di chuyển camera 80 cm (0.8 m) theo trục X

        # Cập nhật góc alpha
        self.alpha_angle_Camera = math.radians(alpha_angle_Camera)  # Chuyển đổi từ độ sang radian
        self.yaw = self.alpha_angle_Camera  # Cập nhật yaw với giá trị alpha mới

        # In ra các giá trị cập nhật
        print(f"Updated Camera Position: {self.translation}")
        print(f"Updated Camera Yaw (alpha): {self.yaw} radians")
# Lop AppState() bao gom thong so huong camera, dich chuyen, khoang cach va trang thai cua chuot
# cung cap thuoc tinh truy cap ma tran xoay va diem tru
state = AppState()
state.adjust_camera_position(delta_x=0.8, alpha_angle_Camera=45)

# Cau hinh depth va color stream, cau hinh luong du lieu
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Yeu cau depth camera co cam bien mau!")
    exit(0)

# luong du lieu (16-bit depth), 30 fps va luong du lieu (8-bit moi kenh mau RGB), 30 fps
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

w, h = depth_intrinsics.width, depth_intrinsics.height

pc = rs.pointcloud()

# Tao bo loc giam kich thuoc du lieu
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

def findOutliers(X, inliers):
    # Chuyển thành tập hợp để loại bỏ trùng lặp
    set_inliers = {tuple(row) for row in inliers}
    set_X = {tuple(row) for row in X}

    # Tìm outliers bằng phép trừ tập hợp
    set_outliers = set_X - set_inliers

    # Chuyển lại thành numpy array
    outliers = np.array(list(set_outliers))
    return outliers


# chuyen doi mang vector 3D thanh mang diem 2D
def project(v):  # "v" mang cac vector 3D
    v = np.array(v)
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
               (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # loai bo cac diem qua gan
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


# tai tao hieu ung, goc nhin va vi tri trong khong gian 3D
def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

# ve diem pointclouds cua dam may diem 3D sang hinh 2D, sap xep tu xa den gan
def pointcloud(out, verts, texcoords, color, painter=True):
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T

    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]

    if painter:
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T

    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    out[i[m], j[m]] = color[u[m], v[m]]


def distance_from_point_to_plane(A, B, C, D, x0, y0, z0):
    # Tính khoảng cách từ điểm (x0, y0, z0) đến mặt phẳng Ax + By + Cz + D = 0
    numerator = abs(A * x0 + B * y0 + C * z0 + D)
    denominator = math.sqrt(A ** 2 + B ** 2 + C ** 2)
    return numerator / denominator

def RANSAC(X, n, k, dist_ref, d, bestErr, bestFit, flag):
    # print(f"X:{len(X)}")
    max_dist_inlier = 0.5
    iterations = 0
    bestFit = [0, 0, 0]
    InliersofBestfit = []
    OutLiersofBestfit = []
    Object = []
    for iterations in range(k):
        loss = []
        maybeInliers = X[np.random.choice(X.shape[0], n, replace=False)]

        XY_maybeInliers = np.array([maybeInliers[:, 0].T, maybeInliers[:, 1].T]).T

        Z_maybeInliers = np.array([maybeInliers[:, 2]]).T

        model.fit(XY_maybeInliers, Z_maybeInliers)

        maybeModel = [model.coef_[0][0], model.coef_[0][1], model.intercept_]
        confirmedInliers = []
        error = 0
        for i in range(X.shape[0]):
            dist = abs(maybeModel[0] * X[i, 0] + maybeModel[1] * X[i, 1] - X[i, 2] + maybeModel[2]) / math.sqrt(
                maybeModel[0] ** 2 + maybeModel[1] ** 2 + 1)
            error += dist
            dist = dist.item()
            if (dist < dist_ref):
                confirmedInliers.append(X[i])
            else:
                loss.append(dist)

        if len(confirmedInliers) > d:
            betterModel = maybeModel
            thisErr = len(X) - len(confirmedInliers)
            if thisErr < bestErr:
                bestFit = betterModel

                InliersofBestfit = np.array(confirmedInliers)
                # print(len(InliersofBestfit))
                bestErr = thisErr
                # OutLiersofBestfit = findOutliers(X, InliersofBestfit, OutLiersofBestfit)

        iterations += 1
    # print(f"Inlier num: {len(InliersofBestfit)}")
    # print(f"Outlier num: {len(OutLiersofBestfit)}")
    # print(bestFit) # BestPlane

    if len(InliersofBestfit) > 0:
        dbscan = DBSCAN(eps=0.08, min_samples=60)
        clusters = dbscan.fit_predict(InliersofBestfit)
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters != -1]
        cluster_centers_3d = []
        for cluster_index in unique_clusters:
            mask = clusters == cluster_index
            points_in_cluster = InliersofBestfit[mask]
            cluster_center_3d = np.mean(points_in_cluster, axis=0)
            cluster_centers_3d.append(cluster_center_3d)
        #print(f'So in: {len(InliersofBestfit)}, So cum: {len(cluster_centers_3d)}')

        if (len(cluster_centers_3d)):
            max_index, max_value = max(enumerate(cluster_centers_3d), key=lambda y: y[1][1])
            mask = clusters == max_index
            InliersofBestfit = InliersofBestfit[mask]
    OutLiersofBestfit = findOutliers(X, InliersofBestfit)

    if (len(OutLiersofBestfit)):
        eps_custom = 0.09  # Điều chỉnh giá trị này để kiểm soát số lượng cụm
        dbscan = DBSCAN(eps=eps_custom, min_samples=30)
        clusters = dbscan.fit_predict(OutLiersofBestfit)
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[unique_clusters != -1]
        cluster_centers_3d = []
        for cluster_index in unique_clusters:
            mask = clusters == cluster_index
            points_in_cluster = OutLiersofBestfit[mask]
            cluster_center_3d = np.mean(points_in_cluster, axis=0)
            cluster_centers_3d.append(cluster_center_3d)
        # print(len(cluster_centers_3d))

        if (len(cluster_centers_3d)):
            max_index, max_value = max(enumerate(cluster_centers_3d), key=lambda y: y[1][1])
            mask = clusters == max_index
            Object_Temp = OutLiersofBestfit[mask]
            if (len(Object_Temp)>60):
                Object_Temp = [point for point in Object_Temp if -0.3 <= point[0] <= 0.3]
                Object = np.array(Object_Temp)
                # print("Obj lon hon 20")
            else:
                if (len(OutLiersofBestfit) > 150):
                    Object_Temp = [point for point in OutLiersofBestfit if -0.3 <= point[0] <= 0.3]
                    Object = np.array(Object_Temp)
                    # print("Do")

    #print(f'Len Out: {len(OutLiersofBestfit)}, Len OBJ: {len(Object)}')

    return InliersofBestfit, OutLiersofBestfit, Object, bestFit

reso_ransac_high = []
reso_ransac_low = []
lock_obj = False
def segmentation_ransac(verts_d, w, h):
    #print(f"Verts:{len(verts_d)}")
    global reso_ransac_high
    global reso_ransac_low
    global lock_obj
    text_ransac = ""
    global flag_process
    if len(verts_d) == 0:
        reso_ransac_high.clear()
        reso_ransac_low.clear()
        segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
        print("a")
        return segmented_image, text_ransac
    # Lấy z
    # else:
    try:  # Không đủ điểm để tạo mặt phẳng
        bestin, bestout, obj, plane = RANSAC(verts_d, n, k, 0.03, d, bestErr, bestFit, True)
        # bestin, bestout, obj, plane = RANSAC_model(verts_d)

        segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)

        # Gán màu sắc cho mỗi nhóm cluster
        if np.all(plane != 0):
            if (len(obj) > 60):
                mean_obj = np.mean(obj, axis=0)
                check = plane[0] * mean_obj[0] + plane[1] * mean_obj[1] - mean_obj[2] + plane[2]
                # print(f'Z_obj: {check}')
                if (check > 0):
                    reso_ransac_low.clear()
                    #min_z = min(obj, key=lambda z: z[2])
                    dis_arr = [distance_from_point_to_plane(plane[0], plane[1], -1, plane[2], x, y, z) * 100 for  x, y, z in obj]
                    min_dis = max(dis_arr)
                    if ((min_dis>3) & (lock_obj == False)):
                        reso_ransac_high.append([len(obj), min_dis])
                       # print("Thỏa high")
                        #print(len(reso_ransac_high))
                        if (len(reso_ransac_high) >= 3):
                            text_ransac = f"High Step: {max(reso_ransac_high, key=lambda item: item[0])[1][0]} centimeter"
                            if (max(reso_ransac_high, key=lambda item: item[0])[1][0] >= threshold_ransac):
                                text_ransac = f"Stop ahead because distance is {max(reso_ransac_high, key=lambda item: item[0])[1][0]} centimeter"
                            lock_obj = True
                            reso_ransac_high.clear()
                    else:
                        reso_ransac_high.clear()
                else:
                    reso_ransac_high.clear()
                    dis_arr = [distance_from_point_to_plane(plane[0], plane[1], -1, plane[2], x, y, z) * 100 for x, y, z in obj]
                    max_dis = max(dis_arr)
                    if ((max_dis > 3) & (lock_obj == False)):
                        #print("Thỏa Low")
                        reso_ransac_low.append([len(obj), max_dis])
                        if ((len(reso_ransac_low) == 3)):
                            text_ransac = f"Low Step: {max(reso_ransac_low, key=lambda item: item[0])[1][0]} centimeter"
                            if (max(reso_ransac_high, key=lambda item: item[0])[1][0] >= threshold_ransac):
                                text_ransac = f"Stop ahead because distance is {max(reso_ransac_low, key=lambda item: item[0])[1][0]} centimeter"
                            lock_obj = True
                            reso_ransac_low.clear()
                    else:
                        reso_ransac_low.clear()
                points_2d_bo = project(bestout[:, :3])
                bo_img_mask = ((points_2d_bo[:, 0] >= 0) & (points_2d_bo[:, 0] <= w)
                               & (points_2d_bo[:, 1] >= 0) & (points_2d_bo[:, 1] <= h))
                points_2d_bo = points_2d_bo[bo_img_mask]
                segmented_image[points_2d_bo[:, 1].astype(int), points_2d_bo[:, 0].astype(int)] = [255, 255, 0]

                points_2d_ob = project(obj[:, :3])
                ob_img_mask = ((points_2d_ob[:, 0] >= 0) & (points_2d_ob[:, 0] <= w)
                               & (points_2d_ob[:, 1] >= 0) & (points_2d_ob[:, 1] <= h))
                points_2d_ob = points_2d_ob[ob_img_mask]
                segmented_image[points_2d_ob[:, 1].astype(int), points_2d_ob[:, 0].astype(int)] = [60, 30, 255]

                if len(bestin):
                    points_2d_in = project(bestin[:, :3])
                    in_img_mask = (points_2d_in[:, 0] >= 0) & (points_2d_in[:, 0] <= w) & (
                            points_2d_in[:, 1] >= 0) & (points_2d_in[:, 1] <= h)
                    points_2d_in = points_2d_in[in_img_mask]
                    segmented_image[points_2d_in[:, 1].astype(int), points_2d_in[:, 0].astype(int)] = [0, 255, 0]
            else:
                reso_ransac_high.clear()
                reso_ransac_low.clear()
                if len(bestin):
                    points_2d_in = project(bestin[:, :3])
                    in_img_mask = (points_2d_in[:, 0] >= 0) & (points_2d_in[:, 0] <= w) & (points_2d_in[:, 1] >= 0) & (
                            points_2d_in[:, 1] <= h)
                    points_2d_in = points_2d_in[in_img_mask]
                    segmented_image[points_2d_in[:, 1].astype(int), points_2d_in[:, 0].astype(int)] = [0, 255, 0]
                points_2d_bo = project(bestout[:, :3])
                bo_img_mask = ((points_2d_bo[:, 0] >= 0) & (points_2d_bo[:, 0] <= w)
                               & (points_2d_bo[:, 1] >= 0) & (points_2d_bo[:, 1] <= h))
                points_2d_bo = points_2d_bo[bo_img_mask]
                segmented_image[points_2d_bo[:, 1].astype(int), points_2d_bo[:, 0].astype(int)] = [255, 255, 0]
                if (len(obj) == 0):
                    lock_obj = False
                    # print("openlock")

        else:
            lock_obj = False
            reso_ransac_high.clear()
            reso_ransac_low.clear()
    except:
        reso_ransac_high.clear()
        reso_ransac_low.clear()
        segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
        print("No enough points to create plane")
    # segmented_image = cv2.dilate(segmented_image, kernel, iterations=1)

    return segmented_image, text_ransac

def ransac_plane(pcd):
    cur_time = time.time()
    # Define a bounding box for cropping (min_bound, max_bound)
    # Let's say we want to crop the region where x, y, z are between certain values
#    min_bound = np.array([-0.2, -0.2, -1])  # Minimum corner of the box
#    max_bound = np.array([0.2, 0.2, 1])  # Maximum corner of the box

    min_bound = np.array([-0.5, -0.4, 0])  # Minimum corner of the box
    max_bound = np.array([0.5, 0.6, 7])  # Maximum corner of the box
    # Create a 3D bounding box using min_bound and max_bound
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop the point cloud using the bounding box
    cropped_pcd = pcd.crop(bbox)

    # Visualize the cropped point cloud
    # o3d.visualization.draw_geometries([cropped_pcd])

    # Lấy lại các điểm đã được downsampled
    verts_downsampled_ransac = np.asarray(cropped_pcd.points)
    #print(verts_downsampled_ransac)
    verts_downsampled_ransac = np.array(verts_downsampled_ransac)
    image_width = out.shape[1]
    image_height = out.shape[0]
    # Phân đoạn pointcloud bằng K-means clustering
    segmented_image = segmentation_ransac(verts_downsampled_ransac, image_width, image_height)
    end_time = time.time()
    inference_time = end_time - cur_time
    # fps = 1 / inference_time
    # Hiển thị FPS trên màn hình
    #print(f'FPS RANSAC: {fps:.2f}')
    # cv2.putText(segmented_image, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    #             cv2.LINE_AA)
    return segmented_image

out = np.empty((h, w, 3), dtype=np.uint8)
flag_speak = True


def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            if not state.paused:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_frame = decimate.process(depth_frame)

                depth_intrinsics = rs.video_stream_profile(
                    depth_frame.profile).get_intrinsics()
                w, h = depth_intrinsics.width, depth_intrinsics.height

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = np.asanyarray(
                    colorizer.colorize(depth_frame).get_data())

                if state.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = pc.calculate(depth_frame)
                pc.map_to(mapped_frame)

                v, t = points.get_vertices(), points.get_texture_coordinates()

                verts_int = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                texcoords_int = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

                # Loc theo khoang cach
                min_distance = 0.2
                max_distance = 1.5
                # Tinh khoang cach cac diem
                distances = np.linalg.norm(verts_int, axis=1)
                # print(distances)
                # index của những điểm năm trong khoảng
                valid_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]

                step = 4  # cu 15 diem, giu lai 1 diem pointcloud
                verts = verts_int[valid_indices][::step]
                texcoords = texcoords_int[valid_indices][::step]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(verts)

                # Áp dụng voxel down-sampling
                pcd = pcd.voxel_down_sample(voxel_size=0.008)


                #print("-" * 50)
                image_width = out.shape[1]
                image_height = out.shape[0]
                # Phân đoạn pointcloud bằng K-means clustering

                # Chạy Segmentation và Object Detection song song
                futures = {
                    executor.submit(ransac_plane, pcd): 'ransac',
                }

                results = {

                    'ransac': None,
                }

                for future in concurrent.futures.as_completed(futures):
                    task_name = futures[future]
                    results[task_name] = future.result()

                ransac_image, text_ransac = results['ransac']

                ransac_image_resize = cv2.resize(ransac_image, (640, 360))
                color_image_resize = cv2.resize(color_image, (640,360))
                if (text_ransac != ""):
                    print(f'RANSAC: {text_ransac}')

                combine_img = np.hstack((ransac_image_resize, color_image_resize))

                cv2.imshow("Test", combine_img)

                #cv2.imshow("Test", combine_img)
                if cv2.waitKey(1) == ord('q'):
                    stop_event.set()
                    for future in futures:
                        future.cancel()
                    break
    cv2.destroyAllWindows()
    pipeline.stop()


def voice():
    while not stop_event.is_set():
        try:
            target = text_queue.get(timeout=10)
            # print(target)
            engine.say(target)
            engine.runAndWait()
            # text_queue.task_done()
        except queue.Empty:
            continue
        time.sleep(3)


if __name__ == "__main__":
    speech_thread = threading.Thread(target=voice)
    speech_thread.start()
    stop_event.clear()
    main()