import concurrent.futures
import numpy as np
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
import open3d as o3d


kernel = np.ones((5, 5), np.uint8)

stop_event = threading.Event()
discn = threading.Event()
speech_queue = queue.LifoQueue()

text = ""
text_queue = queue.LifoQueue()
priority_queue = queue.Queue()

engine = pyttsx3.init()
imi_en = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 180)

vcs = imi_en.getProperty('voices')
imi_en.setProperty('voice', vcs[0].id)
imi_en.setProperty('rate', 180)

db_value = 0
db_threshold = 5


def dis_filter_db(current_value):
    global db_value

    # Nếu không có giá trị trước đó, lưu lại giá trị hiện tại và trả về True
    if (db_value == 0):
        db_value = current_value
        return True

    # Kiểm tra sự thay đổi giữa giá trị hiện tại và giá trị trước đó
    if abs(current_value - db_value) > db_threshold:
        db_value = current_value  # Cập nhật giá trị trước đó
        return True  # Sự thay đổi đủ lớn, thực hiện hành động

    # Nếu sự thay đổi quá nhỏ, bỏ qua hành động
    return False


def direction_control(angle):
    pi = math.pi
    angle_d = -1 * angle * 180 / pi  # Chuyển radian → độ

    hour_map = [
        (180, "9"), (135, "10"), (105, "11"), (75, "12"),
        (45, "1"), (15, "2"), (-1, "3")  # Góc nhỏ hơn 15° là 3h
    ]

    for threshold, hour in hour_map:
        if angle_d >= threshold:
            return hour
    return "none"  # Không xác định được hướng


def calculate_angle_from_center(target, shape):
    center = (shape[1] // 2, shape[0])  # Trung tâm cạnh dưới ảnh
    delta_x = target[0] - center[0]
    delta_y = target[1] - center[1]
    angle = np.arctan2(delta_y, delta_x)  # Tính góc (radian)
    return angle

def draw_navigation_3d(frame, shape, target, ground):
    h, w, _ = frame.shape
    center = (w // 2, h)  # Tọa độ tâm của nửa vòng tròn
    radius = w // 8
    color = (228, 198, 80)
    clock = None
    # Vẽ nửa vòng tròn (mốc giờ từ 9h đến 3h)
    cv2.ellipse(frame, center, (radius, radius), 0, 0, 180, color, 2)
    # Danh sách giờ mong muốn
    hours = [9, 10, 11, 12, 1, 2, 3]
    # Vẽ các mốc giờ
    for i, hour in enumerate(hours):
        angle = np.radians(i * 30)  # Góc mỗi giờ (0, 30, 60,...)
        x = int(center[0] - radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle) - 10)

        cv2.putText(frame, f"{hour}", (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Vẽ dấu chia nhỏ giữa các giờ (mỗi 15 độ có một dấu)
        for i in range(len(hours) - 1):  # Lặp qua từng khoảng giữa 9h -> 3h
            angle = np.radians((i * 30) + 15)  # Góc giữa các giờ (15, 45, 75,...)

            x1 = int(center[0] - radius * np.cos(angle))
            y1 = int(center[1] - radius * np.sin(angle))
            x2 = int(center[0] - (radius - 10) * np.cos(angle))
            y2 = int(center[1] - (radius - 10) * np.sin(angle))

            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, target, radius=5, color=(0, 0, 255), thickness=5)
    if (target is not None):
        # Vẽ mũi tên từ trung tâm cạnh dưới đến `target`
        start_point = (shape[1] // 2, shape[0])
        end_point = target
        if (ground == 0):
            cv2.arrowedLine(frame, start_point, end_point, color=(228, 198, 80), thickness=2, tipLength=0.1)
            cv2.circle(frame, start_point, radius=5, color=(228, 198, 80), thickness=5)

        # Tính toán góc hướng di chuyển
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        angle = np.arctan2(delta_y, delta_x)
        clock = direction_control(angle)

        # Hiển thị hướng di chuyển
        cv2.putText(frame, f"{clock} o'clock", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (228, 198, 80), 2)
        # speak_direction(angle)
    return frame, clock

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
# Tạo một tập hợp các màu sắc cố định

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



fixed_colors = [
    [255, 255, 0],  # Vàng
    [255, 0, 255],  # Tím
    [255, 0, 0],  # Đỏ
    [0, 255, 0],  # Lục
    [0, 255, 255],  # Cyan
    [0, 0, 255]  # Lam
]


def find_distances_min_clusters(cluster_centers_3d, cluster_centers_2d):
    # distances_min_clusters = cluster_centers_3d[0][2]
    point_nearest = [0, 0, 0]
    # print(cluster_centers_3d)
    # for cluster_center in cluster_centers_3d:
    #     if cluster_center[2] <= distances_min_clusters:
    #         distances_min_clusters = cluster_center[2]
    #         point_nearest = cluster_center
    distances_min_clusters = [np.linalg.norm(np.array(point_nearest) - np.array(cluster_center))
                              for cluster_center in cluster_centers_3d]
    point_nearest_idx = np.argmin(distances_min_clusters)
    point_nearest = cluster_centers_3d[point_nearest_idx]
    # print(f"point_nearest:{point_nearest}")
    point_nearest_2d = cluster_centers_2d[point_nearest_idx]
    # print(f"point_nearest_2d:{point_nearest_2d}")
    distances_min_cluster = np.min(distances_min_clusters)
    z_min_cluster = math.sqrt(distances_min_cluster ** 2 - (point_nearest[0] ** 2 + point_nearest[1] ** 2))
    # print(f"distances_min_clusters:{z_min_cluster}")

    # tra ve khoang cach, vi tri trai hay phai, huong mui gio cua vat gan nhat
    return z_min_cluster, point_nearest_2d

# Áp dụng DBSCAN
def DBSCAN_segmentation(pcd, w, h):
    ground = 0
    cur_time = time.time()
    pcd_db = pcd.voxel_down_sample(voxel_size=0.008)
    verts_pre = np.asarray(pcd_db.points)
    verts = np.array(verts_pre)

    text_db = ""
    segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
    center_fixed = None
    temp_rounded_decimal_number_cm = None
    # segmented_image = cv2.resize(segmented_image, (648, 380))
    if len(verts) == 0:
        return segmented_image, text_db
    # Lấy z
    points_2d = verts[:, :2]

    eps_custom = 0.09  # Điều chỉnh giá trị này để kiểm soát số lượng cụm
    dbscan = DBSCAN(eps=eps_custom, min_samples=300)
    clusters = dbscan.fit_predict(points_2d)

    # Lọc ra các cụm duy nhất (loại bỏ cụm nhiễu, được gán nhãn -1)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters != -1]
    cluster_centers_3d = []
    cluster_centers_2d = []
    segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
    cluster_color_dict = {}
    num_clusters = len(unique_clusters)
    # print(f"Số lượng cụm: {num_clusters}")
    # Gán màu sắc cho mỗi nhóm cluster
    for cluster_index in unique_clusters:
        color = fixed_colors[len(cluster_color_dict) % len(fixed_colors)]
        cluster_color_dict[cluster_index] = color
        mask = clusters == cluster_index
        points_in_cluster = verts[mask]
        points_2d = project(points_in_cluster[:, :3])
        in_img_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        points_2d = points_2d[in_img_mask]
        segmented_image[points_2d[:, 1].astype(int), points_2d[:, 0].astype(int)] = color
        cluster_center_3d = np.mean(points_in_cluster, axis=0) if points_in_cluster.size > 0 else np.array(
            [np.nan, np.nan, np.nan])
        cluster_centers_3d.append(cluster_center_3d)

        center_2d = project([cluster_center_3d])
        cluster_centers_2d.append(center_2d)

    if cluster_centers_3d:
        # print(verts_clusters)
        decimal_number_meter, point_nearest_2d = find_distances_min_clusters(cluster_centers_3d, cluster_centers_2d)
        # print(f"point_nearest_2d:{point_nearest_2d}")
        if (point_nearest_2d is not None):
            center = [point_nearest_2d[:, 0].astype(int), point_nearest_2d[:, 1].astype(int)]
            center_fixed = (center[0][0], center[1][0])
            # Chuyển đổi từ mét sang centimet
        decimal_number_cm = decimal_number_meter * 100
        # Làm tròn đến 0 chữ số thập phân
        temp_rounded_decimal_number_cm = round(decimal_number_cm, 0)

        #print(f'cenx {center_fixed[1]}, {h/2}')
        #print(f'lencl: {len(cluster_centers_3d)}')
        if (temp_rounded_decimal_number_cm >= 70) and (center_fixed[1] > h/2):
            ground = 1
    dbscan_image, clock = draw_navigation_3d(segmented_image, segmented_image.shape, center_fixed, ground)
    if (temp_rounded_decimal_number_cm is None) or (clock is None):
        #print("No Objects")
        text_queue.put(text_db)
    else:
        if ground:
            #print("Ground")
            text_queue.put(text_db)
        else:
            # if dis_filter_db(temp_rounded_decimal_number_cm):
            text_db = f"{temp_rounded_decimal_number_cm:.0f} at {clock} clock"
            text_queue.put(text_db)
    end_time = time.time()
    inference_time = end_time - cur_time
    # fps = 1 / inference_time
    # # Hiển thị FPS trên màn hình
    # #print(f'FPS DBSCAN: {fps:.2f}')
    # cv2.putText(segmented_image, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    #             cv2.LINE_AA)
    return segmented_image, text_db


out = np.empty((h, w, 3), dtype=np.uint8)
flag_speak = True

def speak_immediately():
    while not discn.is_set():
        try:
            imi_en.say(priority_queue.get())
            # imi_en.runAndWait()
        except queue.Empty:
            continue


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    is_pipeline_started = False

    while not is_pipeline_started:
        try:
            pipeline.start(config)
            is_pipeline_started = True
            print("[INFO] Camera started.")
        except Exception as e:
            continue

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            if not state.paused:
                try:
                    frames = pipeline.wait_for_frames()

                except RuntimeError as e:
                    print(f"[WARN] Camera error: {e}")
                    print("[INFO] Waiting for camera to reconnect...")
                    discn.set()
                    priority_queue.put("Camera disconected!")
                    # Đợi cho đến khi camera được kết nối lại
                    while True:
                        try:
                            pipeline.stop()
                        except Exception:
                            pass  # có thể chưa start thì không sao

                        try:
                            pipeline.start(config)
                            print("[INFO] Camera reconnected.")
                            priority_queue.put("Camera reconnected!")
                            break
                        except Exception as e:
                            print(f"[WARN] Reconnect failed: {e}")
                            priority_queue.put("Camera disconected!")
                            continue

                    continue  # quay lại đầu vòng lặp
                discn.clear()
                config.enable_stream(rs.stream.depth, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
                # frames = pipeline.wait_for_frames()
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

                futures = {
                    executor.submit(DBSCAN_segmentation, pcd, image_width, image_height): 'dbscan',
                }

                results = {
                    'dbscan': None,
                }

                for future in concurrent.futures.as_completed(futures):
                    task_name = futures[future]
                    results[task_name] = future.result()

                dbscan_image, text_db = results['dbscan']
                dbscan_image_resize = cv2.resize(dbscan_image, (640, 360))
                color_image_resize = cv2.resize(color_image, (640,360))

                if (text_db != ""):
                    print(f'DBSCAN: {text_db}')
                combine_img = np.hstack((dbscan_image_resize, color_image_resize))

                cv2.imshow("Test", combine_img)

                #cv2.imshow("Test", dbscan_image_resize)
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
            if discn.is_set():
                target4 = priority_queue.get(timeout=5)
                engine.say(target4)
                engine.runAndWait()

            target = text_queue.get(timeout=10)
            engine.say(target)
            engine.runAndWait()

            # text_queue.task_done()
        except queue.Empty:
            continue


if __name__ == "__main__":
    speech_thread = threading.Thread(target=voice)
    speech_thread.start()
    discn.clear()
    stop_event.clear()
    main()