import cv2
import math
import copy
import time
import pandas as pd
import numpy as np

start_time = time.time()

cap = cv2.VideoCapture('/Users/yanda/Downloads/output_video1.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/Users/yanda/Downloads/output_video_path11.mp4", fourcc, fps, frame_size)
excel_path = "/Users/yanda/Downloads/frame_data11.xlsx"

records = []
frame_index = 0

def run_dynamic(mask, frame, start, end):
    


    SP_x, SP_y = start[0], start[1]
    EP_x, EP_y = end[0], end[1]




    #image_path = ‘’
    alpha = 6
    diff_co = 0.03
    scale = 0.18  #um/pix
    MR_speed = 1000  #um/s
    MR_radius = 2.15 #um
    deltas = [0] * 100
    save_path = '/Users/yanda/Downloads/click_test_result.png'
    debug = False
    triangle = False

    # draw the path
    #image = cv2.imread(image_path)
    image_height, image_width = mask.shape[:2]



    # Convert image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold for color
    _, threshold_white = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)


    # Detect color
    contours_white, _ = cv2.findContours(threshold_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # Center of area
    centers_radius_white = []
    centers_red = []
    centers_yellow = []
    all_bboxes = []

    for contour in contours_white:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 判断长边和短边以及分割份数
        if w > h:
            ratio = w / h
            n = math.ceil(ratio)  # 取大于等于 ratio 的最小整数
            if ratio >= 1.5:
                bboxes = []
                part_width = w // n  # 每个子区域的宽度（整数除法）
                for i in range(n):
                    # 最后一个区域用剩余部分，避免丢失像素
                    current_width = w - part_width * i if i == n - 1 else part_width
                    bbox = (x + i * part_width, y, current_width, h)
                    bboxes.append(bbox)
            else:
                # 当 n < 2 时，认为不需要拆分
                bboxes = [(x, y, w, h)]
        else:
            ratio = h / w
            n = math.ceil(ratio)
            if ratio >= 1.5:
                bboxes = []
                part_height = h // n
                for i in range(n):
                    current_height = h - part_height * i if i == n - 1 else part_height
                    bbox = (x, y + i * part_height, w, current_height)
                    bboxes.append(bbox)
            else:
                bboxes = [(x, y, w, h)]
        
        # 对每个子区域进行处理
        for (x_new, y_new, w_new, h_new) in bboxes:
            # 绘制子区域矩形
            #cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 0, 255), 2)
            
            # 计算子区域中心（原始坐标系：左上角为原点）
            center_x = x_new + w_new / 2
            center_y_original = y_new + h_new / 2
            # 如果需要以左下角为原点，则转换 y 坐标
            center_y = center_y_original
            
            radius = int(0.5 * np.sqrt(w_new**2 + h_new**2))

            #cv2.circle(image, (int(center_x), int(center_y)), radius, (0, 255, 0), 3)
            
            centers_radius_white.append(((center_x, center_y), radius, (x_new, y_new, w_new, h_new)))
            all_bboxes.append((x_new, y_new, w_new, h_new))

    

    # 初始参数设置
    #SP_x, SP_y = centers_red[0][0], centers_red[0][1]
    SP_ini = (SP_x, SP_y)
    #EP_x, EP_y = centers_yellow[0][0], centers_yellow[0][1]
    EP_ini = (EP_x, EP_y)





    
    def run_entire_code(SP_ini, EP_ini, mask, frame):

        SP_x, SP_y = SP_ini[0], SP_ini[1]
        EP_x, EP_y = EP_ini[0], EP_ini[1]

       

        p_cx = [point[0][0] for point in centers_radius_white]
        p_cy = [point[0][1] for point in centers_radius_white]
        R_c = [point[1] for point in centers_radius_white]

        def draw_dashed_line(img, start, end, color, thickness, dash_length=5):
            # 计算线段的长度
            length = np.linalg.norm(np.array(end) - np.array(start))
            # 计算线段上的单位向量q
            unit_vector = (np.array(end) - np.array(start)) / length
            num_dashes = int(length // dash_length)

            for i in range(num_dashes):
                start_point = start + unit_vector * (i * dash_length)
                end_point = start + unit_vector * ((i + 0.5) * dash_length)
                img = cv2.line(img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), color, thickness)

            return img

        def draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=5):
            # 计算圆周的长度
            circumference = 2 * np.pi * radius
            num_dashes = int(circumference // dash_length)

            for i in range(num_dashes):
                start_angle = i * (360 / num_dashes)
                end_angle = (i + 0.5) * (360 / num_dashes)
                img = cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

            return img

        def draw_line_as_points(frame, point1, point2, image_height, color=(255, 0, 255), thickness=10):
            """Draw a line as a series of points on the image."""
            x1, y1 = point1
            x2, y2 = point2
            
            # Generate a series of points between point1 and point2
            num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to generate
            x_points = np.linspace(x1, x2, num=int(num_points/30), dtype=int)
            y_points = np.linspace(y1, y2, num=int(num_points/30), dtype=int)

            for x, y in zip(x_points, y_points):
                cv2.circle(frame, (x, y), thickness, color, -1)

        def intersection_points(point1, slopes, point2, k):
            x1, y1 = point1
            x2, y2 = point2
            intersections = []

            for slope in slopes:
                if slope != k:
                    x_intersect = (slope * x1 - k * x2 + y2 - y1) / (slope - k)
                    y_intersect = slope * (x_intersect - x1) + y1
                    intersections.append((x_intersect, y_intersect))
                else:
                    intersections.append(None)  # Parallel lines don't intersect
            
            return intersections

        def distance(point1, point2):
            """Calculate the Euclidean distance between two points."""
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def line_circle_intersection(A, B, center, radius):
            """计算线段 AB 与以 center 为圆心、radius 为半径的圆的交点，
            仅返回那些落在线段 AB 内的交点（即对应参数 t 在 [0,1] 内）。
            """
            Ax, Ay = A
            Bx, By = B
            Cx, Cy = center

            dx = Bx - Ax
            dy = By - Ay
            fx = Ax - Cx
            fy = Ay - Cy

            a = dx * dx + dy * dy
            b = 2 * (fx * dx + fy * dy)
            c = (fx * fx + fy * fy) - radius * radius

            if a == 0:  # AB 为单个点
                if c <= 0:  # 点在圆内或圆上
                    return [A]
                else:
                    return []  # 点在圆外

            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return []  # 无交点
            elif discriminant == 0:
                t = -b / (2 * a)
                if 0 <= t <= 1:
                    return [(Ax + t * dx, Ay + t * dy)]
                else:
                    return []  # 虽然相切，但交点不在线段上
            else:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                points = []
                if 0 <= t1 <= 1:
                    intersection1 = (Ax + t1 * dx, Ay + t1 * dy)
                    points.append(intersection1)
                if 0 <= t2 <= 1:
                    intersection2 = (Ax + t2 * dx, Ay + t2 * dy)
                    points.append(intersection2)
                return points

        '''def perpendicular_intersection(waypoint, closest_points):
            """Calculate the intersection of the line through waypoint with slope k and the perpendicular line through closest_points."""
            WP_x, WP_y = waypoint
            CP_x, CP_y = closest_points
            if CP_x == WP_x:
                # closest_points line is vertical, perpendicular line is horizontal
                return (WP_x, CP_y)
            elif CP_y == WP_y:
                # closest_points line is horizontal, perpendicular line is vertical
                return (CP_x, WP_y)
            else:
                # Calculate slopes
                slope_CP = (CP_y - WP_y) / (CP_x - WP_x)
                perp_slope = -1 / slope_CP
                # Calculate intersection point
                x_intersect = (WP_y - CP_y + perp_slope * CP_x - slope_CP * WP_x) / (perp_slope - slope_CP)
                y_intersect = slope_CP * (x_intersect - WP_x) + WP_y
                return (x_intersect, y_intersect)'''

        def perpendicular_intersection(waypoint, closest_points):
            x1, y1 = waypoint
            x2, y2 = closest_points
            
            x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
            y = k1 * (x - x1) + y1
            
            return (x, y)

        def determine_end_point(waypoint, closest_points, closest_rc, intersections):
            # Step 1: Calculate intersection point
            intersection = perpendicular_intersection(waypoint, closest_points[0])
            
            # 如果只有一个最近点，则只计算与该点的距离
            if len(closest_points) == 1:
                distance_to_closest_points1 = distance(intersection, closest_points[0])
                # 根据实际需求，判断是否返回 intersection 或做其他处理
                if distance_to_closest_points1 > closest_rc[0]:
                    return intersection
                else:
                    # 遍历 intersections 寻找最合适的交点
                    min_distance = float('inf')
                    closest_intersection = None
                    for point in intersections:
                        dist_waypoint_to_point = distance(waypoint, point)
                        dist_point_to_closest_points = distance(point, closest_points[0])
                        if dist_waypoint_to_point < min_distance and dist_point_to_closest_points > closest_rc[0]:
                            min_distance = dist_waypoint_to_point
                            closest_intersection = point
                    return closest_intersection

            # 如果有两个或以上最近点，则计算两个距离
            elif len(closest_points) >= 2:
                distance_to_closest_points1 = distance(intersection, closest_points[0])
                distance_to_closest_points2 = distance(intersection, closest_points[1])
                if distance_to_closest_points1 > closest_rc[0] and distance_to_closest_points2 > closest_rc[1]:
                    return intersection
                else:
                    # 遍历 intersections 寻找最合适的交点
                    min_distance = float('inf')
                    closest_intersection = None
                    for point in intersections:
                        dist_waypoint_to_point = distance(waypoint, point)
                        dist_point_to_closest_points = distance(point, closest_points[0])
                        if dist_waypoint_to_point < min_distance and dist_point_to_closest_points > closest_rc[0]:
                            min_distance = dist_waypoint_to_point
                            closest_intersection = point
                    return closest_intersection
            
        def check_line_through_circles(A, B, circles, func, *func_args):
            for center, radius in circles:
                intersection_points = line_circle_intersection(A, B, center, radius)
                if len(intersection_points) == 2:
                    return func(*func_args)
            return B  # One or zero intersections, return B point

  

        # find next closest points
        def closest_two_points(origin, points, R_c, intersections):
            # 创建points、R_c和intersections的副本
            points_copy = copy.deepcopy(points)
            R_c_copy = copy.deepcopy(R_c)
            intersections_copy = copy.deepcopy(intersections)

            # 如果points列表只有一个值，返回updated_points, updated_rc, updated_intersections
            if len(points) == 1:
                return points_copy, R_c_copy, intersections_copy, points_copy, R_c_copy, intersections_copy
            
            # 将点、R_c和intersections值打包在一起
            point_rc_int_pairs = list(zip(points_copy, R_c_copy, intersections_copy))
            
            # 初始化最小距离和最近点坐标
            min_distances = [(float('inf'), None), (float('inf'), None)]  # (distance, (point, rc, intersection))

            # 遍历点集合
            for point, rc, intersection in point_rc_int_pairs:
                # 计算当前点到原点的距离
                distance = math.sqrt((point[0] - origin[0])**2 + (point[1] - origin[1])**2)
                
                # 如果当前距离比已知的最小距离还小，更新最小距离和最近点坐标
                if distance < min_distances[0][0]:
                    min_distances[1] = min_distances[0]
                    min_distances[0] = (distance, (point, rc, intersection))
                elif distance < min_distances[1][0]:
                    min_distances[1] = (distance, (point, rc, intersection))

            # 提取最近的两个点及其对应的R_c和intersections值
            closest_points_rc_int = [min_distances[0][1], min_distances[1][1]]
            closest_points = [closest_points_rc_int[0][0], closest_points_rc_int[1][0]]
            closest_rc = [closest_points_rc_int[0][1], closest_points_rc_int[1][1]]
            closest_intersections = [closest_points_rc_int[0][2], closest_points_rc_int[1][2]]

            # 从副本列表中剔除最近的一个点及其对应的R_c值和intersections值
            points_copy.remove(closest_points[0])
            R_c_copy.remove(closest_rc[0])
            intersections_copy.remove(closest_intersections[0])

            return closest_points, closest_rc, closest_intersections, points_copy, R_c_copy, intersections_copy

        # find end point of a line
        def end_point(x, y, slopes):
            delta_x = 100 # tangents length
            end_points = []
            for slope in slopes:
                delta_y = slope * delta_x
                end_point = (int(x + delta_x), int(y + delta_y))
                end_points.append(end_point)
            
            return end_points

        # find tangent
        def find_m(x1, y1, h, k, r):
            # Calculate the distance from the external point to the center of the circle
            d = np.sqrt((x1 - h)**2 + (y1 - k)**2)
            
            # Calculate the angle of the line connecting the point to the center
            theta = np.arctan2(y1 - k, x1 - h)
            
            # Calculate the angles of the tangent lines
            if -1 <= r/d <= 1:
                alpha1 = theta + np.arcsin(r/d)
                alpha2 = theta - np.arcsin(r/d)
            else:
                return
            
            # Calculate the slopes of the tangent lines
            m1 = np.tan(alpha1)
            m2 = np.tan(alpha2)

            result = [m1, m2]
            
            return result

        # get rid of some obstacles
        '''def pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha):
            m0 = (EP_y - SP_y) / (EP_x - SP_x)
            e0 = SP_y - m0 * SP_x
            
            filtered_intersections = []
            filtered_intercepts = []
            filtered_p_c = []
            filtered_R_c = []

            # 确定矩形区域的边界
            min_x = min(SP_x, EP_x)
            max_x = max(SP_x, EP_x)
            min_y = min(SP_y, EP_y)
            max_y = max(SP_y, EP_y)

            for i, (x, y) in enumerate(zip(p_cx, p_cy)):
                # 检查点是否在矩形区域内
                if not (min_x <= x <= max_x and min_y <= y <= max_y):
                    continue
                
                mp = -1 / m0
                e = y - mp * x
                
                x_int = (e - e0) / (m0 - mp)
                y_int = m0 * x_int + e0
                
                distance = math.sqrt((x_int - x) ** 2 + (y_int - y) ** 2)
                
                if alpha * R_c[i] > distance:
                    filtered_intersections.append([x_int, y_int])
                    filtered_intercepts.append(e)
                    filtered_p_c.append((x, y))
                    filtered_R_c.append(R_c[i])
            
            return filtered_intersections, filtered_intercepts, filtered_p_c, filtered_R_c, m0, mp'''
        
        def pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha):
            m0 = (EP_y - SP_y) / (EP_x - SP_x)
            e0 = SP_y - m0 * SP_x
            mp = -1 / m0

            # 计算起点到终点的方向向量，用于投影
            d_x = EP_x - SP_x
            d_y = EP_y - SP_y
            denom = d_x * d_x + d_y * d_y

            filtered_intersections = []
            filtered_intercepts = []
            filtered_p_c = []
            filtered_R_c = []

            for i, (x, y) in enumerate(zip(p_cx, p_cy)):
                # 计算点 (x,y) 在起点到终点连线上的投影比例 t
                t = ((x - SP_x) * d_x + (y - SP_y) * d_y) / denom
                if not (0 <= t <= 1):
                    continue
                
                # 计算经过 (x,y) 且与主线垂直的直线的截距
                e = y - mp * x
                
                # 计算该垂线与主线的交点
                x_int = (e - e0) / (m0 - mp)
                y_int = m0 * x_int + e0
                
                # 计算交点与障碍点之间的距离
                distance = math.sqrt((x_int - x) ** 2 + (y_int - y) ** 2)
                
                if alpha * R_c[i] > distance:
                    filtered_intersections.append([x_int, y_int])
                    filtered_intercepts.append(e)
                    filtered_p_c.append((x, y))
                    filtered_R_c.append(R_c[i])
            
            return filtered_intersections, filtered_intercepts, filtered_p_c, filtered_R_c, m0, mp

        def adjust_waypoint(waypoint_current, waypoint_next, circles):
            """
            waypoint_current: (x, y) 当前点
            waypoint_next: (x, y) 目标点
            circles: [((cx, cy), r), ...] 每个圆由圆心和半径组成

            如果 waypoint_next 落在任一圆内，则计算由 waypoint_current 到 waypoint_next 的直线
            与该圆的交点，并更新 waypoint_next 为该交点
            """
            # 遍历所有圆
            for (center, r) in circles:
                cx, cy = center
                # 判断 waypoint_next 是否在圆内（使用严格小于，可根据需求调整为<=）
                if (waypoint_next[0] - cx)**2 + (waypoint_next[1] - cy)**2 < r**2:
                    # 用参数方程描述直线：P(t) = P0 + t * d, t ∈ [0,1]
                    P0x, P0y = waypoint_current
                    P1x, P1y = waypoint_next
                    dx = P1x - P0x
                    dy = P1y - P0y

                    # 将 P(t) 带入圆的方程 (x-cx)² + (y-cy)² = r²，得到关于 t 的二次方程：
                    # A*t² + B*t + C = 0
                    A = dx**2 + dy**2
                    B = 2 * ((P0x - cx) * dx + (P0y - cy) * dy)
                    C = (P0x - cx)**2 + (P0y - cy)**2 - r**2

                    discriminant = B**2 - 4 * A * C
                    if discriminant < 0:
                        # 理论上不会出现这种情况
                        continue

                    sqrt_disc = math.sqrt(discriminant)
                    t1 = (-B + sqrt_disc) / (2 * A)
                    t2 = (-B - sqrt_disc) / (2 * A)

                    # 筛选 t 值在 [0, 1] 范围内的交点
                    valid_ts = [t for t in (t1, t2) if 0 <= t <= 1]
                    if valid_ts:
                        # 如果 waypoint_current 在圆外、waypoint_next 在圆内，
                        # 那么直线会“进入”圆一次，对应 t 较大的值即为圆边界交点
                        t_intersect = max(valid_ts) - 0.05
                        new_waypoint_next = (P0x + t_intersect * dx, P0y + t_intersect * dy)
                        # 更新 waypoint_next 为交点，并退出循环（若希望处理所有圆，可根据需要修改）
                        return new_waypoint_next
            # 如果 waypoint_next 未落入任何圆中，则原样返回
            return waypoint_next
        
        def compute_intersection_of_line_with_line_t(m, waypoint, p1, p2):
            """
            计算直线 L（斜率 m，经过 waypoint=(wx,wy)）与直线 t（由 p1, p2 构成）的交点。
            如果直线 t 为垂直线，则 m_t 设为 None。
            """
            wx, wy = waypoint
            # 判断直线 t 是否垂直
            if abs(p2[0] - p1[0]) > 1e-6:
                m_t = (p2[1] - p1[1]) / (p2[0] - p1[0])
                b_t = p1[1] - m_t * p1[0]
            else:
                m_t = None

            # 如果直线 t 为垂直，则交点的 x 坐标为 p1[0]
            if m_t is None:
                x_int = p1[0]
                y_int = m * (x_int - wx) + wy
            else:
                # 当两条直线平行时（m 与 m_t 接近）直接跳过（返回 None）
                if abs(m - m_t) < 1e-6:
                    return None
                # L 的方程： y = m*(x - wx) + wy
                # t 的方程： y = m_t * x + b_t
                # 联立得： m*(x - wx) + wy = m_t*x + b_t
                # => (m - m_t)*x = m*wx - wy + b_t
                x_int = (m * wx - wy + b_t) / (m - m_t)
                y_int = m * (x_int - wx) + wy
            return (x_int, y_int)

        def select_farthest_pair(points):
            """
            给定若干点（列表，每个点格式 (x,y)），返回距离最远的两个点的元组。
            """
            max_d = -1
            far_pair = None
            n = len(points)
            for i in range(n):
                for j in range(i+1, n):
                    d = math.hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
                    if d > max_d:
                        max_d = d
                        far_pair = (points[i], points[j])
            return far_pair

        def draw_transparent_triangle(frame, vertices, color, alpha):
            """
            在 image 上绘制填充三角形，vertices 为三角形顶点列表（格式 [(x,y), ...]），
            color 为 BGR 颜色，alpha 为透明度（0～1）。
            """
            overlay = frame.copy()
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # remove far obstacles
        preprocess_result = pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha)
        if preprocess_result is None:
            print("当前点击未生成有效数据，跳过处理。")
            return None
        
        intersections, intercepts, p_c, R_c, m0, mp = preprocess_result

       

        # pack intersection coordinates and related data into a list of tuples
        packed_data = list(zip(intersections, intercepts, p_c, R_c))

        # sort the tuple list based on the abscissa of the intersection point
        sorted_packed_data = sorted(packed_data, key=lambda x: x[0][0])

        # unpack the data and extract the sorted results
        sorted_intersections = [item[0] for item in sorted_packed_data]
        sorted_intercepts = [item[1] for item in sorted_packed_data]
        sorted_p_c = [item[2] for item in sorted_packed_data]
        sorted_R_c = [item[3] for item in sorted_packed_data]

        # dynamic safety radius
        sorted_intersections_np = np.array(sorted_intersections)
        sorted_R_c = np.array(sorted_R_c) 
        d = np.sqrt((sorted_intersections_np[:, 0] - SP_x) ** 2 + (sorted_intersections_np[:, 1] - SP_y) ** 2) * scale
        R_c_dyn = (np.sqrt(4 * diff_co * d / MR_speed) + sorted_R_c * scale + MR_radius) / scale

        real_r = [scale * r for r in R_c_dyn]
        #print(real_r)
        
        # selected obstacles' radius including hyperparameter
        updated_R_c = [r + d for r, d in zip(R_c_dyn, deltas)]
        for (x, y), radius in zip(sorted_p_c, R_c_dyn):
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3)
        # circle with hyperparameter
        for (x, y), radius in zip(sorted_p_c, updated_R_c):
            draw_dashed_circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3, 50)

        # predefined
        k1 = (EP_y - SP_y) / (EP_x - SP_x)
        k2 = -1 / k1
        waypoint0 = [SP_x, SP_y]
        n = len(updated_R_c)

        # initialize
        updated_points = sorted_p_c
        updated_rc = updated_R_c
        updated_intersections = sorted_intersections
        waypoints = []
        waypoints.append(waypoint0)
        total_length = 0

        for i in range(n):
            waypoint_current = waypoints[-1]  # 获取当前的waypoint
        
            if i == 0:
                circles = list(zip(sorted_p_c, updated_R_c))
            else:
                circles = list(zip(updated_points, updated_rc))

            closest_points, closest_rc, closest_intersections, updated_points, updated_rc, updated_intersections = closest_two_points(
                waypoint_current, updated_points, updated_rc, updated_intersections)

            if len(closest_points) == 1:
                result = find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
                if result is None:
                    print("r/d exceed")
                    break
            else:
                result1 = find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
                result2 = find_m(waypoint_current[0], waypoint_current[1], closest_points[1][0], closest_points[1][1], closest_rc[1])
                if result1 is None or result2 is None:
                    print("r/d exceed")
                    break
                result = result1 + result2

            intersections = intersection_points(waypoint_current, result, closest_points[0], k2)

            waypoint_next = check_line_through_circles(waypoint_current, [EP_x, EP_y], circles, 
                                                    determine_end_point, waypoint_current, closest_points, closest_rc, intersections)

            waypoint_next = adjust_waypoint(waypoint_current, waypoint_next, circles)
            
            # draw path
            if waypoint_next is None:
                break
            draw_line_as_points(frame, waypoint_current, waypoint_next, image_height)

            waypoints.append(waypoint_next)  # 添加新的waypoint到列表

            if len(intersections) == 2:
                waypoint_final = [EP_x, EP_y]
                draw_line_as_points(frame, waypoint_next, waypoint_final, image_height)
                waypoints.append(waypoint_final)  # 添加最终的waypoint
                break  # 如果达到了终点，可以退出循环

            if waypoint_next == [EP_x, EP_y]:
                break

            # 对于 results 中每个斜率，计算直线 L（过 waypoint_current，斜率 m）与直线 t 的交点
            intersection_points_x = []
            for m in result:
                pt = compute_intersection_of_line_with_line_t(m, waypoint_current, waypoint_next, closest_points[0])
                if pt is not None:
                    intersection_points_x.append(pt)
            
            # 选出这四个交点（或其中有效的交点）中距离最远的两个
            far_pair = select_farthest_pair(intersection_points_x)
            if far_pair is None:
                print(f"第 {i+1} 次循环：无法选出最远的两个交点，跳过该次计算。")
                continue
            
            # 用 far_pair 中的两个交点与 waypoint_current 构成三角形
            triangle_vertices = [(waypoint_current[0], waypoint_current[1]), 
                                (far_pair[0][0], far_pair[0][1]), 
                                (far_pair[1][0], far_pair[1][1])]
            
            # 在 image_test 上用蓝色透明填充该三角区域，蓝色 BGR=(255, 0, 0)，透明度 0.5
            if triangle:
                draw_transparent_triangle(frame, triangle_vertices, (255, 255, 0), 0.3)

        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        total_length = np.sum(distances)
        obstacle_amount = len(sorted_intersections)        

        if debug:
            for (x,y) in waypoints:
                cv2.circle(frame, (int(x), int(y)), radius=20, color=(255, 255, 0), thickness=-1)

            # selected obstacles' center
            for (pcx, pcy) in p_c:
                cv2.circle(frame, (int(pcx), int(pcy)), radius=10, color=(255, 0, 0), thickness=-1)

            # draw intersection point
            for(x0,y0) in sorted_intersections:
                cv2.circle(frame, (int(x0), int(y0)), radius=10, color=(255, 0, 0), thickness=-1)

            # selected obstacles' perpendicular lines
            for (pc, intersection) in zip(sorted_p_c, sorted_intersections):
                x1, y1 = int(pc[0]), int(pc[1])
                x2, y2 = int(intersection[0]), int(intersection[1])
                draw_dashed_line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5, 25)

            # optimal path
            draw_dashed_line(frame, (int(SP_x), int(SP_y)), (int(EP_x), int(EP_y)), (0, 128, 255), 5, 100)

        #print(f"total length: {total_length:.2f}")

        '''for idx, (x, y) in enumerate(sorted_p_c):
            cv2.putText(
                image,              # 图像
                str(idx+1),            # 文本内容
                (int(x)-15, int(y)+5),        # 坐标转换为整数
                cv2.FONT_HERSHEY_SIMPLEX,# 字体
                2,                       # 字体大小
                (0, 0, 255),             # 颜色 (BGR)
                4                        # 线条粗细
            )'''
            

        return frame, total_length, obstacle_amount, waypoints
    
    
    # 对每一帧进行处理
    processed_frame, total_length, obstacle_amount, waypoints = run_entire_code(SP_ini, EP_ini, mask, frame)

    return processed_frame, waypoints, total_length, obstacle_amount

    """# 显示处理后的帧
    cv2.imshow('geo', processed_frame)

    processing_time = time.time() - start_time
    # 记录当前帧的数据
    records.append({
        "frame": frame_index,
        "processing_time": processing_time,
        "total_length": total_length,
        "obstacle_amount": obstacle_amount
    })
    frame_index += 1
    out.write(processed_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break"""

"""cap.release()
out.release()
cv2.destroyAllWindows()

df = pd.DataFrame(records)
df.to_excel(excel_path, index=False)
print("Excel 文件已保存：frame_data.xlsx")"""