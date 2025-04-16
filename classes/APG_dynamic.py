import cv2
import math
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

# video input and output
cap = cv2.VideoCapture('/Users/yanda/Downloads/Plan_Planning/no_collision3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/Users/yanda/Downloads/test_result.mp4", fourcc, fps, frame_size)
excel_path = "/Users/yanda/Downloads/test_result.xlsx"
records = []
frame_index = 0

def run_dynamic_pathplanner(mask, frame, start, end, alpha, safety_radius):
    
    trajectory = []
    
    SP_x, SP_y = start[0], start[1]
    EP_x, EP_y = end[0], end[1]

    # static para
    deltas = [0] * 100
    #save_path = 'png'
    debug = False
    triangle = False
    #initial_interval = 50
    #step = 10

    # dynamic para
    grid_width = 500
    arrival_threshold = 500
    MR_size = 10

    # dynamic initialization
    tracking = []
    all_path = []
    colormap = plt.get_cmap('plasma')
    target_point = []
    initialized = False
    reinitialize = True
    target_idx = 0 
    override_active = False
    trajectory_memory = []

    #image_path = ‘’
    
    safety_radius = safety_radius  # .1 > 20
    alpha = alpha                  # 1 - 20

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
    all_bboxes = []
    sub_contours_color = []

    '''# contour mesh 遍历每个识别到的轮廓
    for cnt in contours_white:
        # 1. 获取当前轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 2. 在边界框范围内创建一个二值掩模，并绘制当前轮廓（注意坐标偏移）
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_shifted = cnt - [x, y]
        cv2.drawContours(mask, [cnt_shifted], -1, 255, thickness=cv2.FILLED)
        
        # 3. 对边界框区域按照网格划分
        for i in range(0, w, grid_width):
            for j in range(0, h, grid_width):
                # 截取当前网格单元
                cell = mask[j:j+grid_width, i:i+grid_width]
                # 如果该区域存在轮廓部分（非零像素数大于0）
                if cv2.countNonZero(cell) > 0:
                    # 对当前网格单元进行轮廓提取
                    sub_cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for sub in sub_cnts:
                        # 调整子轮廓的坐标：先加上网格单元在局部图中的偏移，再加上边界框偏移
                        sub = sub + [i, j]   # 网格单元内偏移
                        sub = sub + [x, y]   # 边界框偏移
                        sub_contours_color.append(sub)

    for contour in sub_contours_color:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 绘制子区域矩形
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # 计算子区域中心（原始坐标系：左上角为原点）
        center_x = x + w / 2
        center_y_original = y + h / 2
        # 如果需要以左下角为原点，则转换 y 坐标
        center_y = image_height - center_y_original
        
        radius = int(safety_radius * np.sqrt(w**2 + h**2))

        #cv2.circle(image, (int(center_x), image_height - int(center_y)), radius, (0, 255, 0), 3)
        
        centers_radius_white.append(((center_x, center_y), radius, (x, y, w, h)))'''

    for contour in contours_white:
        x, y, w, h = cv2.boundingRect(contour)
        # 判断横向还是纵向长
        if w > h:
            ratio = w / h
            if ratio >= 1.5:
                n = math.ceil(ratio)
                part_width = w // n
                # 利用列表推导式构造所有子区域，最后一个区域可能宽度不同
                bboxes = [(x + i * part_width, y,
                            (w - part_width * i) if i == n - 1 else part_width, h)
                        for i in range(n)]
            else:
                bboxes = [(x, y, w, h)]
        else:
            ratio = h / w
            if ratio >= 1.5:
                n = math.ceil(ratio)
                part_height = h // n
                bboxes = [(x, y + i * part_height, w,
                            (h - part_height * i) if i == n - 1 else part_height)
                        for i in range(n)]
            else:
                bboxes = [(x, y, w, h)]
        
        # 对每个子区域进行处理
        for (x_new, y_new, w_new, h_new) in bboxes:

            #cv2.rectangle(image, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 0, 255), 2)

            # 计算子区域中心（左上角为原点，转换成左下角为原点）
            center_x = x_new + w_new * 0.5
            center_y = y_new + h_new * 0.5
            
            # 使用 math.hypot 替代 np.sqrt 进行半径计算
            radius = int(safety_radius * math.hypot(w_new, h_new)) + MR_size
            
            centers_radius_white.append(((center_x, center_y), radius, (x_new, y_new, w_new, h_new)))
            all_bboxes.append((x_new, y_new, w_new, h_new))

    # 初始参数设置
    SP_ini = (SP_x, SP_y)
    if not override_active:
        EP_ini = (EP_x, EP_y)

    distances = (EP_x - SP_x)**2 + (EP_y - SP_y)**2

    def run_entire_code(SP_ini, EP_ini, mask, frame, distances):

        SP_x, SP_y = SP_ini[0], SP_ini[1]
        EP_x, EP_y = EP_ini[0], EP_ini[1]

        p_cx = [point[0][0] for point in centers_radius_white]
        p_cy = [point[0][1] for point in centers_radius_white]
        R_c = [point[1] for point in centers_radius_white]

        def draw_dashed_line(img, start, end, color, thickness, dash_length=5):
            # 将起始点转换为 numpy 数组
            start_arr = np.array(start, dtype=float)
            end_arr = np.array(end, dtype=float)
            diff = end_arr - start_arr
            length = np.linalg.norm(diff)
            if length == 0:
                return img  # 避免除 0
            unit_vector = diff / length
            num_dashes = int(length // dash_length)

            for i in range(num_dashes):
                # 计算每段虚线的起点和终点
                s_pt = start_arr + unit_vector * (i * dash_length)
                e_pt = start_arr + unit_vector * ((i + 0.5) * dash_length)
                # 直接在 cv2.line 中转换为整型元组
                cv2.line(img, tuple(s_pt.astype(int)), tuple(e_pt.astype(int)), color, thickness)

            return img

        '''def draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=5):
            # 计算圆周的长度
            circumference = 2 * np.pi * radius
            num_dashes = int(circumference // dash_length)

            for i in range(num_dashes):
                start_angle = i * (360 / num_dashes)
                end_angle = (i + 0.5) * (360 / num_dashes)
                img = cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

            return img'''

        def draw_line_as_points(image, point1, point2, image_height, draw, color=(0, 191, 255), thickness=10):
            """Draw a line as a series of points on the image and return the list of coordinate tuples."""
            x1, y1 = point1
            x2, y2 = point2

            num_pixels = max(abs(x2 - x1), abs(y2 - y1))
            # 保证至少两个点，同时调节采样密度，这里除以50可以根据实际情况调整
            num_samples = max(2, int(num_pixels / 50))
            x_points = np.linspace(x1, x2, num=num_samples, dtype=int)
            y_points = np.linspace(y1, y2, num=num_samples, dtype=int)

            if draw:
                for x, y in zip(x_points, y_points):
                    cv2.circle(image, (x, y), thickness, color, -1)
                    trajectory.append([x,y])
            
            return list(zip(x_points.tolist(), y_points.tolist()))
            #print("this should be called all the time")
                
            #trajectory_memory.append(trajectory)  #add entire trajectory list to a memory list to record all previous trajectories
            #if len(trajectory_memory) > 3:
            #    trajectory_memory.pop(0)

        def intersection_points(point1, slopes, point2, k):
            x1, y1 = point1
            x2, y2 = point2
            intersections = []

            for slope in slopes:
                if slope != k:
                    denom = slope - k
                    x_intersect = (slope * x1 - k * x2 + y2 - y1) / denom
                    y_intersect = slope * (x_intersect - x1) + y1
                    intersections.append((x_intersect, y_intersect))
                else:
                    intersections.append(None)  # 平行线没有交点
            return intersections

        def distance(point1, point2):
            """Calculate the Euclidean distance between two points."""
            return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

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
            # 如果 a==0，表示 A 和 B 重合
            if a == 0:
                return [A] if (fx * fx + fy * fy) <= radius * radius else []

            b = 2 * (fx * dx + fy * dy)
            c = (fx * fx + fy * fy) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                return []  # 无交点
            elif discriminant == 0:
                t = -b / (2 * a)
                return [(Ax + t * dx, Ay + t * dy)] if 0 <= t <= 1 else []
            else:
                sqrt_disc = math.sqrt(discriminant)
                denom = 2 * a  # 统一分母
                t1 = (-b + sqrt_disc) / denom
                t2 = (-b - sqrt_disc) / denom
                points = []
                if 0 <= t1 <= 1:
                    points.append((Ax + t1 * dx, Ay + t1 * dy))
                if 0 <= t2 <= 1:
                    points.append((Ax + t2 * dx, Ay + t2 * dy))
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

        def perpendicular_intersection(waypoint, closest_point, k1, k2):
            """
            计算从 waypoint 到直线（斜率为 k1）与另一条直线（斜率为 k2）之间的垂直交点。
            注意：原代码依赖全局 k1 和 k2，这里建议作为参数传入。
            """
            x1, y1 = waypoint
            x2, y2 = closest_point
            denominator = (k1 - k2)
            if denominator == 0:
                return None  # 处理平行情况，可根据需要调整
            x = (k1 * x1 - y1 - k2 * x2 + y2) / denominator
            y = k1 * (x - x1) + y1
            return (x, y)

        def determine_end_point(waypoint, closest_points, closest_rc, intersections, k1, k2):
            """
            根据 waypoint 和最近点（closest_points）计算理想的终点。
            参数：
            - waypoint：参考点
            - closest_points：一个或多个最近点（元组列表）
            - closest_rc：与最近点相关的距离阈值列表，如 [rc1] 或 [rc1, rc2]
            - intersections：预计算的交点列表
            - k1, k2：用于计算垂直交点的斜率参数
            """
            # Step 1: 计算垂直交点
            intersection = perpendicular_intersection(waypoint, closest_points[0], k1, k2)
            if intersection is None:
                return None  # 或者返回其它默认值

            if len(closest_points) == 1:
                d0 = distance(intersection, closest_points[0])
                if d0 > closest_rc[0]:
                    return intersection
                else:
                    min_distance = float('inf')
                    closest_intersection = None
                    for point in intersections:
                        d_waypoint = distance(waypoint, point)
                        d_closest = distance(point, closest_points[0])
                        if d_waypoint < min_distance and d_closest > closest_rc[0]:
                            min_distance = d_waypoint
                            closest_intersection = point
                    return closest_intersection

            elif len(closest_points) >= 2:
                d0 = distance(intersection, closest_points[0])
                d1 = distance(intersection, closest_points[1])
                if d0 > closest_rc[0] and d1 > closest_rc[1]:
                    return intersection
                else:
                    min_distance = float('inf')
                    closest_intersection = None
                    for point in intersections:
                        d_waypoint = distance(waypoint, point)
                        d_closest0 = distance(point, closest_points[0])
                        d_closest1 = distance(point, closest_points[1])
                        if d_waypoint < min_distance and d_closest0 > closest_rc[0] and d_closest1 > closest_rc[1]:
                            min_distance = d_waypoint
                            closest_intersection = point
                    return closest_intersection
            
        def check_line_through_circles(A, B, circles, func, *func_args):
            """
            检查线段 AB 是否穿过 circles 中的某个圆（要求交点数正好为 2），
            如果是，则调用 func(*func_args) 并返回其结果；否则返回 B 点。
            """
            for center, radius in circles:
                pts = line_circle_intersection(A, B, center, radius)
                if len(pts) == 2:
                    return func(*func_args)
            return B  # 若穿过的圆不足两个交点，返回 B 点

        # find next closest points
        def closest_two_points(origin, points, R_c, intersections):
            """
            输入：
            origin: (x, y) 原点坐标
            points: [(x, y), ...] 每个圆的圆心
            R_c: [r, ...] 每个圆的半径（与 points 一一对应）
            intersections: [intersection, ...] 每个圆对应的交点信息，
                ——这些交点均在一条直线上且顺序已经排列好，
                    列表第一个点最靠近 (SP_x, SP_y)，最后一个点靠近 (EP_x, EP_y)
            功能：
            1. 筛选掉 origin 落在对应圆内的圆；
            2. 在剩余圆中找出距离 origin 最近的两个点（及对应信息）；
            3. 从副本列表中剔除距离 origin 最近的那个点及其对应信息；
            4. 根据给定直线（经过 (SP_x, SP_y) 与 (EP_x, EP_y)）对剩余列表进行投影筛选，
                仅保留靠近 (EP_x, EP_y) 侧的点。
            返回：
            closest_points, closest_rc, closest_intersections,
            updated_points, updated_R_c, updated_intersections
            """
            # 如果元素是不可变对象，用 list() 复制即可
            points_copy = list(points)
            R_c_copy = list(R_c)
            intersections_copy = list(intersections)
            
            # 筛选：移除所有 origin 落在对应圆内的圆（使用平方距离避免调用 sqrt）
            filtered = [(p, rc, inter) for p, rc, inter in zip(points_copy, R_c_copy, intersections_copy)
                        if (origin[0] - p[0])**2 + (origin[1] - p[1])**2 >= rc**2]
            if not filtered:
                return [], [], [], [], [], []
            points_copy, R_c_copy, intersections_copy = map(list, zip(*filtered))
            
            # 如果筛选后只有一个圆，直接返回
            if len(points_copy) == 1:
                return points_copy, R_c_copy, intersections_copy, points_copy, R_c_copy, intersections_copy

            # 寻找距离 origin 最近的两个点（使用平方距离）
            min1, min2 = float('inf'), float('inf')
            closest1, closest2 = None, None
            for p, rc, inter in zip(points_copy, R_c_copy, intersections_copy):
                d_sq = (p[0] - origin[0])**2 + (p[1] - origin[1])**2
                if d_sq < min1:
                    min2, closest2 = min1, closest1
                    min1, closest1 = d_sq, (p, rc, inter)
                elif d_sq < min2:
                    min2, closest2 = d_sq, (p, rc, inter)
            closest_points = [closest1[0], closest2[0]]
            closest_rc = [closest1[1], closest2[1]]
            closest_intersections = [closest1[2], closest2[2]]
            
            # 从副本中剔除距离 origin 最近的那个点
            # 假设最近点是唯一的
            for idx, p in enumerate(points_copy):
                if p == closest_points[0]:
                    del points_copy[idx]
                    del R_c_copy[idx]
                    del intersections_copy[idx]
                    break
            
            # 投影筛选：保留交点投影在直线上靠近 (EP_x, EP_y) 的点
            # 这里假定全局变量 SP_x, SP_y, EP_x, EP_y 已经定义
            SP = np.array([SP_x, SP_y])
            EP = np.array([EP_x, EP_y])
            line_vec = EP - SP
            line_len_sq = np.dot(line_vec, line_vec)
            origin_np = np.array(origin)
            t_proj = np.dot(origin_np - SP, line_vec) / line_len_sq

            new_points_copy = []
            new_R_c_copy = []
            new_intersections_copy = []
            for p, rc, inter in zip(points_copy, R_c_copy, intersections_copy):
                inter_np = np.array(inter)
                t_inter = np.dot(inter_np - SP, line_vec) / line_len_sq
                if t_inter >= t_proj:
                    new_points_copy.append(p)
                    new_R_c_copy.append(rc)
                    new_intersections_copy.append(inter)
            
            return closest_points, closest_rc, closest_intersections, new_points_copy, new_R_c_copy, new_intersections_copy

        # find end point of a line
        '''def end_point(x, y, slopes):
            delta_x = 100 # tangents length
            end_points = []
            for slope in slopes:
                delta_y = slope * delta_x
                end_point = (int(x + delta_x), int(y + delta_y))
                end_points.append(end_point)
            
            return end_points'''

        # find tangent
        def find_m(x1, y1, h, k, r):
            """
            给定外部点 (x1, y1) 和圆心 (h, k) 及圆半径 r，
            计算该点与圆的两条切线的斜率，返回 [m1, m2]。
            """
            # 计算外部点到圆心的距离（使用 np.hypot 更直观）
            d = np.hypot(x1 - h, y1 - k)
            if d == 0:
                return None  # 点与圆心重合，切线不存在
            ratio = r / d
            # 如果 ratio 超出定义域，则无切线
            if abs(ratio) > 1:
                return None
            # 缓存 arcsin 值
            delta = np.arcsin(ratio)
            theta = np.arctan2(y1 - k, x1 - h)
            m1 = np.tan(theta + delta)
            m2 = np.tan(theta - delta)
            return [m1, m2]

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
            # 主线的斜率及截距（假设 SP_x != EP_x）
            m0 = (EP_y - SP_y) / (EP_x - SP_x)
            e0 = SP_y - m0 * SP_x
            # 主线的垂直方向斜率
            mp = -1 / m0

            # 计算起点到终点的方向向量及其长度平方
            d_x = EP_x - SP_x
            d_y = EP_y - SP_y
            denom = d_x * d_x + d_y * d_y

            filtered_intersections = []
            filtered_intercepts = []
            filtered_p_c = []
            filtered_R_c = []

            # 使用 zip 遍历各点信息
            for x, y, r in zip(p_cx, p_cy, R_c):
                # 计算点 (x,y) 在主线上投影的比例 t
                t = ((x - SP_x) * d_x + (y - SP_y) * d_y) / denom
                if not (0 <= t <= 1):
                    continue

                # 计算经过 (x,y) 且与主线垂直的直线的截距
                e = y - mp * x

                # 计算该垂直线与主线的交点
                # 此处 m0 - mp 不为 0（否则主线与垂直线平行，不可能）
                x_int = (e - e0) / (m0 - mp)
                y_int = m0 * x_int + e0

                # 用平方距离避免调用 sqrt，每次比较时同时平方 alpha * r
                dist_sq = (x_int - x) ** 2 + (y_int - y) ** 2
                if (alpha * r) ** 2 > dist_sq:
                    filtered_intersections.append([x_int, y_int])
                    filtered_intercepts.append(e)
                    filtered_p_c.append((x, y))
                    filtered_R_c.append(r)

            return filtered_intersections, filtered_intercepts, filtered_p_c, filtered_R_c, m0, mp
        
        # 辅助函数：判断一个点是否严格在任意圆内
        def in_any_circle(point, circles):
            x, y = point
            for idx, (center, rad) in enumerate(circles):
                cx, cy = center
                if (x - cx)**2 + (y - cy)**2 < rad**2:
                    return True, idx  # 返回 True 和圆的索引
            return False, -1  # 没有落在任何圆内

        def adjust_waypoint(waypoint_current, waypoint_next, circles):
            """
            如果 waypoint_next 落在某个圆内，则：
            1. 利用全局变量 k2 构造经过该圆心的直线（斜率为 k2）
            2. 求该直线与圆的两个交点（切点）
            3. 检查这两个交点是否分别落在任意圆内
                - 如果只有一个交点在圆内，则返回另一个交点；
                - 如果两个交点都在或都不在，则返回离 waypoint_current 更近的交点。
            如果 waypoint_next 不在任何圆内，则直接返回 waypoint_next。
            """
            for (center, r) in circles:
                cx, cy = center
                # 判断 waypoint_next 是否在圆内（使用平方距离比较）
                if (waypoint_next[0] - cx)**2 + (waypoint_next[1] - cy)**2 < r**2:
                    # 构造经过圆心且斜率为 k2 的直线： y = k2*(x - cx) + cy
                    # 计算该直线与圆的切点： (x - cx)^2 + (k2*(x - cx))^2 = r^2
                    denom = math.sqrt(1 + k2**2)
                    u = r / denom
                    p1 = (cx + u, cy + k2 * u)
                    p2 = (cx - u, cy - k2 * u)

                    # 检查两个交点是否落在任意圆内
                    p1_inside = in_any_circle(p1, circles)
                    p2_inside = in_any_circle(p2, circles)

                    if p1_inside and not p2_inside:
                        return p2
                    elif p2_inside and not p1_inside:
                        return p1
                    else:
                        # 比较平方距离
                        d1 = (p1[0] - waypoint_current[0])**2 + (p1[1] - waypoint_current[1])**2
                        d2 = (p2[0] - waypoint_current[0])**2 + (p2[1] - waypoint_current[1])**2
                        return p1 if d1 <= d2 else p2

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
                # 当两直线平行时直接返回 None
                if abs(m - m_t) < 1e-6:
                    return None
                # 联立直线方程求解交点
                x_int = (m * wx - wy + b_t) / (m - m_t)
                y_int = m * (x_int - wx) + wy

            return (x_int, y_int)
        
        def select_farthest_pair(points):
            """
            给定若干点（列表，每个点格式 (x,y)），返回距离最远的两个点的元组。
            优化：采用平方距离比较，避免每次调用开方运算。
            """
            max_d_sq = -1
            far_pair = None
            n = len(points)
            for i in range(n):
                for j in range(i+1, n):
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    d_sq = dx * dx + dy * dy
                    if d_sq > max_d_sq:
                        max_d_sq = d_sq
                        far_pair = (points[i], points[j])
            return far_pair

        def draw_transparent_triangle(image, vertices, color, alpha):
            """
            在 image 上绘制填充三角形，vertices 为三角形顶点列表（格式 [(x,y), ...]），
            color 为 BGR 颜色，alpha 为透明度（0～1）。
            """
            overlay = image.copy()
            pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        def filter_sync_by_projection(points, R_c, intersections, origin, SP, EP):
            """
            输入：
            points: [(x, y), ...] 需要过滤的点列表（假定这些点都落在直线上）
            R_c: [r, ...] 与 points 对应的半径列表
            intersections: [intersection, ...] 与 points 对应的交点信息列表
            origin: (x, y) 用于投影的原点
            SP: (x, y) 直线起点（靠近 SP 侧）
            EP: (x, y) 直线终点（靠近 EP 侧）
            
            功能：
            1. 将 origin 垂直投影到由 SP 和 EP 所确定的直线上，得到投影参数 t_proj。
            2. 对 points 列表中每个点计算其在直线上对应的参数 t，只有当 t >= t_proj 时保留该点及对应 R_c 和 intersections。
            
            返回：
            filtered_points, filtered_R_c, filtered_intersections
            """
            SP_arr = np.array(SP)
            EP_arr = np.array(EP)
            origin_arr = np.array(origin)
            
            line_vec = EP_arr - SP_arr
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                raise ValueError("SP 和 EP 不能是同一点")
            
            # 计算 origin 在直线上的投影参数
            t_proj = np.dot(origin_arr - SP_arr, line_vec) / line_len_sq
            
            # 将 points 转换为 numpy 数组（形状 (N,2)）
            pts = np.array(points)
            if pts.size == 0:
                print("No points to process.")
                t_pts = np.array([])  # So later code doesn't crash
            else:
                t_pts = np.dot(pts - SP_arr, line_vec) / line_len_sq


            mask = t_pts >= t_proj
            # 过滤后的点转换回列表
            filtered_points = pts[mask].tolist()
            filtered_R_c = [rc for rc, m in zip(R_c, mask) if m]
            filtered_intersections = [inter for inter, m in zip(intersections, mask) if m]
            
            return filtered_points, filtered_R_c, filtered_intersections
        
        '''def move_away_from_line(waypoint_current, SP, EP, global_circles, initial_interval, step):
            """
            从 waypoint_current 出发，沿着远离直线（由 SP 和 EP 定义）的方向移动，
            移动的距离从 initial_interval 开始，每次增加 step，直到找到一个不在 global_circles 内的点，
            返回该点。

            参数：
            waypoint_current: (x, y) 当前点
            SP: (x, y) 直线起点
            EP: (x, y) 直线终点
            global_circles: [(center, radius), ...] 圆的列表（用于 in_any_circle 检测）
            initial_interval: 初始移动距离（例如 10）
            step: 每次增加的距离（例如 10）

            返回：
            (x, y) 新点，该点沿垂直于直线方向移动，且不在任一圆内
            """
            # 转换为 numpy 数组（浮点型）
            A = np.array(waypoint_current, dtype=float)
            S = np.array(SP, dtype=float)
            E = np.array(EP, dtype=float)
            
            # 直线向量及其长度平方
            line_vec = E - S
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                raise ValueError("SP 和 EP 不能是同一点")
            
            # 计算 A 在直线上的投影
            t = np.dot(A - S, line_vec) / line_len_sq
            projection = S + t * line_vec
            
            # 垂直向量 v：如果 A 恰在直线上，则选择 (-dy, dx) 作为垂直方向
            v = A - projection
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-6:
                v = np.array([-line_vec[1], line_vec[0]])
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-6:
                    raise ValueError("直线向量异常")
            
            unit_dir = v / norm_v  # 单位垂直方向
            
            interval = initial_interval
            while True:
                new_point = A + interval * unit_dir  # 沿垂直方向移动
                # 检查 new_point 是否在 global_circles 内（in_any_circle 接受元组）
                if not in_any_circle(tuple(new_point), global_circles):
                    return tuple(new_point)
                interval += step'''

        # remove far obstacles
        preprocess_result = pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha)
        if preprocess_result is None:
            print("当前点击未生成有效数据，跳过处理。")
            return None
        
        # 解包预处理结果
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

        # 根据超参数 deltas（例如：deltas = [0]*100）更新障碍物半径
        updated_R_c = [r + d for r, d in zip(sorted_R_c, deltas)]

        # 绘制障碍物（绘制原始半径；若需要用 updated_R_c 绘制虚线圆，可打开对应代码）
        for (x, y), radius in zip(sorted_p_c, sorted_R_c):
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        '''
        # 绘制带超参数的虚线圆（调试用）
        for (x, y), radius in zip(sorted_p_c, updated_R_c):
            draw_dashed_circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3, 50)
        '''

        # 预定义直线斜率（用于后续计算）
        if EP_x == SP_x:
            EP_x = EP_x + 1
        k1 = (EP_y - SP_y) / (EP_x - SP_x)
        k2 = -1 / k1

        # 初始 waypoint 为起点
        waypoint0 = [SP_x, SP_y]

        # 这里 n 表示更新后的障碍数量（可能用于后续循环）
        n = len(updated_R_c)

        # 对数据重命名，便于后续处理（若后续不需要区分，直接使用 sorted_p_c、sorted_intersections 等即可）
        updated_points = sorted_p_c
        updated_rc = updated_R_c
        updated_intersections = sorted_intersections

        # 初始化航点列表与累计路径长度
        waypoints = [waypoint0]
        total_length = 0

        # 构造全局障碍信息列表，后续 in_any_circle 调用会使用
        global_circles = list(zip(sorted_p_c, updated_R_c))
        dynamic_updated_R_c = [x for x in updated_R_c]
        dynamic_global_circles = list(zip(sorted_p_c, dynamic_updated_R_c))
        # 初始化 far_pairs（后续可能用于求最远点对）
        far_pairs = []

        # 若无障碍，则直接绘制起点到终点的直线
        if n == 0:
            draw_line_as_points(frame, SP_ini, EP_ini, image_height, True)

        for i in range(n):
            # 当前起点始终取最新添加的航点
            waypoint_current = waypoints[-1]
            
            # 第一次循环使用初始数据，否则使用更新后的障碍数据
            circles = list(zip(sorted_p_c, updated_R_c)) if i == 0 else list(zip(updated_points, updated_rc))

            # 找到当前起点与障碍（更新后的列表）之间最近的两个点及相关信息
            closest_points, closest_rc, closest_intersections, updated_points, updated_rc, updated_intersections = closest_two_points(
                waypoint_current, updated_points, updated_rc, updated_intersections)
            
            # 若没有可用障碍，则直接将终点作为下一个航点，并退出循环
            if len(closest_points) == 0:
                waypoint_final = [EP_x, EP_y]
                waypoints.append(waypoint_final)
                break

            # 根据最近点数量，计算斜率(s)；当只有一个障碍时调用 find_m，
            # 当有两个时分别计算并合并结果
            if len(closest_points) == 1:
                result = find_m(waypoint_current[0], waypoint_current[1],
                                closest_points[0][0], closest_points[0][1],
                                closest_rc[0])
                if result is None:
                    print("r/d exceed")
                    break
            else:
                result1 = find_m(waypoint_current[0], waypoint_current[1],
                                closest_points[0][0], closest_points[0][1],
                                closest_rc[0])
                result2 = find_m(waypoint_current[0], waypoint_current[1],
                                closest_points[1][0], closest_points[1][1],
                                closest_rc[1])
                if result1 is None or result2 is None:
                    print("r/d exceed")
                    break
                # 合并两个结果（这里假设 result1、result2 均为列表或可迭代对象）
                result = result1 + result2

            # 根据当前 waypoint_current、result（斜率信息）及第一个最近障碍，计算交点集合
            intersections = intersection_points(waypoint_current, result, closest_points[0], k2)
            
            # 利用 check_line_through_circles 判断并得到下一个航点（若交点数量不合要求则返回 waypoint_next = B，即终点）
            waypoint_next = check_line_through_circles(waypoint_current, [EP_x, EP_y], circles, 
                                                    determine_end_point, waypoint_current, closest_points, closest_rc, intersections, k1, k2)
            if waypoint_next is None:
                break
            # 根据障碍位置调整 waypoint_next
            waypoint_next = adjust_waypoint(waypoint_current, waypoint_next, circles)
            if waypoint_next is None:
                break

            # 添加新的航点
            waypoints.append(waypoint_next)

            # 检查是否达到终点或交点数表明路径已闭合
            if len(intersections) == 2:
                waypoint_final = [EP_x, EP_y]
                '''# 计算当前航点到下一个航点之间的中间点序列（不绘制端点）
                midpoints = draw_line_as_points(frame, waypoint_current, waypoint_next, image_height, False)[1:-1]
                # 添加中点（可选，确保列表中有至少一个点）
                midpoints.append(((waypoint_current[0] + waypoint_next[0]) / 2, (waypoint_current[1] + waypoint_next[1]) / 2))
                # 若中间任一点落在障碍内，则通过 move_away_from_line 调整
                for midpoint in midpoints:
                    if in_any_circle(midpoint, global_circles):
                        waypoint_middle = move_away_from_line(waypoint_current, (SP_x, SP_y), (EP_x, EP_y), global_circles, initial_interval, step)
                        # 回退部分航点，确保路径调整（这里以删除最后两个航点为例）
                        waypoints = waypoints[:-2]
                        cv2.circle(frame, (int(waypoint_middle[0]), int(waypoint_middle[1])), 15, (0, 128, 255), -1)
                        waypoints.append(waypoint_middle)
                        break'''
                waypoints.append(waypoint_final)
                break

            # 若已经到达终点，则退出循环
            if waypoint_next == [EP_x, EP_y]:
                break

            # 如果启用 triangle 模式，则进行额外的三角形构造与绘制
            if triangle:
                if i == 0:
                    intersection_points_x = []
                    for m in result:
                        pt = compute_intersection_of_line_with_line_t(m, waypoint_current, waypoint_next, closest_points[0])
                        if pt is not None:
                            intersection_points_x.append(pt)
                    far_pair = select_farthest_pair(intersection_points_x)
                    if far_pair is None:
                        print(f"第 {i+1} 次循环：无法选出最远的两个交点，跳过该次计算。")
                        continue
                    triangle_vertices = [(waypoint_current[0], waypoint_current[1]),
                                        (far_pair[0][0], far_pair[0][1]),
                                        (far_pair[1][0], far_pair[1][1])]
                    far_pairs.extend(far_pair)
                    draw_transparent_triangle(frame, triangle_vertices, (219, 112, 147), 0.3)

            '''# 计算当前段的中间点序列，再次检查其中是否有点落入障碍内
            midpoints = draw_line_as_points(frame, waypoint_current, waypoint_next, image_height, False)[1:-1]
            midpoints.append(((waypoint_current[0] + waypoint_next[0]) / 2, (waypoint_current[1] + waypoint_next[1]) / 2))
            for midpoint in midpoints:
                if in_any_circle(midpoint, global_circles):
                    waypoint_middle = move_away_from_line(waypoint_current, (SP_x, SP_y), (EP_x, EP_y), global_circles, initial_interval, step)
                    # 回退部分航点以进行调整
                    waypoints = waypoints[:-2]
                    cv2.circle(frame, (int(waypoint_middle[0]), int(waypoint_middle[1])), 10, (0, 128, 255), -1)
                    waypoints.append(waypoint_middle)
                    break'''

            # 根据最新的 waypoint_next 更新障碍数据（利用投影过滤，同步更新列表）
            updated_points, updated_rc, updated_intersections = filter_sync_by_projection(updated_points, updated_rc, updated_intersections, 
                                                                                        waypoint_next, (SP_x, SP_y), (EP_x, EP_y))

        # 循环结束后，将所有航点依次连接
        for i in range(len(waypoints) - 1):
            draw_line_as_points(frame, waypoints[i], waypoints[i+1], image_height, True)

        # 利用 NumPy 计算航点之间距离与总路径长度
        total_length = distances
        obstacle_amount = 1

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
        
        # check microrobot's position is inside circle or not
        check_position = False
        collision_center = []
        if SP_ini is not None:
            robot_position = np.array(SP_ini)
            decide_D, collision_index = in_any_circle(robot_position, dynamic_global_circles)
            if len(dynamic_global_circles) > 0:
                collision_center = dynamic_global_circles[collision_index][0]
            if decide_D:
                check_position = True
                
        #print("trajectory = ", trajectory)
        return frame, total_length, obstacle_amount, trajectory, check_position, collision_center
    
    
    # 对每一帧进行处理
    processed_frame, total_length, obstacle_amount, trajectory, check_position, collision_center = run_entire_code(SP_ini, EP_ini, mask, frame, distances)

    # mirror next waypoint when collision happens
    if check_position == True and not initialized:
        mirror_x = 2 * SP_ini[0] - collision_center[0]
        mirror_y = 2 * SP_ini[1] - collision_center[1]
        EP_ini = (mirror_x, mirror_y)
        initialized = True
        override_active = True
        print("change target point")

    da = SP_ini[0] - EP_ini[0]
    db = SP_ini[1] - EP_ini[1]
    threshold = da**2 + db**2

    if threshold < arrival_threshold:
        # 如果处于覆盖状态下，先取消覆盖，使得下一次循环继续使用列表中的目标
        if override_active:
            override_active = False
            EP_ini = (EP_x, EP_y)
        # 如果当前到达的是列表中的目标，就更新为下一目标
        else:
            # 这里直接更新EP_ini为新的列表目标，也可以等待下一次循环通过非覆盖分支更新
            EP_ini = (EP_x, EP_y)
        
        reinitialize = True
        initialized = False

    return processed_frame, trajectory, total_length, obstacle_amount

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