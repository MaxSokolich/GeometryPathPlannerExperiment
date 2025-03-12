import cv2
import math
import copy
import time
import numpy as np

class geo_algorithm:
    def __init__(self, image, start_point, end_point, alpha_geo, safety_radius, deltas):
        self.start_time = time.time()

        self.image = image
        self.safety_radius = safety_radius
        self.alpha = alpha_geo
        

        self.deltas = deltas # hyperparameter

        print(deltas)
        self.start_point = start_point
        self.end_point = end_point

        self.trajectory = []

        #input
        #image, start_point, end_point

    def draw_dashed_line(self, img, start, end, color, thickness, dash_length=5):
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

    def draw_dashed_circle(self, img, center, radius, color, thickness=1, dash_length=5):
        # 计算圆周的长度
        circumference = 2 * np.pi * radius
        num_dashes = int(circumference // dash_length)

        for i in range(num_dashes):
            start_angle = i * (360 / num_dashes)
            end_angle = (i + 0.5) * (360 / num_dashes)
            img = cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

        return img

    def draw_line_as_points(self, image, point1, point2, image_height, color=(255, 0, 255), thickness=20):
        """Draw a line as a series of points on the image."""
        x1, y1 = point1
        x2, y2 = point2
        
        # Generate a series of points between point1 and point2
        num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to generate
        x_points = np.linspace(x1, x2, num=int(num_points/30), dtype=int)
        y_points = np.linspace(y1, y2, num=int(num_points/30), dtype=int)
        
        for x, y in zip(x_points, y_points):
            #cv2.circle(image, (x, y), thickness, color, -1)
            self.trajectory.append([x,y])
        
    def intersection_points(self, point1, slopes, point2, k):
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

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def line_circle_intersection(self, A, B, center, radius):
        """Calculate the intersection points of a line segment AB with a circle centered at center with radius."""
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

    def perpendicular_intersection(self, waypoint, closest_points):
        x1, y1 = waypoint
        x2, y2 = closest_points
        
        x = (self.k1 * x1 - y1 - self.k2 * x2 + y2) / (self.k1 - self.k2)
        y = self.k1 * (x - x1) + y1
        
        return (x, y)

    def determine_end_point(self, waypoint, closest_points, closest_rc, intersections):
        # Step 1: Calculate intersection point
        intersection = self.perpendicular_intersection(waypoint, closest_points[0])
        
        # 如果只有一个最近点，则只计算与该点的距离
        if len(closest_points) == 1:
            distance_to_closest_points1 = self.distance(intersection, closest_points[0])
            # 根据实际需求，判断是否返回 intersection 或做其他处理
            if distance_to_closest_points1 > closest_rc[0]:
                return intersection
            else:
                # 遍历 intersections 寻找最合适的交点
                min_distance = float('inf')
                closest_intersection = None
                for point in intersections:
                    dist_waypoint_to_point = self.distance(waypoint, point)
                    dist_point_to_closest_points = self.distance(point, closest_points[0])
                    if dist_waypoint_to_point < min_distance and dist_point_to_closest_points > closest_rc[0]:
                        min_distance = dist_waypoint_to_point
                        closest_intersection = point
                return closest_intersection

        # 如果有两个或以上最近点，则计算两个距离
        elif len(closest_points) >= 2:
            distance_to_closest_points1 = self.distance(intersection, closest_points[0])
            distance_to_closest_points2 = self.distance(intersection, closest_points[1])
            if distance_to_closest_points1 > closest_rc[0] and distance_to_closest_points2 > closest_rc[1]:
                return intersection
            else:
                # 遍历 intersections 寻找最合适的交点
                min_distance = float('inf')
                closest_intersection = None
                for point in intersections:
                    dist_waypoint_to_point = self.distance(waypoint, point)
                    dist_point_to_closest_points = self.distance(point, closest_points[0])
                    if dist_waypoint_to_point < min_distance and dist_point_to_closest_points > closest_rc[0]:
                        min_distance = dist_waypoint_to_point
                        closest_intersection = point
                return closest_intersection
        
    def check_line_through_circles(self, A, B, circles, func, *func_args):
        for center, radius in circles:
            intersection_points = self.line_circle_intersection(A, B, center, radius)
            if len(intersection_points) == 2:
                return func(*func_args)
        return B  # One or zero intersections, return B point

    # draw same length line
    def draw_clipped_lines(self, image, start_point, end_points, clip_length, color, thickness):
        new_end_points = []

        for end_point in end_points:
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = np.sqrt(dx**2 + dy**2)
            scale = clip_length / length if length > 0 else 0
            new_end_x = int(start_point[0] + scale * dx)
            new_end_y = int(start_point[1] + scale * dy)
            new_end_point = (new_end_x, new_end_y)
            cv2.line(image, start_point, new_end_point, color, thickness)
            new_end_points.append(new_end_point)

        return new_end_points

    # find next closest points
    def closest_two_points(self, origin, points, R_c, intersections):
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
    def end_point(self, x, y, slopes):
        delta_x = 100 # tangents length
        end_points = []
        for slope in slopes:
            delta_y = slope * delta_x
            end_point = (int(x + delta_x), int(y + delta_y))
            end_points.append(end_point)
        
        return end_points

    # find tangent
    def find_m(self, x1, y1, h, k, r):
        # Calculate the distance from the external point to the center of the circle
        d = np.sqrt((x1 - h)**2 + (y1 - k)**2)
        
        # Calculate the angle of the line connecting the point to the center
        theta = np.arctan2(y1 - k, x1 - h)
        
        # Calculate the angles of the tangent lines
        alpha1 = theta + np.arcsin(r / d)
        alpha2 = theta - np.arcsin(r / d)
        
        # Calculate the slopes of the tangent lines
        m1 = np.tan(alpha1)
        m2 = np.tan(alpha2)

        result = [m1, m2]
        
        return result

    # get rid of some obstacles
    def pre_process(self, SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, alpha):
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
    
    def adjust_waypoint(wself, waypoint_current, waypoint_next, circles):
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
                    t_intersect = max(valid_ts)
                    new_waypoint_next = (P0x + t_intersect * dx, P0y + t_intersect * dy)
                    # 更新 waypoint_next 为交点，并退出循环（若希望处理所有圆，可根据需要修改）
                    return new_waypoint_next
        # 如果 waypoint_next 未落入任何圆中，则原样返回
        return waypoint_next

    def compute_intersection_of_line_with_line_t(self, m, waypoint, p1, p2):
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

    def select_farthest_pair(self, points):
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

    def draw_transparent_triangle(self, image, vertices, color, alpha):
        """
        在 image 上绘制填充三角形，vertices 为三角形顶点列表（格式 [(x,y), ...]），
        color 为 BGR 颜色，alpha 为透明度（0～1）。
        """
        overlay = image.copy()
        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
    def run(self):

        # draw the path
        image = self.image
        image_height, image_width = image.shape[:2]
        print(image_height, image_width)

        # Convert image from BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold for color
        _, threshold_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Detect color
        contours_white, _ = cv2.findContours(threshold_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
        # Center of area
        centers_radius_white = []

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
                cv2.rectangle(image, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 0, 255), 2)
                
                # 计算子区域中心（原始坐标系：左上角为原点）
                center_x = x_new + w_new / 2
                center_y_original = y_new + h_new / 2
                # 如果需要以左下角为原点，则转换 y 坐标
                center_y = image_height - center_y_original
                
                radius = int(self.safety_radius * np.sqrt(w_new**2 + h_new**2))

                #cv2.circle(image, (int(center_x), image_height - int(center_y)), radius, (0, 255, 0), 3)
                
                centers_radius_white.append(((center_x, center_y), radius, (x_new, y_new, w_new, h_new)))
                all_bboxes.append((x_new, y_new, w_new, h_new))

        SP_x, SP_y = self.start_point[0], self.start_point[1]
        EP_x, EP_y = self.end_point[0], self.end_point[1]

        p_cx = [point[0][0] for point in centers_radius_white]
        p_cy = [point[0][1] for point in centers_radius_white]
        R_c = [point[1] for point in centers_radius_white]

        # remove far obstacles
        intersections, intercepts, p_c, R_c, m0, mp = self.pre_process(SP_x, SP_y, EP_x, EP_y, p_cx, p_cy, R_c, self.alpha)

        # pack intersection coordinates and related data into a list of tuples
        packed_data = list(zip(intersections, intercepts, p_c, R_c))

        # sort the tuple list based on the abscissa of the intersection point
        sorted_packed_data = sorted(packed_data, key=lambda x: x[0][0])

        # unpack the data and extract the sorted results
        sorted_intersections = [item[0] for item in sorted_packed_data]
        sorted_intercepts = [item[1] for item in sorted_packed_data]
        sorted_p_c = [item[2] for item in sorted_packed_data]
        sorted_R_c = [item[3] for item in sorted_packed_data]
        
        # selected obstacles' radius including hyperparameter
        updated_R_c = [r + d for r, d in zip(sorted_R_c, self.deltas)]
        for (x, y), radius in zip(sorted_p_c, sorted_R_c):
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 3)
        # circle with hyperparameter
        for (x, y), radius in zip(sorted_p_c, updated_R_c):
            self.draw_dashed_circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 3, 50)

        # predefined
        self.k1 = (EP_y - SP_y) / (EP_x - SP_x)
        self.k2 = -1 / self.k1
        waypoint0 = [SP_x, SP_y]
        n = len(updated_R_c)

        # initialize
        updated_points = sorted_p_c
        updated_rc = updated_R_c
        updated_intersections = sorted_intersections
        waypoints = []
        waypoints.append(waypoint0)
        total_length = 0
        nodes = []

        for i in range(n):
            waypoint_current = waypoints[-1]  # 获取当前的waypoint
        
            if i == 0:
                circles = list(zip(sorted_p_c, updated_R_c))
            else:
                circles = list(zip(updated_points, updated_rc))

            closest_points, closest_rc, closest_intersections, updated_points, updated_rc, updated_intersections = self.closest_two_points(
                waypoint_current, updated_points, updated_rc, updated_intersections)

            if len(closest_points) == 1:
                result = self.find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
                if result is None:
                    print("r/d exceed")
                    break
            else:
                result1 = self.find_m(waypoint_current[0], waypoint_current[1], closest_points[0][0], closest_points[0][1], closest_rc[0])
                result2 = self.find_m(waypoint_current[0], waypoint_current[1], closest_points[1][0], closest_points[1][1], closest_rc[1])
                if result1 is None or result2 is None:
                    print("r/d exceed")
                    break
                result = result1 + result2

            intersections = self.intersection_points(waypoint_current, result, closest_points[0], self.k2)
            waypoint_next = self.check_line_through_circles(waypoint_current, [EP_x, EP_y], circles, self.determine_end_point, waypoint_current, closest_points, closest_rc, intersections)
            waypoint_next = self.adjust_waypoint(waypoint_current, waypoint_next, circles)
            
            # draw path
            if waypoint_next is None:
                break
            self.draw_line_as_points(image, waypoint_current, waypoint_next, image_height)

            waypoints.append(waypoint_next)  # 添加新的waypoint到列表

            if len(intersections) == 2:
                waypoint_final = [EP_x, EP_y]
                self.draw_line_as_points(image, waypoint_next, waypoint_final, image_height)
                waypoints.append(waypoint_final)  # 添加最终的waypoint
                break  # 如果达到了终点，可以退出循环

            if waypoint_next == [EP_x, EP_y]:
                break

            # 对于 results 中每个斜率，计算直线 L（过 waypoint_current，斜率 m）与直线 t 的交点
            intersection_points_x = []
            for m in result:
                pt = self.compute_intersection_of_line_with_line_t(m, waypoint_current, waypoint_next, closest_points[0])
                if pt is not None:
                    intersection_points_x.append(pt)
            
            # 选出这四个交点（或其中有效的交点）中距离最远的两个
            far_pair = self.select_farthest_pair(intersection_points_x)
            if far_pair is None:
                print(f"第 {i+1} 次循环：无法选出最远的两个交点，跳过该次计算。")
                continue
            
            # 用 far_pair 中的两个交点与 waypoint_current 构成三角形
            triangle_vertices = [(waypoint_current[0], image_height - waypoint_current[1]), 
                                (far_pair[0][0], image_height - far_pair[0][1]), 
                                (far_pair[1][0], image_height - far_pair[1][1])]
            
            # 在 image_test 上用蓝色透明填充该三角区域，蓝色 BGR=(255, 0, 0)，透明度 0.5

            #draw_transparent_triangle(image, triangle_vertices, (255, 255, 0), 0.3)

        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        total_length = np.sum(distances)

        print(f"total length: {total_length:.2f}")

        '''for idx, (x, y) in enumerate(sorted_p_c):
            cv2.putText(
                image,              # 图像
                str(idx+1),            # 文本内容
                (int(x)-15, image_height-int(y)+5),        # 坐标转换为整数
                cv2.FONT_HERSHEY_SIMPLEX,# 字体
                2,                       # 字体大小
                (0, 0, 255),             # 颜色 (BGR)
                4                        # 线条粗细
            )'''

        cv2.circle(image,(self.start_point[0], self.start_point[1]),60,(0,0,255), -1,)
        cv2.circle(image,(self.end_point[0], self.end_point[1]),60,(0,255,255), -1,)
        
        for node in self.trajectory:
            print(node)
            cv2.circle(image,(int(node[0]), int(node[1])),15,(255,0,255), -1,)

        cv2.imshow('geo', image)
        name = "media/result{}.png".format(time.time())
        cv2.imwrite(name, image) 
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Code running time: {elapsed_time:.2f} s")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.trajectory
        #cv2.imwrite('/Users/yanda/Downloads/Plan Planning/test/result3_2.png', image) 

if __name__ == "__main__":
    image = "/Users/yanda/Downloads/Plan Planning/test/4.png"
    image = cv2.imread(image)
    start_point = (517,415)
    end_point = (1501,1273)
    algorithm = geo_algorithm(image, start_point, end_point)
    points = algorithm.run()
    
    print(points)