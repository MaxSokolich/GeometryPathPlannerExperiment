import numpy as np
import cv2

class control_algorithm:
    def __init__(self):
        self.node = 0
        self.arrival_threshold = 10
        
    def run_static(self, frame, robot_list):
        
        pts = np.array(robot_list[-1].trajectory, np.int32)
        cv2.polylines(frame, [pts], False, (0, 0, 255), 4)
        

        #logic for arrival condition
        if self.node == len(robot_list[-1].trajectory):
            #weve arrived
           
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = 0
            gamma = np.pi/2   #disregard
            freq = 0    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard


        #closed loop algorithm 
        else:
            #define target coordinate
            targetx = robot_list[-1].trajectory[self.node][0]
            targety = robot_list[-1].trajectory[self.node][1]

            #define robots current position
            robotx = robot_list[-1].position_list[-1][0]
            roboty = robot_list[-1].position_list[-1][1]
            
            #calculate error between node and robot
            direction_vec = [targetx - robotx, targety - roboty]
            error = np.sqrt(direction_vec[0] ** 2 + direction_vec[1] ** 2)
            if error < self.arrival_threshold:
                self.node += 1
            
            cv2.arrowedLine(
                    frame,
                    (int(robotx), int(roboty)),
                    (int(targetx), int(targety)),
                    [100, 100, 100],
                    3,
                )
            
                
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = np.arctan2(-direction_vec[1], direction_vec[0])  - np.pi/2
            gamma = np.pi/2   #disregard
            freq = 5    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard
        
        
        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq













    def run_dynamic(self, frame, current_robot_position, trajectory, end):
        #print("traj length = ", len(trajectory))
        
        
        
        
        
        if len(trajectory) > 1:
            
            #find the closest node to the robots current position
            #current_array = np.array(current_robot_position)
            #trajectory_array = np.array(trajectory)
            #distances = np.linalg.norm(trajectory_array - current_array, axis=1)
            #closest_index = np.argmin(distances)
            
            
            #if trajectory_array[closest_index] < self.arrival_threshold:
            #    closest_index += 1


            targetx = trajectory[1][0]
            targety = trajectory[1][1]
            robotx = current_robot_position[0]
            roboty = current_robot_position[1]

            
            direction_vec = [targetx - robotx, targety - roboty]
            cv2.arrowedLine(
                        frame,
                        (int(robotx), int(roboty)),
                        (int(targetx), int(targety)),
                        [100, 100, 100],
                        3,
                    )
            cv2.putText(frame,"trajectory length: {}".format(len(trajectory)),
                        (int(2448  / 80),int(2048 / 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, 
                        thickness=4,
                        color = (255, 255, 255))


            #print("actuating", trajectory)
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = np.arctan2(-direction_vec[1], direction_vec[0])  - np.pi/2
            gamma = np.pi/2   #disregard
            freq = 10    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard
        else:
            targetx = end[0]
            targety = end[1]
            robotx = current_robot_position[0]
            roboty = current_robot_position[1]
            print(targetx, robotx)
            direction_vec = [targetx - robotx, targety - roboty]
            error = np.sqrt(direction_vec[0] ** 2 + direction_vec[1] ** 2)
            if error < self.arrival_threshold:
                Bx = 0 #disregard
                By = 0 #disregard
                Bz = 0 #disregard
                alpha = 0
                gamma = np.pi/2   #disregard
                freq = 0    #CHANGE THIS EACH FRAME
                psi = np.pi/2      #disregard
                gradient = 0 # #disregard
                acoustic_freq = 0  #disregard
            else:
                Bx = 0 #disregard
                By = 0 #disregard
                Bz = 0 #disregard
                alpha = np.arctan2(-direction_vec[1], direction_vec[0])  - np.pi/2
                gamma = np.pi/2   #disregard
                freq = 10    #CHANGE THIS EACH FRAME
                psi = np.pi/2      #disregard
                gradient = 0 # #disregard
                acoustic_freq = 0  #disregard

        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq