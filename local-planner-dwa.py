#!/usr/bin/env python3

# Local Planner

# A custom Dynamic Window Approach (DWA) implementation for use with Husky.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance.

import rospy
import math

import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


class Config():

    def __init__(self):
        
        # Configuration parameters
        self.max_speed = 0.5  # [m/s]
        self.min_speed = 0.0 # [m/s]
        self.max_yawrate = 0.5  # [rad/s]
        self.max_accel = 0.5  # [m/ss]
        self.max_dyawrate = 1.2 #3.2  # [rad/ss]
        
        self.v_reso = 0.15  # [m/s]
        self.yawrate_reso = 0.05 #0.05  # [rad/s]
        
        # trajectory prediction time horizon
        self.dt = 0.05  # [s]
        self.dt_dwa = 0.2 # [s]       
        self.predict_time = 3.5  # [s]
        
        # Gains/weights for each cost term
        self.to_goal_cost_gain = 10.0   # lower -> detour more from goal/waypoint
        self.speed_cost_gain = 1        # lower -> faster motion
        self.obs_cost_gain = 30         # lower -> fearless, i.e., move too close to the obstacles
        
        self.robot_radius = 0.3 # [m] # Affects the obstacle avoidance and goal reaching thresholds
        self.laser_range_max = 6.0  # [m] Clamp infinite range of laserscan if needed
        self.goal_received = False  # Flag to check if goal has been published

        # Initial robot state parameters
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.w_z = 0.0
        self.goalX = None 
        self.goalY = None 
        self.th = 0.0

        self.r = rospy.Rate(20)

        # Planner output initialization
        self.min_u = []


    # Callback for Odometry
    def assignOdomCoords(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
        
        self.th = theta

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z 

    # Callback for waypoint goal
    def goal_waypoint_callback(self, data):

        self.goal_received = True  # Mark that a valid goal has been received

        # If the goal is receiving w.r.t. robot's based link in polar coordinates
        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        
        # Goal wrt robot frame        
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)

        # If goal is published as x, y coordinates wrt odom frame, uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y


class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in the field of view
        self.obst = set()
        self.collision_status = False

    def assignObs(self, msg, config):
        """
        LaserScan callback that converts scan data into global (x, y) obstacle points
        and stores them in self.obst.
        """
        self.obst = set()  # Reset visible obstacle set

        angle = msg.angle_min

        for i, distance in enumerate(msg.ranges):
            # Skip invalid, NaN, or negative values
            if np.isnan(distance) or np.isinf(distance):
                angle += msg.angle_increment
                continue

            # Clamp max usable range
            if distance < msg.range_min or distance > min(msg.range_max, config.laser_range_max):
                angle += msg.angle_increment
                continue

            # Transform to global coordinates (odom frame)
            objTheta = angle + config.th  # global angle
            obsX = config.x + distance * math.cos(objTheta)
            obsY = config.y + distance * math.sin(objTheta)

            self.obst.add((obsX, obsY))

            angle += msg.angle_increment


# Motion model (unicycle) to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):

    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    x[3] = u[0]
    x[4] = u[1]

    return x


# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt_dwa,
          x[3] + config.max_accel * config.dt_dwa,
          x[4] - config.max_dyawrate * config.dt_dwa,
          x[4] + config.max_dyawrate * config.dt_dwa]

    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt # next sample

    return traj

def calc_to_goal_cost_vec(traj_arr, config):
    
    dx = config.goalX - traj_arr[:,-1, 0]
    dy = config.goalY - traj_arr[:,-1, 1]

    goal_cost = np.sqrt(dx**2 + dy**2)

    return goal_cost

def calc_speed_cost_vec(traj_arr, config):
    
    speed_cost = config.max_speed - traj_arr[:,-1, 3]
    return speed_cost


def calc_final_input(x, u, dw, config, ob):

    xinit = x[:]

    config.min_u = u
    config.min_u[0] = 0.0

    traj_arr = []
    commands_arr = []

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + config.v_reso/2, config.v_reso):
        for w in np.arange(dw[2], dw[3] + config.yawrate_reso/2, config.yawrate_reso):            
            traj = calc_trajectory(xinit, v, w, config)
            traj_arr.append(traj)
            commands_arr.append([v,w])

    traj_arr = np.array(traj_arr)

    to_goal_costs = config.to_goal_cost_gain * calc_to_goal_cost_vec(traj_arr, config)
    speed_costs = config.speed_cost_gain * calc_speed_cost_vec(traj_arr, config)
    obs_costs = config.obs_cost_gain * calc_obstacle_cost_vec(traj_arr, ob, config)

    final_costs = to_goal_costs + obs_costs + speed_costs
    min_index = np.argmin(final_costs)
    config.min_u = commands_arr[min_index]

    # print("min_u = %.2f %.2f"% (config.min_u[0], config.min_u[1]))#, "Goal cost = %.2f"% to_goal_cost, "min cost = %.2f"% min_cost)
    # print(f"Obs costs: min={np.min(obs_costs):.2f}, max={np.max(obs_costs):.2f}")

    return config.min_u
    

def calc_obstacle_cost_vec(traj_arr, ob, config):
    """
    Vectorized obstacle cost: traj_arr shape = [N, T, 5]
    Returns: array of shape [N] with obstacle cost for each trajectory
    """
    costs = np.zeros(len(traj_arr))
    robot_radius = config.robot_radius

    if len(ob) == 0:
        print("Warning: Obstacle set is empty.")
        return costs

    for idx, traj in enumerate(traj_arr):
        minr = float("inf")
        collision = False

        for ox, oy in ob:
            dx = traj[:, 0] - ox
            dy = traj[:, 1] - oy
            dists = np.sqrt(dx ** 2 + dy ** 2)

            if np.any(dists <= robot_radius):
                collision = True
                break  # stop checking this trajectory

            minr = min(minr, np.min(dists))

        COLLISION_PENALTY = 1e6  # instead of inf

        if collision:
            costs[idx] = COLLISION_PENALTY
        elif minr < float("inf"):
            costs[idx] = 1.0 / minr
        else:
            costs[idx] = 0.0

    return costs


# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - 0)

    cost = math.sqrt(dx**2 + dy**2)
    return cost


# Begin DWA calculations
def dwa_control(x, u, config, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, dw, config, ob)

    return u


# Determine whether the robot has reached its goal
def atGoal(config, x):
    # check at goal
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False


def main():
    print(__file__ + " start!!")
    
    config = Config()
    obs = Obstacles()

    subOdom = rospy.Subscriber("/go2_0/lio/odometry", Odometry, config.assignOdomCoords)
    subLaser = rospy.Subscriber("/go2_0/scan", LaserScan, obs.assignObs, config)
    subWaypoint = rospy.Subscriber('/waypoint', Twist, config.goal_waypoint_callback)

    pubVel = rospy.Publisher("/velocity_cmd", Twist, queue_size=1)

    speed = Twist()
    
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])

    # initial linear and angular velocities
    u = np.array([0.0, 0.0])

    # runs until terminated externally
    while not rospy.is_shutdown():

        if not config.goal_received:
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        # Pursuing but not reached the goal
        elif (atGoal(config,x) == False): 

                u = dwa_control(x, u, config, obs.obst)

                x[0] = config.x
                x[1] = config.y
                x[2] = config.th
                x[3] = u[0]
                x[4] = u[1]

                speed.linear.x = x[3]
                speed.angular.z = x[4]

        # If at goal then stay there until new goal published
        else:
            print("Goal reached!")
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        pubVel.publish(speed)
        config.r.sleep()

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('local_planner_dwa')
    main()