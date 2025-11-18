#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import yaml
from enum import Enum, auto
import os
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport
from std_msgs.msg import Bool


class TurnState(Enum):
    FORWARD = auto()
    LEFT = auto()
    RIGHT = auto()

class LKASNode(Node):
    def __init__(self, config_path='/home/rubis/autoware/src/lkas_controller/lkas_controller/cfg/param.yaml'):
        super().__init__('lkas_node')
        self.bridge = CvBridge()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.filter_thr_dict = config['filter_thr_dict']
        self.birdeye_warp_param = config['birdeye_warp_param']
        self.velocity = float(config['velocity'])
        self.steer_sensitivity = float(config['steer_sensitivity'])
        self.image_width = config.get('image_width', 800)
        self.image_height = config.get('image_height', 600)
        self.turn_state = TurnState.FORWARD

        self.last_cmd = AckermannControlCommand()
        self.last_cmd.lateral.steering_tire_angle = 0.0
        self.last_cmd.lateral.steering_tire_rotation_rate = 0.0
        self.last_cmd.longitudinal.speed = 0.0
        self.last_cmd.longitudinal.acceleration = 0.0
        self.last_cmd.longitudinal.jerk = 0.0

        self.declare_parameter('publish_rate_hz', 7.5)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.timer_period = 1.0 / self.publish_rate_hz
        

        # Steering angle limit (in radians)
        # self.steering_limit = 1.0
        self.steering_limit = 0.4

        # For tracking the current vehicle lateral speed
        self.current_longitudinal_speed = 0.0

        # Subscriptions
        self.subscription = self.create_subscription(
            Image,
            '/sensing/camera/traffic_light/image_raw',
            self.image_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.velocity_sub = self.create_subscription(
            VelocityReport,
            '/vehicle/status/velocity_status',
            self.velocity_report_callback,
            qos_profile=qos_profile_sensor_data
        )

        self.timer = self.create_timer(self.timer_period, self.timer_publish)
        # Publishers
        self.twist_pub = self.create_publisher(
            TwistStamped, '/twist_cmd_lkas', 10)
        self.control_cmd_pub = self.create_publisher(
            AckermannControlCommand, '/control/command/control_cmd', 10)
        self.pub_original = self.create_publisher(
            Image, '/lkas/debug/original_image', qos_profile_sensor_data)
        self.pub_birdeye = self.create_publisher(
            Image, '/lkas/debug/birdeye_image', qos_profile_sensor_data)
        self.pub_filtered = self.create_publisher(
            Image, '/lkas/debug/filtered_image', qos_profile_sensor_data)
        self.pub_lane_viz = self.create_publisher(
            Image, '/lkas/lane_viz/image', qos_profile_sensor_data)
        self.pub_birdeye_lane_arrow = self.create_publisher(
            Image, '/lkas/debug/lane_and_decision_birdeye', qos_profile_sensor_data)


        self.get_logger().info('LKASNode initialized!!')

    def velocity_report_callback(self, msg):
        # Store the latest lateral speed for access in image_callback
        self.current_longitudinal_speed = msg.longitudinal_velocity

    def clip_steering(self, angle):
        """Clamps the steering angle to the configured steering limit (±1.0 radians)."""
        return max(min(angle, self.steering_limit), -self.steering_limit)

    def timer_publish(self):
        self.control_cmd_pub.publish(self.last_cmd)


    def image_callback(self, msg): 
        try:
            # Original image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            orig_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            orig_msg.header = msg.header
            self.pub_original.publish(orig_msg)

            # Bird's eye view (color)
            birdeye_image = birdeye_warp(cv_image, self.birdeye_warp_param)
            birdeye_msg = self.bridge.cv2_to_imgmsg(birdeye_image, encoding='bgr8')
            birdeye_msg.header = msg.header
            self.pub_birdeye.publish(birdeye_msg)

            # Filtering output (HSV + x-grad)
            filtered_birdeye = hsv_xgrad_filter(birdeye_image, self.filter_thr_dict)
            filtered_birdeye_vis = (filtered_birdeye * 255).astype(np.uint8)
            filtered_birdeye_vis = cv2.cvtColor(filtered_birdeye_vis, cv2.COLOR_GRAY2BGR)
            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_birdeye_vis, encoding='bgr8')
            filtered_msg.header = msg.header
            self.pub_filtered.publish(filtered_msg)
            
            # Lane viz and sliding window
            sliding_window_img, ls, rs, lv, rv, lx, rx, left_pts, right_pts = calculate_sliding_window_and_points(
                filtered_birdeye)
            lane_viz = sliding_window_img.astype(np.uint8)
            lane_viz_msg = self.bridge.cv2_to_imgmsg(lane_viz, encoding='bgr8')
            lane_viz_msg.header = msg.header
            self.pub_lane_viz.publish(lane_viz_msg)

            # Vehicle command
            velocity, angle = self.calculate_velocity_and_angle(ls, rs, lv, rv, lx, rx)
            angle = self.clip_steering(angle)  # Clamp

            cmd_msg = TwistStamped()
            cmd_msg.header = msg.header
            cmd_msg.twist.linear.x = float(velocity)
            cmd_msg.twist.angular.z = float(angle)
            self.twist_pub.publish(cmd_msg)

            # Conditional acceleration based on lateral speed
            
            if self.current_longitudinal_speed < velocity:
                acceleration_setpoint = 0.5
            else:
                acceleration_setpoint = 0.0

            ackermann_cmd = AckermannControlCommand()
            ackermann_cmd.lateral.steering_tire_angle = float(angle)
            ackermann_cmd.lateral.steering_tire_rotation_rate = 0.0
            ackermann_cmd.longitudinal.speed = float(velocity)
            ackermann_cmd.longitudinal.acceleration = acceleration_setpoint
            ackermann_cmd.longitudinal.jerk = 0.0
            self.last_cmd = ackermann_cmd


            # Color birdeye with detected lanes and arrow
            birdeye_lane_arrow = birdeye_image.copy()
            if len(left_pts) > 1:
                cv2.polylines(birdeye_lane_arrow, [np.array(left_pts, np.int32)], False, (0, 255, 0), 5)
            if len(right_pts) > 1:
                cv2.polylines(birdeye_lane_arrow, [np.array(right_pts, np.int32)], False, (0, 255, 0), 5)
            center = (self.image_width // 2, self.image_height - 10)
            arrow_length = 80
            arrow_theta = float(-angle)
            tip_x = int(center[0] + arrow_length * math.sin(arrow_theta))
            tip_y = int(center[1] - arrow_length * math.cos(arrow_theta))
            cv2.arrowedLine(birdeye_lane_arrow, center, (tip_x, tip_y), (0, 0, 255), 7, tipLength=0.3)
            final_msg = self.bridge.cv2_to_imgmsg(birdeye_lane_arrow, encoding='bgr8')
            final_msg.header = msg.header
            self.pub_birdeye_lane_arrow.publish(final_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

    def calculate_velocity_and_angle(self, ls, rs, lv, rv, lx, rx):
        left_ground_truth = 7.0
        right_ground_truth = -7.0
        velocity = float(self.velocity)
        target_angle = 0.0
        if self.turn_state == TurnState.LEFT:
            if (lv and ls > 0):
                self.turn_state = TurnState.FORWARD
            if rv:
                target_angle = (float(rx) - self.image_height / 6) / 10
            else:
                target_angle = left_ground_truth
        elif self.turn_state == TurnState.RIGHT:
            if (rv and rs < 0):
                self.turn_state = TurnState.FORWARD
            if lv:
                target_angle = (self.image_height / 6 - float(lx)) / 10
            else:
                target_angle = right_ground_truth
        else:
            if lv and rv:
                if ls < right_ground_truth and rs < right_ground_truth * 3:
                    self.turn_state = TurnState.LEFT
                if rs > left_ground_truth and ls > left_ground_truth * 3:
                    self.turn_state = TurnState.RIGHT
                ls_f = float(ls)
                rs_f = float(rs)
                angle_gap = abs(ls_f - rs_f)  # deg

                if angle_gap > 20.0:
                    chosen = ls_f if abs(ls_f) < abs(rs_f) else rs_f
                    target_angle = -chosen / 5.0
                else:
                    avg_angle = 0.5 * (ls_f + rs_f)
                    target_angle = -avg_angle / 5.0
            elif rv:
                target_angle = -float(rs)/5
            elif lv:
                target_angle = -float(ls)/5
        target_angle = float(target_angle) * float(self.steer_sensitivity)
        return velocity, target_angle

def find_firstfit(binary_array, direction, lower_threshold=20):
    start, end = (binary_array.shape[0] - 1, -1) if direction == -1 else (0, binary_array.shape[0])
    for i in range(start, end, direction):
        if binary_array[i] > lower_threshold:
            return i
    return -1

def hsv_xgrad_filter(img, filter_thr_dict):
    h_thresh = filter_thr_dict['hue_thr']
    s_thresh = filter_thr_dict['saturation_thr']
    v_thresh = filter_thr_dict['value_thr']
    sx_thresh = filter_thr_dict['x_grad_thr']

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    max_sx = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sx) if max_sx else np.zeros_like(abs_sobelx, dtype=np.uint8)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(h_binary == 1) & (s_binary == 1) & (v_binary == 1) & (sxbinary == 1)] = 1
    return combined_binary

def birdeye_warp(img, birdeye_warp_param):
    x_size, y_size = img.shape[1], img.shape[0]
    x_mid = x_size / 2
    top = birdeye_warp_param['top']
    bottom = birdeye_warp_param['bottom']
    top_margin = birdeye_warp_param['top_width']
    bottom_margin = birdeye_warp_param['bottom_width']
    birdeye_margin = birdeye_warp_param['birdeye_width']
    src = np.float32([
        [x_mid + top_margin, top],
        [x_mid + bottom_margin, bottom],
        [x_mid - bottom_margin, bottom],
        [x_mid - top_margin, top]
    ])
    dst = np.float32([
        [x_mid + birdeye_margin, 0],
        [x_mid + birdeye_margin, y_size],
        [x_mid - birdeye_margin, y_size],
        [x_mid - birdeye_margin, 0]
    ])
    trans_matrix = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    birdeye_warped = cv2.warpPerspective(img, trans_matrix, img_size)
    return birdeye_warped

def calculate_sliding_window_and_points(filtered_img):
    H, W = filtered_img.shape[:2]
    windows_num = 24
    window_width = 40
    x_margin = W/20
    consecutive_y_margin = 100
    noise_threshold = W/10
    window_threshold = 4
    slope_threshold = 60

    window_height = H // windows_num
    consecutive_y_idx = max(1, consecutive_y_margin // max(1, window_height))
    x_mid = W // 2

    out_img = (np.dstack([filtered_img]*3) * 255).astype(np.uint8)
    lw_arr, rw_arr = [], []
    left_pts, right_pts = [], []

    for window_idx in range(windows_num):
        top = window_height * (windows_num - window_idx - 1)
        bottom = window_height * (windows_num - window_idx)
        y_mid = (top + bottom) // 2

        l_hist = np.sum(filtered_img[top:bottom, :x_mid], axis=0)
        leftx = find_firstfit(l_hist, -1, window_height // 2) 
        if x_margin < leftx < x_mid - x_margin*3:
            left_pts.append((leftx, y_mid))
            if not lw_arr:
                lw_arr.append((leftx, window_idx))
                cv2.rectangle(out_img, (leftx - window_width, bottom), (leftx, top), (0, 255, 0), 2)
            else:
                last_x, last_idx = lw_arr[-1]
                if (window_idx - last_idx) <= consecutive_y_idx and abs(last_x - leftx) < noise_threshold:
                    lw_arr.append((leftx, window_idx))
                    cv2.rectangle(out_img, (leftx - window_width, bottom), (leftx, top), (255, 255, 0), 2)

        r_hist = np.sum(filtered_img[top:bottom, x_mid:], axis=0)
        rightx_local = find_firstfit(r_hist, +1, window_height // 2)
        right_width = W - x_mid
        if x_margin*3 < rightx_local < right_width - x_margin:
            x_abs = rightx_local + x_mid
            right_pts.append((x_abs, y_mid))
            if not rw_arr:
                rw_arr.append((x_abs, window_idx))
                cv2.rectangle(out_img, (x_abs, bottom), (x_abs + window_width, top), (0, 255, 0), 2)
            else:
                last_x, last_idx = rw_arr[-1]
                if (window_idx - last_idx) <= consecutive_y_idx and abs(last_x - x_abs) < noise_threshold:
                    rw_arr.append((x_abs, window_idx))
                    cv2.rectangle(out_img, (x_abs, bottom), (x_abs + window_width, top), (255, 255, 0), 2)

    def fit_angle(arr):
        if len(arr) < window_threshold:
            return None
        xs = [float(x) for (x, widx) in arr]
        ys = [ (widx + 0.5) * window_height for (_, widx) in arr ]  
        m = np.polyfit(xs, ys, 1)[0]
        a = math.degrees(math.atan(m))  
        return (90 - a) if a > 0 else (-90 - a)  

    left_angle = fit_angle(lw_arr)
    right_angle = fit_angle(rw_arr)

    isLeftValid = left_angle is not None
    isRightValid = right_angle is not None
    if isLeftValid and left_angle < -slope_threshold:
        isLeftValid = False
    if isRightValid and right_angle > slope_threshold:
        isRightValid = False

    first_left_x_margin = 0
    first_right_x_margin = 0

    if isLeftValid and isRightValid:
        li = ri = 0
        while li < len(lw_arr) and ri < len(rw_arr) and lw_arr[li][1] != rw_arr[ri][1]:
            if lw_arr[li][1] > rw_arr[ri][1]:
                ri += 1
            else:
                li += 1
        if li < len(lw_arr) and ri < len(rw_arr):
            first_left_x_margin = lw_arr[li][0]            
            first_right_x_margin = W - rw_arr[ri][0]       
        else:
            isLeftValid = False
            isRightValid = False
    elif isLeftValid:
        first_left_x_margin = lw_arr[0][0]
    elif isRightValid:
        first_right_x_margin = W - rw_arr[0][0]

    return (out_img, left_angle or -1, right_angle or -1,
            isLeftValid, isRightValid,
            first_left_x_margin, first_right_x_margin,
            left_pts, right_pts)


def main(args=None):
    rclpy.init(args=args)
    node = LKASNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
