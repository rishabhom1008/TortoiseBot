#!/home/tortoisebot/vision_venv/bin/python

import math
import time

import cv2
import mediapipe as mp
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        # =========================================================
        # ROBOT CONTROL PARAMETERS
        # =========================================================
        self.max_linear_speed = 0.10       # m/s
        self.max_angular_speed = 0.7      # rad/s

        # Safety watchdog timeout.
        self.command_timeout = 0.50        # seconds

        # Gesture must be detected continuously for this many frames.
        self.required_stable_frames = 5

        # V-sign rotation:
        # =========================================================
        # IMU-CONTROLLED 90-DEGREE ROTATION
        # =========================================================
        self.rotation_angle = math.pi / 2       # 90 degrees

        # Proportional controller:
        # angular_speed = Kp * angle_error
        self.rotation_kp = 1.0                  # rad/s per rad

        # Minimum effective motor turning speed.
        self.min_turn_speed = 0.5              # rad/s

        # Acceptable final angle: 85° to 95°
        self.rotation_tolerance = math.radians(5.0)

        # Safety limits
        self.max_invalid_imu_readings = 10
        self.imu_timeout = 1.0                  # seconds
        self.rotation_timeout = 25.0             # seconds

        # Reject physically implausible yaw jumps during rotation.
        self.max_yaw_rate = math.radians(90.0)   # degrees/second
        self.yaw_jump_tolerance = math.radians(10.0)

        # Maximum number of target crossings/corrections
        self.max_rotation_corrections = 5

        # Physical /cmd_vel convention:
        # existing code uses negative angular.z for RIGHT.
        self.clockwise_cmd_sign = -1.0

        # This is the quaternion-yaw direction observed from
        # the IMU representation. We will verify this on the
        # first controlled test.
        self.clockwise_yaw_sign = -1.0

        # ---------------------------------------------------------
        # Pointing convention
        # ---------------------------------------------------------
        self.image_left_is_robot_left = True

        # =========================================================
        # STATE
        # =========================================================
        self.frame_count = 0
        self.error_count = 0

        self.last_valid_command_time = time.monotonic()

        # Gesture stability state
        self.last_gesture = None
        self.stable_gesture_count = 0

        # Debug state
        self.last_logged_gesture = None

        # Rotation state
        self.rotation_active = False
        self.rotation_end_time = 0.0
        self.correction_pause = 2.0
        self.rotation_pause_until = 0.0

        # One V-sign = one rotation
        self.v_sign_armed = True

        # =========================================================
        # IMU / ROTATION STATE
        # =========================================================
        self.current_yaw = None
        self.last_valid_imu_time = None
        self.invalid_imu_count = 0

        self.rotation_start_yaw = None
        self.rotation_target_yaw = None

        self.rotation_start_time = 0.0

        self.rotation_stable_count = 0
        self.rotation_correction_count = 0

        self.rotation_last_error_sign = 0

        # =========================================================
        # ROS
        # =========================================================
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )

        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Watchdog runs at 10 Hz.
        self.watchdog_timer = self.create_timer(
            0.10,
            self.watchdog_callback
        )

        # =========================================================
        # CAMERA / OPENCV
        # =========================================================
        self.br = CvBridge()

        # =========================================================
        # MEDIAPIPE
        # =========================================================
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60
        )

        self.get_logger().info(
            'TortoiseBot gesture-control node started.'
        )

    # =============================================================
    # CAMERA CALLBACK
    # =============================================================
    def listener_callback(self, msg):

        self.frame_count += 1
        try:
            # -----------------------------------------------------
            # Validate ROS image message
            # -----------------------------------------------------
            if msg is None:
                self.handle_error(
                    'Received empty ROS image message.'
                )
                return

            if msg.width <= 0 or msg.height <= 0:
                self.handle_error(
                    f'Invalid image dimensions: '
                    f'{msg.width}x{msg.height}'
                )
                return

            # -----------------------------------------------------
            # ROS Image -> OpenCV
            # -----------------------------------------------------
            cv_image = self.br.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            if cv_image is None or cv_image.size == 0:
                self.handle_error(
                    'Received empty OpenCV image.'
                )
                return

            # -----------------------------------------------------
            # Gesture recognition
            # -----------------------------------------------------
            gesture = self.process_image(cv_image)

            # -----------------------------------------------------
            # Debug logging
            #
            # Log only when the RAW gesture changes.
            # -----------------------------------------------------
            if gesture != self.last_logged_gesture:

                self.get_logger().info(
                    f'RAW GESTURE: {gesture}'
                )

                self.last_logged_gesture = gesture

            # -----------------------------------------------------
            # No valid gesture
            # -----------------------------------------------------
            if gesture is None:
                self.last_gesture = None
                self.stable_gesture_count = 0

                if not self.rotation_active:
                    self.v_sign_armed = True

                self.handle_no_gesture()
                return

            # -----------------------------------------------------
            # Gesture stability
            # -----------------------------------------------------
            stable_gesture = self.update_gesture_stability(
                gesture
            )

            if stable_gesture is None:
                return

            # -----------------------------------------------------
            # Debug: stable gesture
            # -----------------------------------------------------
            self.get_logger().info(
                f'STABLE GESTURE: {stable_gesture}'
            )

            # Execute the recognized stable gesture
            self.execute_gesture(stable_gesture)

        except CvBridgeError as exc:
            self.handle_error(
                f'CvBridge error: {exc}'
            )

        except Exception as exc:
            self.handle_error(
                f'Unexpected processing error: {exc}'
            )

    # =============================================================
    # GESTURE DETECTION
    # =============================================================
    def process_image(self, image):
        """
        Detect hand landmarks and classify the supported gestures.

        FIST       -> STOP
        PALM       -> FORWARD
        POINT_LEFT -> TURN LEFT
        POINT_RIGHT-> TURN RIGHT
        V_SIGN     -> 90-degree rotation
        """

        # ---------------------------------------------------------
        # OpenCV BGR -> RGB
        # ---------------------------------------------------------
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------------------
        # MediaPipe hand detection
        # ---------------------------------------------------------
        results = self.hands.process(rgb_image)

        # No hand detected
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # ---------------------------------------------------------
        # Determine finger states
        # ---------------------------------------------------------
        index_extended = self.is_finger_extended(
            hand_landmarks,
            pip_id=6,
            dip_id=7,
            tip_id=8
        )

        middle_extended = self.is_finger_extended(
            hand_landmarks,
            pip_id=10,
            dip_id=11,
            tip_id=12
        )

        ring_extended = self.is_finger_extended(
            hand_landmarks,
            pip_id=14,
            dip_id=15,
            tip_id=16
        )

        pinky_extended = self.is_finger_extended(
            hand_landmarks,
            pip_id=18,
            dip_id=19,
            tip_id=20
        )

        # ---------------------------------------------------------
        # Count extended fingers
        # ---------------------------------------------------------
        extended_count = sum([
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended
        ])

        # =========================================================
        # GESTURE CLASSIFICATION
        # =========================================================

        # ---------------------------------------------------------
        # Closed fist
        # ---------------------------------------------------------
        if extended_count == 0:
            return 'FIST'

        # ---------------------------------------------------------
        # Open palm
        # ---------------------------------------------------------
        if extended_count == 4:
            return 'PALM'

        # ---------------------------------------------------------
        # V sign
        # Index + middle extended
        # Ring + pinky folded
        # ---------------------------------------------------------
        if (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return 'V_SIGN'

        # ---------------------------------------------------------
        # Pointing
        # ---------------------------------------------------------
        if (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return self.detect_point_direction(
                hand_landmarks
            )

        # Unknown hand pose
        return None

    # =============================================================
    # FINGER EXTENSION
    # =============================================================
    def is_finger_extended(
        self,
        landmarks,
        pip_id,
        dip_id,
        tip_id
    ):
        """
        Determine whether a finger is extended.

        Uses:
            1. PIP joint angle
            2. Tip distance from MCP compared with DIP distance
        """
        pip = landmarks.landmark[pip_id]
        mcp = landmarks.landmark[pip_id - 1]
        dip = landmarks.landmark[dip_id]
        tip = landmarks.landmark[tip_id]

        angle = self.calculate_angle(
            mcp,
            pip,
            dip
        )

        tip_distance = math.sqrt(
            (tip.x - mcp.x) ** 2 +
            (tip.y - mcp.y) ** 2
        )

        dip_distance = math.sqrt(
            (dip.x - mcp.x) ** 2 +
            (dip.y - mcp.y) ** 2
        )

        return (
            angle > 150.0
            and tip_distance > dip_distance
        )

    # =============================================================
    # ANGLE CALCULATION
    # =============================================================
    def calculate_angle(self, a, b, c):
        """
        Calculate angle ABC in degrees.
        """

        ba = (
            a.x - b.x,
            a.y - b.y
        )

        bc = (
            c.x - b.x,
            c.y - b.y
        )

        dot_product = (
            ba[0] * bc[0]
            + ba[1] * bc[1]
        )

        magnitude_ba = math.sqrt(
            ba[0] ** 2 +
            ba[1] ** 2
        )

        magnitude_bc = math.sqrt(
            bc[0] ** 2 +
            bc[1] ** 2
        )

        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0

        cosine_angle = (
            dot_product /
            (magnitude_ba * magnitude_bc)
        )

        cosine_angle = max(
            -1.0,
            min(1.0, cosine_angle)
        )

        return math.degrees(
            math.acos(cosine_angle)
        )

    # =============================================================
    # POINT LEFT / RIGHT
    # =============================================================
    def detect_point_direction(self, landmarks):

        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]

        dx = index_tip.x - wrist.x

        # Ignore nearly vertical pointing.
        dead_zone = 0.10

        if abs(dx) < dead_zone:
            return None

        image_points_right = dx > 0

        if self.image_left_is_robot_left:
            if image_points_right:
                return 'POINT_RIGHT'

            return 'POINT_LEFT'
        else:
            if image_points_right:
                return 'POINT_LEFT'

            return 'POINT_RIGHT'

    # =============================================================
    # GESTURE STABILITY
    # =============================================================
    def update_gesture_stability(self, gesture):

        if gesture == self.last_gesture:
            self.stable_gesture_count += 1
        else:
            self.last_gesture = gesture
            self.stable_gesture_count = 1

        if (
            self.stable_gesture_count
            >= self.required_stable_frames
        ):
            return gesture

        return None

    # =============================================================
    # IMU HELPERS
    # =============================================================
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle <= -math.pi:
            angle += 2.0 * math.pi

        return angle

    def quaternion_to_yaw(self, x, y, z, w):
        """
        Convert a normalized ROS quaternion (x,y,z,w)
        into yaw in radians.
        """
        values = (x, y, z, w)

        if not all(math.isfinite(v) for v in values):
            return None

        norm = math.sqrt(
            x * x +
            y * y +
            z * z +
            w * w
        )

        # Reject obviously invalid quaternion.
        if norm < 0.90 or norm > 1.10:
            return None

        # Normalize small numerical deviations.
        x /= norm
        y /= norm
        z /= norm
        w /= norm

        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(sin_yaw, cos_yaw)

    # =============================================================
    # IMU CALLBACK
    # =============================================================
    def imu_callback(self, msg):
        now = time.monotonic()
        try:
            q = msg.orientation

            yaw = self.quaternion_to_yaw(
                q.x,
                q.y,
                q.z,
                q.w
            )

            if yaw is None:
                self.invalid_imu_count += 1

                self.get_logger().warn(
                    f'Invalid IMU quaternion '
                    f'({self.invalid_imu_count}/'
                    f'{self.max_invalid_imu_readings})'
                )

                if (
                    self.rotation_active
                    and self.invalid_imu_count
                    >= self.max_invalid_imu_readings
                ):
                    self.stop_rotation(
                        f'{self.max_invalid_imu_readings} consecutive invalid IMU readings'
                    )

                return

            # Reject an implausibly large yaw jump during active rotation.
            if (
                self.rotation_active
                and self.current_yaw is not None
                and self.last_valid_imu_time is not None
            ):
                dt = now - self.last_valid_imu_time

                if dt > 0.0:
                    yaw_delta = abs(
                        self.normalize_angle(yaw - self.current_yaw)
                    )

                    max_allowed_delta = (
                        self.max_yaw_rate * dt
                        + self.yaw_jump_tolerance
                    )

                    if yaw_delta > max_allowed_delta:
                        self.invalid_imu_count += 1

                        self.get_logger().warn(
                            f'Rejected IMU yaw jump | '
                            f'previous={math.degrees(self.current_yaw):.2f}° | '
                            f'new={math.degrees(yaw):.2f}° | '
                            f'delta={math.degrees(yaw_delta):.2f}°'
                        )

                        if (
                            self.invalid_imu_count
                            >= self.max_invalid_imu_readings
                        ):
                            self.stop_rotation(
                                'Repeated implausible IMU yaw jumps'
                            )

                        return

            # Valid measurement
            self.current_yaw = yaw
            self.last_valid_imu_time = now
            self.invalid_imu_count = 0
        except Exception as exc:
            self.invalid_imu_count += 1

            self.get_logger().error(
                f'IMU processing error: {exc}'
            )

            if (
                self.rotation_active
                and self.invalid_imu_count
                >= self.max_invalid_imu_readings
            ):
                self.stop_rotation(
                    'Repeated IMU processing errors'
                )

    # =============================================================
    # ROTATION CONTROLLER
    # =============================================================
    def control_rotation(self):

        if not self.rotation_active:
            return

        now = time.monotonic()

        # ---------------------------------------------------------
        # Overall rotation timeout
        # ---------------------------------------------------------
        if now >= self.rotation_end_time:

            self.stop_rotation(
                '90-degree rotation timeout'
            )

            return

        # ---------------------------------------------------------
        # IMU dropout timeout
        # ---------------------------------------------------------
        if self.last_valid_imu_time is None:
            self.stop_rotation(
                'No valid IMU reading available'
            )

            return

        if (
            now - self.last_valid_imu_time
            > self.imu_timeout
        ):
            self.stop_rotation(
                'IMU data timeout'
            )

            return

        # ---------------------------------------------------------
        # Pause after a target crossing/correction
        # ---------------------------------------------------------
        if now < self.rotation_pause_until:
            self.stop_robot()
            return

        # ---------------------------------------------------------
        # Calculate current error
        # ---------------------------------------------------------
        error = self.normalize_angle(
            self.rotation_target_yaw
            - self.current_yaw
        )

        abs_error = abs(error)

        # ---------------------------------------------------------
        # Target reached: 85° to 95°
        # ---------------------------------------------------------
        if abs_error <= self.rotation_tolerance:
            # Stop immediately when entering the target zone.
            # Subsequent IMU readings are used to observe settling.
            self.stop_robot()

            if self.rotation_stable_count == 0:
                self.get_logger().info(
                    f'TARGET ENTRY | '
                    f'start={math.degrees(self.rotation_start_yaw):.2f}° | '
                    f'current={math.degrees(self.current_yaw):.2f}° | '
                    f'turn={math.degrees(self.normalize_angle(self.current_yaw - self.rotation_start_yaw)):.2f}°'
                )

            self.rotation_stable_count += 1

            if self.rotation_stable_count >= 2:
                self.get_logger().info(
                    f'POST-STOP | '
                    f'current={math.degrees(self.current_yaw):.2f}° | '
                    f'turn={math.degrees(self.normalize_angle(self.current_yaw - self.rotation_start_yaw)):.2f}°'
                )

                self.stop_rotation(
                    'Target reached: within 85-95 degrees'
                )
                return

            return

        # Outside target zone
        self.rotation_stable_count = 0

        # ---------------------------------------------------------
        # Detect crossing of target
        # ---------------------------------------------------------
        error_sign = 1 if error > 0 else -1

        if self.rotation_last_error_sign == 0:
            self.rotation_last_error_sign = error_sign
        elif error_sign != self.rotation_last_error_sign:
            self.rotation_correction_count += 1
            self.rotation_last_error_sign = error_sign

            self.get_logger().warn(
                f'Rotation correction '
                f'{self.rotation_correction_count}/'
                f'{self.max_rotation_corrections} | '
                f'current={math.degrees(self.current_yaw):.2f}° | '
                f'target={math.degrees(self.rotation_target_yaw):.2f}° | '
                f'error={math.degrees(error):.2f}°'
            )

            self.stop_robot()

            # Give the robot time to settle before reversing.
            self.rotation_pause_until = (
                now + self.correction_pause
            )

            if (
                self.rotation_correction_count
                >= self.max_rotation_corrections
            ):

                self.stop_rotation(
                    'Maximum 5 rotation corrections reached'
                )

                return

            return

        # ---------------------------------------------------------
        # P-controller
        # ---------------------------------------------------------
        speed = (
            self.rotation_kp
            * abs_error
        )

        speed = min(
            speed,
            self.max_angular_speed
        )

        # Slow down earlier near the target to reduce overshoot.
        slow_approach_angle = math.radians(15.0)

        if abs_error <= slow_approach_angle:
            speed = self.min_turn_speed
        elif speed < self.min_turn_speed:
            speed = self.min_turn_speed

        # ---------------------------------------------------------
        # Direction
        # ---------------------------------------------------------
        angular_z = (
            self.clockwise_cmd_sign
            * self.clockwise_yaw_sign
            * error_sign
            * speed
        )

        self.publish_velocity(
            linear_x=0.0,
            angular_z=angular_z
        )

    # =============================================================
    # STOP ACTIVE ROTATION
    # =============================================================
    def stop_rotation(self, reason):

        self.stop_robot()

        self.rotation_active = False
        self.rotation_end_time = 0.0

        self.get_logger().info(
            f'Rotation stopped: {reason}'
        )

    # =============================================================
    # GESTURE -> ROBOT COMMAND
    # =============================================================
    def execute_gesture(self, gesture):
        # ---------------------------------------------------------
        # FIST = STOP
        # ---------------------------------------------------------
        if gesture == 'FIST':
            self.stop_robot()
            self.rotation_active = False

            self.get_logger().info(
                'GESTURE: FIST -> STOP'
            )

            self.last_valid_command_time = (
                time.monotonic()
            )

        # ---------------------------------------------------------
        # PALM = FORWARD
        # ---------------------------------------------------------
        elif gesture == 'PALM':

            if self.rotation_active:
                return

            self.publish_velocity(
                linear_x=self.max_linear_speed,
                angular_z=0.0
            )

            self.get_logger().info(
                'GESTURE: OPEN PALM -> FORWARD'
            )

        # ---------------------------------------------------------
        # POINT LEFT
        # ---------------------------------------------------------
        elif gesture == 'POINT_LEFT':
            if self.rotation_active:
                return

            self.publish_velocity(
                linear_x=0.0,
                angular_z=abs(
                    self.max_angular_speed
                )
            )

            self.get_logger().info(
                'GESTURE: POINT LEFT -> TURN LEFT'
            )

        # ---------------------------------------------------------
        # POINT RIGHT
        # ---------------------------------------------------------
        elif gesture == 'POINT_RIGHT':
            if self.rotation_active:
                return

            self.publish_velocity(
                linear_x=0.0,
                angular_z=-abs(
                    self.max_angular_speed
                )
            )

            self.get_logger().info(
                'GESTURE: POINT RIGHT -> TURN RIGHT'
            )

        # ---------------------------------------------------------
        # V SIGN = ONE 90-DEGREE ROTATION
        # ---------------------------------------------------------
        elif gesture == 'V_SIGN':

            if (
                not self.rotation_active
                and self.v_sign_armed
            ):
                self.start_rotation()

    # =============================================================
    # START ROTATION
    # =============================================================
    def start_rotation(self):
        now = time.monotonic()
        self.rotation_pause_until = 0.0

        # Need a recent valid IMU reading before starting.
        if self.current_yaw is None:

            self.get_logger().error(
                'V SIGN ignored: no valid IMU yaw available.'
            )

            self.stop_robot()
            return

        if (
            self.last_valid_imu_time is None
            or now - self.last_valid_imu_time
            > self.imu_timeout
        ):
            self.get_logger().error(
                'V SIGN ignored: IMU data is stale.'
            )

            self.stop_robot()
            return

        # ---------------------------------------------------------
        # Initialize rotation
        # ---------------------------------------------------------
        self.rotation_active = True
        self.v_sign_armed = False
        self.rotation_start_time = now

        self.rotation_end_time = (
            now + self.rotation_timeout
        )

        self.rotation_start_yaw = self.current_yaw

        self.rotation_target_yaw = self.normalize_angle(
            self.rotation_start_yaw
            + (
                self.clockwise_yaw_sign
                * math.radians(57.0)
            )
        )

        self.rotation_stable_count = 0
        self.rotation_correction_count = 0
        self.rotation_last_error_sign = 0

        self.get_logger().info(
            f'V SIGN -> clockwise 90-degree turn | '
            f'start={math.degrees(self.rotation_start_yaw):.2f}° | '
            f'target={math.degrees(self.rotation_target_yaw):.2f}°'
        )

        # Immediately generate the first control command.
        self.control_rotation()

    # =============================================================
    # VELOCITY PUBLISHER + SAFETY LIMIT
    # =============================================================
    def publish_velocity(
        self,
        linear_x,
        angular_z
    ):
        # ---------------------------------------------------------
        # Reject NaN / infinity
        # ---------------------------------------------------------
        if (
            not math.isfinite(linear_x)
            or not math.isfinite(angular_z)
        ):
            self.get_logger().error(
                'Invalid velocity detected. STOP.'
            )

            self.stop_robot()
            return

        # ---------------------------------------------------------
        # Limit linear velocity
        # ---------------------------------------------------------
        linear_x = max(
            -self.max_linear_speed,
            min(
                linear_x,
                self.max_linear_speed
            )
        )

        # ---------------------------------------------------------
        # Limit angular velocity
        # ---------------------------------------------------------
        angular_z = max(
            -self.max_angular_speed,
            min(
                angular_z,
                self.max_angular_speed
            )
        )

        cmd = Twist()

        cmd.linear.x = linear_x

        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = angular_z

        self.safe_publish(cmd)

        self.last_valid_command_time = (
            time.monotonic()
        )

    # =============================================================
    # WATCHDOG
    # =============================================================
    def watchdog_callback(self):
        now = time.monotonic()

        # ---------------------------------------------------------
        # Active rotation
        # ---------------------------------------------------------
        if self.rotation_active:
            self.control_rotation()
            return

        # ---------------------------------------------------------
        # Command timeout
        # ---------------------------------------------------------
        elapsed = (
            now - self.last_valid_command_time
        )

        if elapsed > self.command_timeout:
            self.stop_robot()

    # =============================================================
    # NO GESTURE
    # =============================================================
    def handle_no_gesture(self):
        if not self.rotation_active:
            self.stop_robot()

    # =============================================================
    # ERROR HANDLING
    # =============================================================
    def handle_error(self, message):
        self.error_count += 1

        self.get_logger().error(
            f'{message} | '
            f'frame={self.frame_count}, '
            f'errors={self.error_count}'
        )

        self.rotation_active = False

        self.stop_robot()

    # =============================================================
    # SAFE PUBLISH
    # =============================================================
    def safe_publish(self, cmd):
        if not rclpy.ok():
            return

        try:
            self.cmd_vel_publisher.publish(cmd)
        except Exception as exc:
            self.get_logger().error(
                f'Unable to publish /cmd_vel: {exc}'
            )

    # =============================================================
    # STOP ROBOT
    # =============================================================

    def stop_robot(self):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        self.safe_publish(cmd)

    # =============================================================
    # SHUTDOWN
    # =============================================================

    def destroy_node(self):
        self.get_logger().info(
            f'Shutting down. '
            f'Frames={self.frame_count}, '
            f'Errors={self.error_count}'
        )

        # Only try the safety stop while ROS is still active.
        if rclpy.ok():
            self.stop_robot()

        if self.hands is not None:
            self.hands.close()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Keyboard interrupt received.'
        )
    except Exception as exc:
        node.get_logger().error(
            f'Fatal node error: {exc}'
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
