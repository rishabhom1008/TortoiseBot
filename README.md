# TortoiseBot Gesture-Controlled Robot Using Computer Vision

## Detailed Project Description

This project implements a **gesture-controlled mobile robot** using computer vision and ROS 2. The TortoiseBot uses its onboard camera to capture live video frames, while a Python-based custom vision node processes the camera stream and recognizes predefined hand gestures.

The system uses **MediaPipe Hands** to detect hand landmarks from each camera frame. These landmarks are analyzed using custom gesture-classification logic to identify supported hand gestures. Each recognized gesture is mapped to a corresponding robot action and converted into a ROS 2 `geometry_msgs/msg/Twist` command on `/cmd_vel`.

The robot uses a differential-drive controller to translate these velocity commands into wheel motion.

For the V-sign gesture, the robot uses **IMU-based closed-loop rotation control** to perform one approximately 90-degree clockwise rotation. The controller uses the BNO055 IMU orientation feedback, proportional control, a target tolerance, stable-reading confirmation, and bounded correction logic. The camera is used only to detect the V-sign and does not participate in angle correction.

## System Pipeline

```text
Onboard Camera
      |
      v
/camera/image_raw
      |
      v
CvBridge
      |
      v
OpenCV
      |
      v
MediaPipe Hands
      |
      v
Hand Landmarks
      |
      v
Gesture Classification
      |
      v
Gesture Stability Check
      |
      v
Safety / Control Validation
      |
      v
/cmd_vel
      |
      v
differential.py
      |
      v
Robot Motors
```

## Gesture-to-Action Mapping

| Hand Gesture | Robot Action                                                      |
| ------------ | ----------------------------------------------------------------- |
| Closed fist  | Stop                                                              |
| Open palm    | Move forward                                                      |
| Point left   | Turn left                                                         |
| Point right  | Turn right                                                        |
| V sign       | Perform one approximately 90-degree clockwise rotation, then stop |

The V-sign is treated as a **one-shot command**. A single detected V-sign starts one rotation, and holding the V-sign does not continuously trigger additional rotations. A new V-sign becomes eligible only after the previous gesture has been released.

## Problem Statement

The objective of this project is to control a TortoiseBot using hand gestures detected through computer vision.

The robot should:

* Detect predefined hand gestures.
* Recognize the gestures using onboard computer vision.
* Map each gesture to a specific robot action.
* Send validated movement commands through ROS 2.
* Perform a controlled approximately 90-degree rotation when a V-sign is detected.

The supported gesture mapping is:

```text
Closed fist  -> Stop

Open palm    -> Move forward

Point left   -> Turn left

Point right  -> Turn right

V sign       -> One approximately 90-degree clockwise rotation
```

## Defined Outcome

The expected outcome is a **working gesture-controlled TortoiseBot** that receives visual commands through its onboard camera and performs the corresponding robot actions without a conventional remote controller.

The completed system demonstrates:

```text
Closed fist
    -> Robot stops

Open palm
    -> Robot moves forward

Point left
    -> Robot turns left

Point right
    -> Robot turns right

V sign
    -> Robot performs one approximately 90-degree clockwise rotation
    -> Robot stops
```

## V-Sign Rotation Control

The V-sign rotation is implemented as a **closed-loop IMU-controlled action**, rather than a fixed-duration motor command.

The control flow is:

```text
V-sign detected
      |
      v
Capture current IMU yaw
      |
      v
Define clockwise rotation target
      |
      v
Calculate yaw error
      |
      v
P-controller determines angular velocity
      |
      v
Publish /cmd_vel
      |
      v
Robot rotates
      |
      v
Read updated IMU yaw
      |
      v
Repeat control cycle
      |
      v
Enter target tolerance zone
      |
      v
Stop rotation
      |
      v
Confirm stable IMU readings
      |
      v
Complete one-shot rotation
```

The target is calibrated empirically for the physical robot so that the resulting physical rotation is approximately 90 degrees, accounting for the robot's stopping response and mechanical behavior.

## Safety and Error Handling

The system follows a **safe-default philosophy**: when the controller cannot reliably determine a valid movement command, it commands zero velocity.

```text
Invalid camera frame
        |
        v
      STOP

No recognized gesture
        |
        v
      STOP

Processing error
        |
        v
      STOP

Invalid velocity command
        |
        v
      STOP

Normal command timeout
        |
        v
      STOP

IMU data timeout during rotation
        |
        v
      STOP

Repeated invalid IMU readings
        |
        v
      STOP

Repeated implausible yaw jumps
        |
        v
      STOP

Rotation timeout / excessive corrections
        |
        v
      STOP

Completed V-sign rotation
        |
        v
      STOP
```

During an active V-sign rotation, temporary loss of the V-sign does **not** interrupt the rotation. The rotation is controlled by the IMU feedback until it successfully completes or a safety condition is triggered.

The implementation includes validation for camera messages, OpenCV image conversion, processing exceptions, velocity limits, IMU quaternion validity, IMU freshness, implausible yaw transitions, rotation timeout, and correction limits.

## Control Parameters

The current implementation uses:

```text
Maximum linear speed        = 0.10 m/s
Maximum angular speed       = 0.70 rad/s
Gesture stability           = 5 consecutive frames
Command timeout             = 0.50 s

V-sign rotation             = IMU-controlled approximately 90-degree turn

Minimum effective turn speed = 0.50 rad/s
Rotation tolerance            = ±5 degrees
Stable IMU readings required  = 2
Correction pause              = 2.0 s
Maximum corrections           = 5
Rotation timeout              = 20 s
IMU timeout                   = 1.0 s
Invalid IMU reading limit     = 10 consecutive readings
```

The minimum effective turn speed is based on the physical behavior of the robot, which does not reliably rotate below approximately `0.5 rad/s`.

The V-sign controller uses proportional control to adjust angular velocity based on the current yaw error, with configured minimum and maximum angular-speed limits.

## ROS 2 Interfaces

### Subscribed Topics

#### Camera

```text
/camera/image_raw

Type:
sensor_msgs/msg/Image
```

The camera publishes the image stream and the vision node receives each image through a ROS 2 subscription.

#### IMU

```text
/imu/data

Type:
sensor_msgs/msg/Imu
```

The vision node uses the IMU orientation quaternion to derive yaw for V-sign rotation control.

### Published Topic

```text
/cmd_vel

Type:
geometry_msgs/msg/Twist
```

The vision node publishes validated linear and angular velocity commands to the differential-drive controller.

## Main Software Components

### `tortoisebot_vision`

The vision package contains the custom gesture-control node:

```text
tortoisebot_vision/

└── scripts/

    └── image_subscriber.py
```

The node performs:

```text
Camera image subscription
        |
        v
Image validation
        |
        v
CvBridge conversion
        |
        v
MediaPipe hand detection
        |
        v
Gesture classification
        |
        v
Gesture stability checking
        |
        v
Safety / control validation
        |
        +----------------------+
        |                      |
        v                      v
Normal gesture           V-sign rotation
control                  IMU closed-loop control
        |                      |
        +----------+-----------+
                   |
                   v
                /cmd_vel
```

### `differential.py`

The TortoiseBot firmware/controller node subscribes to `/cmd_vel` and converts the commanded linear and angular velocity into differential-drive wheel velocities.

The existing TortoiseBot differential-drive controller is retained as the base motor-control component.

## Error-Handling Strategy

The control system follows a safe-default philosophy:

> When the system cannot reliably determine what command should be issued, it commands zero velocity.

The implementation includes:

* Camera/message validation
* OpenCV image validation
* `CvBridgeError` handling
* Unexpected processing exception handling
* NaN/infinity velocity rejection
* Linear and angular velocity limits
* Gesture stability checking
* Command timeout/watchdog in the vision controller
* IMU quaternion validation
* IMU freshness timeout
* Implausible yaw-jump rejection
* V-sign one-shot protection
* Rotation timeout
* Maximum correction limit
* Safe stop on shutdown

## Architecture Decision

The project uses the **onboard camera and onboard inference** approach.

```text
TortoiseBot Camera
      |
      v
TortoiseBot ROS 2
      |
      v
MediaPipe + OpenCV
      |
      v
Gesture Recognition
      |
      v
Control / Safety
      |
      v
/cmd_vel
      |
      v
TortoiseBot Differential Drive
      |
      v
Robot Motors
```

A separate development or monitoring computer may be used for ROS 2 tools, visualization, debugging, and development. The required gesture-processing pipeline is intended to execute on the robot.

## Scope and V1 Boundary

The completed V1 focuses on:

```text
Camera-based gesture recognition
        +
ROS 2 command generation
        +
Differential-drive robot control
        +
IMU-controlled single 90-degree clockwise turn
        +
Basic safety and fault handling
```

Advanced features such as LiDAR-based sensor fusion, encoder-based fusion, SLAM, autonomous navigation, and multi-step path execution are outside the V1 scope.

## Expected Demonstration

A successful demonstration should show:

1. The robot receives the onboard camera stream.
2. A closed fist stops the robot.
3. An open palm commands forward motion.
4. Pointing left commands a left turn.
5. Pointing right commands a right turn.
6. A V-sign commands one approximately 90-degree clockwise rotation.
7. The robot stops after completing the V-sign rotation.
8. Holding the V-sign does not repeatedly trigger additional rotations.
9. Loss of a valid gesture during an active V-sign rotation does not interrupt the rotation.
10. Invalid sensor data, processing errors, command faults, or rotation safety-limit
