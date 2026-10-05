"""ROS 2 Node running SpatialReasoningPolicy on live or simulated images."""

import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vln_interfaces.msg import EpisodeControl, PolicyAction
except ImportError:
    rclpy = None
    Node = object
    Image = None
    EpisodeControl = None
    PolicyAction = None

from vln_policy.spatial_reasoning_policy import SpatialReasoningPolicy, SpatialReasoningConfig


class VlnSpatialReasoningPolicyNode(Node if rclpy else object):
    """ROS 2 Node executing limited-FOV spatial reasoning on /vln/input/image."""

    def __init__(self):
        if rclpy is None:
            raise RuntimeError("rclpy is not installed; cannot initialize VlnSpatialReasoningPolicyNode")
        super().__init__('vln_spatial_reasoning_policy_node')

        # Declare parameters
        self.declare_parameter('episode_id', 'spatial_ep_001')
        self.declare_parameter('instruction', 'navigate through the room')
        self.declare_parameter('approach_velocity', 0.35)
        self.declare_parameter('explore_velocity', 0.25)
        self.declare_parameter('autostart', False)

        ep_id = self.get_parameter('episode_id').get_parameter_value().string_value
        instruction = self.get_parameter('instruction').get_parameter_value().string_value
        v_approach = self.get_parameter('approach_velocity').get_parameter_value().double_value
        v_explore = self.get_parameter('explore_velocity').get_parameter_value().double_value

        config = SpatialReasoningConfig(
            approach_linear_velocity=v_approach,
            explore_linear_velocity=v_explore,
        )
        self.policy = SpatialReasoningPolicy(config=config)
        self.policy.reset(episode_id=ep_id)
        self._instruction = instruction
        self._active = self.get_parameter('autostart').get_parameter_value().bool_value

        # QoS
        qos_sensor = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_action = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.action_pub = self.create_publisher(PolicyAction, '/vln/policy_action', qos_action)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/vln/input/image', self._on_image, qos_sensor
        )
        self.ctrl_sub = self.create_subscription(
            EpisodeControl, '/vln/episode_control', self._on_episode_control, qos_action
        )

        self.get_logger().info(
            f"VlnSpatialReasoningPolicyNode initialized (ep={ep_id}, active={self._active})"
        )

    def _on_episode_control(self, msg: EpisodeControl):
        if msg.command_type == EpisodeControl.CMD_START:
            self._active = True
            self._instruction = msg.instruction or self._instruction
            self.policy.reset(episode_id=msg.episode_id)
            self.get_logger().info(f"Episode started: id={msg.episode_id}, prompt='{self._instruction}'")
        elif msg.command_type == EpisodeControl.CMD_STOP:
            self._active = False
            self.get_logger().info(f"Episode stopped: id={msg.episode_id}")

    def _on_image(self, msg: Image):
        if not self._active:
            return

        # Convert ROS Image to RGB numpy array
        try:
            if msg.encoding in ('rgb8', 'bgr8'):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                rgb = raw if msg.encoding == 'rgb8' else raw[:, :, ::-1]
            elif msg.encoding == 'mono8':
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")
            return

        action_data, info = self.policy.step(rgb, instruction=self._instruction)

        out_msg = PolicyAction()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = "base_link"
        out_msg.episode_id = action_data.episode_id
        out_msg.sequence_id = action_data.sequence_id
        out_msg.linear_velocity = float(action_data.linear_velocity)
        out_msg.angular_velocity = float(action_data.angular_velocity)
        out_msg.stop_probability = float(action_data.stop_probability)

        self.action_pub.publish(out_msg)


def main(args=None):
    if rclpy is None:
        print("rclpy not installed; cannot run node directly", file=sys.stderr)
        return
    rclpy.init(args=args)
    node = VlnSpatialReasoningPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
