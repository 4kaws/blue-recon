#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, base64, requests, json, time, os

URL_FILE = os.path.expanduser('~/gdrive/cosmos-cookoff/inference_url.txt')

def get_inference_url():
    if os.path.exists(URL_FILE):
        with open(URL_FILE) as f:
            url = f.read().strip()
        return url + '/infer'
    raise RuntimeError(f'inference_url.txt not found at {URL_FILE}. Start notebook 08 first.')


class BlueReconBridgeNode(Node):
    def __init__(self):
        super().__init__('blue_recon_bridge')
        self.inference_url = get_inference_url()
        self.bridge        = CvBridge()
        self.latest_frame  = None
        self.frame_id      = 0

        self.cam_sub       = self.create_subscription(
            Image, '/auv/camera/image_raw', self.camera_callback,
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE))
        self.cmd_pub       = self.create_publisher(Twist,  '/auv/cmd_vel', 10)
        self.reasoning_pub = self.create_publisher(String, '/blue_recon/reasoning', 10)
        self.timer         = self.create_timer(2.0, self.inference_cycle)

        self.get_logger().info('Blue Recon Bridge Node started')
        self.get_logger().info(f'Inference server: {self.inference_url}')

    def camera_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def inference_cycle(self):
        if self.latest_frame is None:
            self.get_logger().warn('No camera frame yet...')
            return

        self.frame_id += 1
        t0 = time.time()

        _, buf = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        try:
            resp = requests.post(self.inference_url, json={
                'image_b64': img_b64,
                'frame_id':  self.frame_id,
                'timestamp': t0,
            }, timeout=15)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e} -> STOP')
            self.cmd_pub.publish(Twist())
            return

        rj      = result['reasoning']
        action  = rj.get('recommended_action', 'STOP')
        hazard  = rj.get('hazard_level', 'UNKNOWN')
        conf    = rj.get('confidence', 0.0)
        latency = (time.time() - t0) * 1000

        self.get_logger().info(
            f'[Frame #{self.frame_id}] {action} | hazard={hazard} | '
            f'conf={conf:.2f} | {latency:.0f}ms | '
            f'{rj.get("action_reasoning", "")}'
        )

        self.reasoning_pub.publish(String(data=json.dumps(rj, indent=2)))
        cmd = self.action_to_twist(action, conf, hazard)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f'  /auv/cmd_vel -> linear.x={cmd.linear.x:.2f} angular.z={cmd.angular.z:.2f}'
        )

    def action_to_twist(self, action, confidence, hazard):
        twist = Twist()
        hazard_speed = {
            'NONE': 0.5, 'LOW': 0.4, 'MEDIUM': 0.3,
            'HIGH': 0.2, 'CRITICAL': 0.1, 'UNKNOWN': 0.2
        }
        speed = max(0.5, hazard_speed.get(hazard, 0.5))
        mapping = {
            'PROCEED':       (speed,        0.0),
            'REDUCE_SPEED':  (speed * 0.6,  0.0),
            'STOP':          (speed * 0.3,  0.0),
            'TURN_LEFT':     (speed * 0.4,  0.4),
            'TURN_RIGHT':    (speed * 0.4, -0.4),
            'ASCEND':        (speed * 0.3,  0.0),
            'DESCEND':       (speed * 0.3,  0.0),
            'ABORT_MISSION': (speed * 0.2,  0.2),
        }
        lx, az = mapping.get(action, (0.0, 0.0))
        twist.linear.x  = lx
        twist.angular.z = az
        if action == 'ASCEND':        twist.linear.z =  0.3
        if action == 'DESCEND':       twist.linear.z = -0.3
        if action == 'ABORT_MISSION': twist.linear.z =  0.3
        return twist


def main(args=None):
    rclpy.init(args=args)
    node = BlueReconBridgeNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
