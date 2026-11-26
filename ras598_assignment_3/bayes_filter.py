#!/usr/bin/env python3
"""
RAS 598: Mobile Robotics - Assignment 3
3D Discrete Bayes Filter (Histogram Filter) for Robot Localization

State space: (x, y, theta)  ->  belief array of shape (80, 80, NUM_THETA_BINS)
- World spans -20m to +20m in X and Y  (40m x 40m)
- Spatial resolution: 0.5 m/cell  -> 80 x 80 grid
- Angular resolution: 10 deg/bin  -> 36 orientation bins

Author: RAS598 Student
"""

import math
import numpy as np
from scipy.ndimage import gaussian_filter

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

# ─────────────────────────────────────────────
#  Stage / world – landmark map (ID: [x, y])
# ─────────────────────────────────────────────
LANDMARKS = {
    10: (-8.0,  8.0),   # Near starting area
    20: (-4.0,  2.0),   # Center-left corridor
    30: (-8.0, -6.0),   # Top-left alcove
    40: ( 0.0, -2.0),   # Central passage
    50: ( 4.0,  2.0),   # Near central junction
}

# ─────────────────────────────────────────────
#  Grid / world constants
# ─────────────────────────────────────────────
WORLD_MIN   = -20.0   # metres
WORLD_MAX   =  20.0
WORLD_SIZE  =  40.0   # metres total

CELL_SIZE   = 0.5     # metres per cell
GRID_W      = 80      # cells along X
GRID_H      = 80      # cells along Y

DEG_RES     = 10      # degrees per orientation bin
NUM_THETA   = 360 // DEG_RES   # 36 bins

# ─────────────────────────────────────────────
#  Noise / tuning parameters
# ─────────────────────────────────────────────
SIGMA_RANGE   = 1.2   # metres – landmark range noise
SIGMA_BEARING = 15.0  # degrees – landmark bearing noise (converted to rad below)
SIGMA_BEARING_RAD = math.radians(SIGMA_BEARING)

# Gaussian diffusion after motion (cells)
DIFFUSE_XY    = 0.8   # spatial blur sigma (cells)
DIFFUSE_THETA = 0.6   # orientation blur sigma (bins)

EPSILON = 1e-300      # numerical stability floor


# ══════════════════════════════════════════════════════════
#  Coordinate mapping helpers
# ══════════════════════════════════════════════════════════

def real_to_grid(x_world: float, y_world: float, theta_world: float):
    """
    Convert world coordinates (metres, radians) to grid indices (int, int, int).

    World  :  x in [-20, 20],  y in [-20, 20],  theta in [-pi, pi]
    Grid   :  ix in [0, 79],   iy in [0, 79],   it in [0, 35]

    NOTE on Y-axis:
        World  y = +20 (top)    -> grid row iy = 0  (top of array)
        World  y = -20 (bottom) -> grid row iy = 79 (bottom of array)
    """
    ix = int((x_world - WORLD_MIN) / CELL_SIZE)
    iy = int((WORLD_MAX - y_world) / CELL_SIZE)   # flip Y
    ix = np.clip(ix, 0, GRID_W - 1)
    iy = np.clip(iy, 0, GRID_H - 1)

    # theta: 0 rad = facing +X = bin 0
    theta_deg = math.degrees(theta_world) % 360.0
    it = int(theta_deg / DEG_RES) % NUM_THETA

    return ix, iy, it


def grid_to_real(ix: int, iy: int, it: int):
    """
    Convert grid indices back to world coordinates (centre of cell, radians).
    """
    x_world = WORLD_MIN + (ix + 0.5) * CELL_SIZE
    y_world = WORLD_MAX - (iy + 0.5) * CELL_SIZE   # flip Y
    theta_rad = math.radians(it * DEG_RES)
    return x_world, y_world, theta_rad


# Pre-compute centre coordinates of every cell (vectorised helpers)
_ix_all  = np.arange(GRID_W)
_iy_all  = np.arange(GRID_H)
_it_all  = np.arange(NUM_THETA)

# Shape (80,) – world x/y of each column/row
_cell_x  = WORLD_MIN + (_ix_all + 0.5) * CELL_SIZE           # (80,)
_cell_y  = WORLD_MAX - (_iy_all + 0.5) * CELL_SIZE           # (80,)
_cell_th = np.radians(_it_all * DEG_RES)                      # (36,)

# Meshgrid (80, 80, 36) for full vectorised likelihood
_X, _Y, _TH = np.meshgrid(_cell_x, _cell_y, _cell_th, indexing='ij')
#  _X[ix, iy, it] = world x of cell (ix, iy, it)   etc.


# ══════════════════════════════════════════════════════════
#  Bayes Filter
# ══════════════════════════════════════════════════════════

class BayesFilter:
    """
    3-D Histogram Filter over (x, y, theta).

    belief : numpy array of shape (GRID_W, GRID_H, NUM_THETA)
             belief[ix, iy, it]  =  P(robot at cell (ix,iy,it))
    """

    def __init__(self):
        # Uniform prior
        self.belief = np.ones((GRID_W, GRID_H, NUM_THETA), dtype=np.float64)
        self._normalize()

    # ──────────────────────────────────────────
    #  Normalisation
    # ──────────────────────────────────────────

    def _normalize(self):
        total = self.belief.sum() + EPSILON
        self.belief /= total

    # ──────────────────────────────────────────
    #  Prediction step  (motion model)
    # ──────────────────────────────────────────

    def predict(self, delta_x: float, delta_y: float, delta_theta: float):
        """
        Shift the belief by (delta_x, delta_y, delta_theta) and diffuse.

        Turn-Go-Turn decomposition:
          1. Rotate by delta_theta_1  (initial yaw change)
          2. Translate by (delta_x, delta_y) relative to NEW heading
          3. Rotate by delta_theta_2  (final yaw correction)
        Here we receive the net delta_x/y in *world* frame and net delta_theta.

        np.roll wraps the grid, which is acceptable for a closed-world grid.
        """
        if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6 and abs(delta_theta) < 1e-6:
            return

        # ── Orientation shift ──────────────────────────────
        # delta_theta in radians -> bins to shift
        dtheta_bins = int(round(math.degrees(delta_theta) / DEG_RES))
        if dtheta_bins != 0:
            self.belief = np.roll(self.belief, dtheta_bins, axis=2)

        # ── Spatial shift ──────────────────────────────────
        # world delta -> cell delta
        dx_cells = int(round(delta_x / CELL_SIZE))
        # Y-axis is FLIPPED: +world_y -> smaller row index -> negative roll on axis-1
        dy_cells = int(round(-delta_y / CELL_SIZE))

        if dx_cells != 0:
            self.belief = np.roll(self.belief, dx_cells, axis=0)
        if dy_cells != 0:
            self.belief = np.roll(self.belief, dy_cells, axis=1)

        # ── Diffusion (entropy growth) ─────────────────────
        self.belief = gaussian_filter(
            self.belief,
            sigma=[DIFFUSE_XY, DIFFUSE_XY, DIFFUSE_THETA],
            mode='wrap'
        )
        self.belief = np.clip(self.belief, EPSILON, None)
        self._normalize()

    # ──────────────────────────────────────────
    #  Update step  (measurement model)
    # ──────────────────────────────────────────

    def update(self, measurements: list):
        """
        measurements : list of (landmark_id, meas_range, meas_bearing_rad)
        For each measurement compute the likelihood over the full grid and
        multiply into the belief (Bayes rule).
        """
        if not measurements:
            return

        for lm_id, meas_range, meas_bearing in measurements:
            if lm_id not in LANDMARKS:
                continue

            lm_x, lm_y = LANDMARKS[lm_id]

            # ── Expected range for every cell ──────────────
            # _X, _Y : (80, 80, 36)
            dx = lm_x - _X
            dy = lm_y - _Y
            exp_range = np.sqrt(dx**2 + dy**2)                 # (80,80,36)

            # ── Expected bearing for every cell ────────────
            # bearing = angle to landmark in robot-local frame
            #   = atan2(dy, dx) - robot_heading
            angle_to_lm = np.arctan2(dy, dx)                   # global angle
            exp_bearing = angle_to_lm - _TH                    # relative bearing

            # Wrap to [-pi, pi]
            exp_bearing = (exp_bearing + math.pi) % (2 * math.pi) - math.pi

            # ── Gaussian likelihoods ───────────────────────
            range_diff   = exp_range - meas_range
            bearing_diff = exp_bearing - meas_bearing
            bearing_diff = (bearing_diff + math.pi) % (2*math.pi) - math.pi

            like_range   = np.exp(-0.5 * (range_diff   / SIGMA_RANGE      )**2)
            like_bearing = np.exp(-0.5 * (bearing_diff / SIGMA_BEARING_RAD)**2)

            likelihood = like_range * like_bearing + EPSILON

            # ── Bayes update ───────────────────────────────
            self.belief *= likelihood
            self.belief  = np.clip(self.belief, EPSILON, None)
            self._normalize()

    # ──────────────────────────────────────────
    #  MAP estimate
    # ──────────────────────────────────────────

    def best_estimate(self):
        """Return (x_world, y_world, theta_rad) of the highest-probability cell."""
        idx = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return grid_to_real(*idx)

    # ──────────────────────────────────────────
    #  2-D marginalised belief (for OccupancyGrid)
    # ──────────────────────────────────────────

    def marginalised_xy(self) -> np.ndarray:
        """Sum over orientation axis, return (GRID_W, GRID_H) float array."""
        return self.belief.sum(axis=2)


# ══════════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════════

class BayesFilterNode(Node):

    def __init__(self):
        super().__init__('bayes_filter')

        # ── Bayes filter ──────────────────────────────────
        self.bf = BayesFilter()

        # ── State tracking ────────────────────────────────
        self.prev_odom        = None   # last Odometry message
        self.odom_path_poses  = []
        self.gt_path_poses    = []

        # ── Publishers ────────────────────────────────────
        self.pub_belief = self.create_publisher(
            OccupancyGrid, 'viz/belief_costmap', 10)
        self.pub_landmarks = self.create_publisher(
            MarkerArray, 'viz/landmarks', 10)
        self.pub_gt_path = self.create_publisher(
            Path, 'viz/gt_path', 10)
        self.pub_odom_path = self.create_publisher(
            Path, 'viz/odom_path', 10)

        # ── TF broadcaster (publish estimated pose) ───────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Subscribers ───────────────────────────────────
        # Noisy odometry
        from nav_msgs.msg import Odometry as OdomMsg
        self.sub_odom = self.create_subscription(
            OdomMsg, '/odom', self.odom_callback, 10)

        # Fiducial landmarks
        # The message type for /fiducials in stage_ros2 is
        # geometry_msgs/PoseArray or a custom fiducials message.
        # We handle both: try the stage_ros2 Fiducials type first.
        try:
            from stage_ros2_interfaces.msg import Fiducials
            self.sub_fiducials = self.create_subscription(
                Fiducials, '/fiducials', self.fiducials_callback, 10)
            self.get_logger().info('Using stage_ros2_interfaces Fiducials msg')
        except ImportError:
            # Fallback: geometry_msgs/PoseArray  (id encoded in orientation.w)
            from geometry_msgs.msg import PoseArray
            self.sub_fiducials = self.create_subscription(
                PoseArray, '/fiducials', self.fiducials_pose_array_callback, 10)
            self.get_logger().warn(
                'stage_ros2_interfaces not found – using PoseArray fallback')

        # Ground truth (for visualisation ONLY – never fed to filter)
        from nav_msgs.msg import Odometry as OdomMsg2
        self.sub_gt = self.create_subscription(
            OdomMsg2, '/ground_truth', self.gt_callback, 10)

        # ── Visualisation timer ────────────────────────────
        self.create_timer(0.5, self.publish_visualisations)
        self.get_logger().info('BayesFilterNode started.')

    # ══════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════

    def odom_callback(self, msg):
        """Compute delta motion from consecutive odometry and call predict()."""
        if self.prev_odom is None:
            self.prev_odom = msg
            return

        # ── Extract pose from Odometry ────────────────────
        p_cur = msg.pose.pose.position
        q_cur = msg.pose.pose.orientation

        p_prv = self.prev_odom.pose.pose.position
        q_prv = self.prev_odom.pose.pose.orientation

        # Yaw from quaternion
        yaw_cur = self._quat_to_yaw(q_cur)
        yaw_prv = self._quat_to_yaw(q_prv)

        delta_x     = p_cur.x - p_prv.x
        delta_y     = p_cur.y - p_prv.y
        delta_theta = self._angle_diff(yaw_cur, yaw_prv)

        self.prev_odom = msg

        # ── Predict step ──────────────────────────────────
        self.bf.predict(delta_x, delta_y, delta_theta)

        # ── Record path for visualisation ─────────────────
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp    = msg.header.stamp
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose            = msg.pose.pose
        self.odom_path_poses.append(pose_stamped)

    def fiducials_callback(self, msg):
        """
        Handle stage_ros2_interfaces/Fiducials message.
        Each fiducial has: id, pose (robot-relative, metres).
        """
        measurements = []
        for fid in msg.fiducials:
            lm_id = fid.id
            # fid.pose is relative to robot: x = forward, y = left
            rel_x = fid.pose.x
            rel_y = fid.pose.y
            meas_range   = math.sqrt(rel_x**2 + rel_y**2)
            meas_bearing = math.atan2(rel_y, rel_x)
            measurements.append((lm_id, meas_range, meas_bearing))

        self.bf.update(measurements)

    def fiducials_pose_array_callback(self, msg):
        """
        Fallback for PoseArray encoding of fiducials.
        Convention: pose.position = (rel_x, rel_y, id_encoded_in_z_or_orientation)
        """
        measurements = []
        for pose in msg.poses:
            lm_id = int(round(pose.orientation.w))  # id encoded in w
            rel_x = pose.position.x
            rel_y = pose.position.y
            meas_range   = math.sqrt(rel_x**2 + rel_y**2)
            meas_bearing = math.atan2(rel_y, rel_x)
            measurements.append((lm_id, meas_range, meas_bearing))

        self.bf.update(measurements)

    def gt_callback(self, msg):
        """Record ground-truth path (visualisation only – NEVER used by filter)."""
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp    = msg.header.stamp
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose            = msg.pose.pose
        self.gt_path_poses.append(pose_stamped)

    # ══════════════════════════════════════════
    #  Visualisation
    # ══════════════════════════════════════════

    def publish_visualisations(self):
        now = self.get_clock().now().to_msg()
        self._publish_belief_costmap(now)
        self._publish_landmarks(now)
        self._publish_gt_path(now)
        self._publish_odom_path(now)
        self._publish_estimate_tf(now)

    def _publish_belief_costmap(self, stamp):
        """Marginalise belief over theta, publish as OccupancyGrid."""
        bel_2d = self.bf.marginalised_xy()  # (80, 80)

        # Normalise to [0, 1]
        bel_min = bel_2d.min()
        bel_max = bel_2d.max() + EPSILON
        bel_norm = (bel_2d - bel_min) / (bel_max - bel_min)

        # OccupancyGrid expects row-major, y-first (row=y, col=x)
        # belief array is [ix, iy] = [col, row], so transpose
        bel_t = bel_norm.T  # shape (80, 80), row=iy col=ix

        # Map to [0, 100]
        data = (bel_t * 100).astype(np.int8).flatten().tolist()

        grid                = OccupancyGrid()
        grid.header.stamp   = stamp
        grid.header.frame_id = 'map'
        grid.info.resolution = CELL_SIZE
        grid.info.width      = GRID_W
        grid.info.height     = GRID_H
        # Origin = bottom-left corner of the grid in world frame
        grid.info.origin.position.x = WORLD_MIN
        grid.info.origin.position.y = WORLD_MIN
        grid.info.origin.position.z = 0.0
        grid.data = data

        self.pub_belief.publish(grid)

    def _publish_landmarks(self, stamp):
        """Publish landmark cylinders as MarkerArray."""
        ma = MarkerArray()
        for i, (lm_id, (lx, ly)) in enumerate(LANDMARKS.items()):
            m               = Marker()
            m.header.stamp  = stamp
            m.header.frame_id = 'map'
            m.ns            = 'landmarks'
            m.id            = lm_id
            m.type          = Marker.CYLINDER
            m.action        = Marker.ADD
            m.pose.position.x = lx
            m.pose.position.y = ly
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 1.0
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.9
            ma.markers.append(m)
        self.pub_landmarks.publish(ma)

    def _publish_gt_path(self, stamp):
        """Publish ground-truth path in green."""
        path             = Path()
        path.header.stamp     = stamp
        path.header.frame_id  = 'map'
        path.poses            = self.gt_path_poses[-2000:]
        self.pub_gt_path.publish(path)

    def _publish_odom_path(self, stamp):
        """Publish raw odometry path in red."""
        path             = Path()
        path.header.stamp    = stamp
        path.header.frame_id = 'map'
        path.poses           = self.odom_path_poses[-2000:]
        self.pub_odom_path.publish(path)

    def _publish_estimate_tf(self, stamp):
        """Broadcast TF from map -> base_link_estimated."""
        x, y, th = self.bf.best_estimate()
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'map'
        t.child_frame_id  = 'base_link_estimated'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        cy = math.cos(th / 2.0)
        sy = math.sin(th / 2.0)
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)

    # ══════════════════════════════════════════
    #  Utilities
    # ══════════════════════════════════════════

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Extract yaw from a geometry_msgs/Quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed angular difference a - b, wrapped to [-pi, pi]."""
        d = a - b
        return (d + math.pi) % (2 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = BayesFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
