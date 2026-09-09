# -- coding: UTF-8
"""Watch, without touching, everything that can make an arm move.

Read-only by construction: this node creates subscriptions and no publishers,
so running it cannot itself command the arms. It exists to answer one question
after the arms dive at startup - what was sent to them, by whom, and when.

Three things can move an arm on this stack, and all three are watched here:

  * a PosCmd on the topic the arm subscribes to (`--arm-cmd-topics`, the
    filtered topics under the normal launcher). Both vr_pose_filter and
    collect.py publish there, so the publisher list is recorded alongside every
    command and re-checked as it changes: that is what says which of the two
    sent a given frame.
  * an /arx_joy GO_HOME request, which puts X5Controller in GO_HOME and walks
    it to go_home_position.
  * go_home_position itself, applied by the SDK as the controller comes up.
    Nothing is published for that, so it shows only as motion in the status
    stream with no command preceding it - which is exactly what makes it worth
    telling apart from the other two.

The lift is watched too (`/body_information`), because "the robot dived" reads
the same to an operator whether the column dropped or the arms did.

Output goes to one directory: events.log is the human-readable timeline, and
the CSVs hold every message for exact analysis afterwards. Everything is
flushed on write, so a kill -9 loses nothing.
"""

import os
import sys
import time

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import csv
import datetime
from collections import deque

from utils.setup_loader import setup_loader


def stamp(when=None):
    when = time.time() if when is None else when
    return datetime.datetime.fromtimestamp(when).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def build_node(args):
    import rclpy
    from rclpy.node import Node
    from arm_control.msg import PosCmd
    from arx5_arm_msg.msg import RobotStatus
    from std_msgs.msg import Int32MultiArray

    POS_CMD_FIELDS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper', 'height',
                      'chx', 'chy', 'chz', 'head_pit', 'head_yaw', 'mode1', 'mode2')

    class DiveProbe(Node):
        def __init__(self):
            super().__init__(args.node_name)
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.out = out

            self.events = open(out / 'events.log', 'a', buffering=1)
            self.cmd_csv, self.cmd_rows = self._csv(
                'commands.csv',
                ['wall', 'mono', 'topic', 'publishers'] + list(POS_CMD_FIELDS))
            self.status_csv, self.status_rows = self._csv(
                'status.csv',
                ['wall', 'mono', 'topic', 'ex', 'ey', 'ez', 'eroll', 'epitch', 'eyaw']
                + [f'joint{i}' for i in range(7)])
            self.other_csv, self.other_rows = self._csv(
                'other.csv', ['wall', 'mono', 'topic', 'payload'])

            self.t0 = time.monotonic()
            # Context kept per arm so a dive can be explained by what came just
            # before it rather than by what happens to arrive after.
            self.recent_cmd = {}
            self.recent_status = {}
            self.z_history = {}
            self.publishers_seen = {}
            self.first_seen = set()
            self.counts = {}
            self.dives = 0

            for topic in args.arm_cmd_topics:
                self._watch_pos_cmd(topic, 'cmd')
            for topic in args.vr_topics:
                self._watch_pos_cmd(topic, 'vr')
            for topic in args.status_topics:
                self.recent_status[topic] = deque(maxlen=args.context)
                self.z_history[topic] = deque(maxlen=4000)
                self.create_subscription(
                    RobotStatus, topic,
                    lambda msg, t=topic: self.on_status(t, msg), 50)
            self.create_subscription(
                Int32MultiArray, args.joy_topic,
                lambda msg: self.on_joy(msg), 20)
            self.create_subscription(
                PosCmd, args.body_topic,
                lambda msg: self.on_body(msg), 20)

            self.create_timer(args.publisher_poll, self.poll_publishers)
            self.create_timer(args.report_period, self.report)

            self.log('probe', f'read-only probe up, writing to {out}')
            self.log('probe', f'arm command topics: {" ".join(args.arm_cmd_topics)}')
            self.log('probe', f'raw VR topics:      {" ".join(args.vr_topics)}')
            self.log('probe', f'arm status topics:  {" ".join(args.status_topics)}')
            self.log('probe', f'dive rule: end_pos z falling more than '
                              f'{args.dive_drop:.3f} m within {args.dive_window:.2f} s')

        # setup helpers

        def _csv(self, name, header):
            path = self.out / name
            fresh = not path.exists() or path.stat().st_size == 0
            handle = open(path, 'a', newline='', buffering=1)
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(header)
            return handle, writer

        def _watch_pos_cmd(self, topic, kind):
            from arm_control.msg import PosCmd
            self.recent_cmd[topic] = deque(maxlen=args.context)
            self.create_subscription(
                PosCmd, topic,
                lambda msg, t=topic, k=kind: self.on_pos_cmd(t, k, msg), 50)

        # logging

        def log(self, tag, message):
            line = f'{stamp()}  [{tag}] {message}'
            self.events.write(line + '\n')
            print(line, flush=True)

        def publishers_of(self, topic):
            try:
                infos = self.get_publishers_info_by_topic(topic)
            except Exception as exc:                      # topic not up yet
                return f'<unknown: {exc}>'
            if not infos:
                return '<none>'
            return ' '.join(sorted(f'{i.node_namespace.rstrip("/")}/{i.node_name}'
                                   .lstrip('/') for i in infos))

        def poll_publishers(self):
            """Who is publishing each watched topic, recorded as it changes.

            Two nodes publish to the arm command topic - the pose filter and
            collect.py - so the identity of the publisher is the difference
            between "the headset drove the arm down" and "the collector did".
            """
            for topic in list(self.recent_cmd) + list(self.recent_status):
                now = self.publishers_of(topic)
                if self.publishers_seen.get(topic) != now:
                    was = self.publishers_seen.get(topic, '<first look>')
                    self.publishers_seen[topic] = now
                    self.log('pub', f'{topic}: {was}  ->  {now}')

        # callbacks

        def on_pos_cmd(self, topic, kind, msg):
            wall, mono = time.time(), time.monotonic() - self.t0
            values = [float(getattr(msg, f, 0.0) or 0.0) for f in POS_CMD_FIELDS]
            who = self.publishers_seen.get(topic) or self.publishers_of(topic)
            self.cmd_rows.writerow([f'{wall:.6f}', f'{mono:.6f}', topic, who]
                                   + [f'{v:.6f}' for v in values])
            self.recent_cmd[topic].append((wall, mono, who, values))
            self.counts[topic] = self.counts.get(topic, 0) + 1
            if topic not in self.first_seen:
                self.first_seen.add(topic)
                self.log(kind, f'{topic}: FIRST message, from [{who}]  '
                               f'xyz=({values[0]:+.4f} {values[1]:+.4f} {values[2]:+.4f}) '
                               f'rpy=({values[3]:+.3f} {values[4]:+.3f} {values[5]:+.3f}) '
                               f'gripper={values[6]:+.3f} height={values[7]:+.3f}')

        def on_status(self, topic, msg):
            wall, mono = time.time(), time.monotonic() - self.t0
            end = [float(v) for v in list(msg.end_pos)[:6]]
            joints = [float(v) for v in list(msg.joint_pos)[:7]]
            while len(end) < 6:
                end.append(0.0)
            while len(joints) < 7:
                joints.append(0.0)
            self.status_rows.writerow([f'{wall:.6f}', f'{mono:.6f}', topic]
                                      + [f'{v:.6f}' for v in end]
                                      + [f'{v:.6f}' for v in joints])
            self.recent_status[topic].append((wall, mono, end, joints))
            self.counts[topic] = self.counts.get(topic, 0) + 1
            if topic not in self.first_seen:
                self.first_seen.add(topic)
                self.log('arm', f'{topic}: FIRST feedback, '
                                f'xyz=({end[0]:+.4f} {end[1]:+.4f} {end[2]:+.4f})')
            self.check_dive(topic, mono, end)

        def on_joy(self, msg):
            wall, mono = time.time(), time.monotonic() - self.t0
            data = list(msg.data)
            self.other_rows.writerow([f'{wall:.6f}', f'{mono:.6f}',
                                      args.joy_topic, ' '.join(str(v) for v in data)])
            go_home = len(data) > 1 and data[1] == 1
            if go_home != getattr(self, '_joy_go_home', None):
                self._joy_go_home = go_home
                self.log('joy', f'{args.joy_topic} data={data}'
                                f'{"  -> GO_HOME requested" if go_home else ""}')

        def on_body(self, msg):
            wall, mono = time.time(), time.monotonic() - self.t0
            height = float(getattr(msg, 'height', 0.0) or 0.0)
            self.other_rows.writerow([f'{wall:.6f}', f'{mono:.6f}',
                                      args.body_topic, f'height={height:.6f}'])
            last = getattr(self, '_body_height', None)
            if last is None or abs(height - last) > args.body_step:
                self._body_height = height
                self.log('lift', f'{args.body_topic} height={height:.4f}'
                                 + ('' if last is None else f' (was {last:.4f})'))

        # the thing this probe exists for

        def check_dive(self, topic, mono, end):
            history = self.z_history[topic]
            history.append((mono, end[2]))
            cutoff = mono - args.dive_window
            while len(history) > 1 and history[0][0] < cutoff:
                history.popleft()
            if len(history) < 2:
                return
            highest = max(z for _, z in history)
            drop = highest - end[2]
            if drop < args.dive_drop:
                return
            history.clear()
            self.dives += 1
            self.log('DIVE', f'{topic}: z fell {drop:.4f} m within {args.dive_window:.2f} s '
                             f'({highest:+.4f} -> {end[2]:+.4f}). Context follows.')
            self.dump_context(topic)

        def dump_context(self, status_topic):
            for topic, entries in self.recent_cmd.items():
                self.log('DIVE', f'  last {min(len(entries), args.dump)} on {topic} '
                                 f'(publishers: {self.publishers_seen.get(topic, "?")})')
                for wall, mono, who, values in list(entries)[-args.dump:]:
                    self.log('DIVE', f'    {stamp(wall)} t={mono:8.3f} [{who}] '
                                     f'xyz=({values[0]:+.4f} {values[1]:+.4f} {values[2]:+.4f}) '
                                     f'rpy=({values[3]:+.3f} {values[4]:+.3f} {values[5]:+.3f}) '
                                     f'gripper={values[6]:+.3f}')
                if not entries:
                    self.log('DIVE', '    <nothing was ever published here>')
            entries = self.recent_status.get(status_topic, ())
            self.log('DIVE', f'  last {min(len(entries), args.dump)} on {status_topic}')
            for wall, mono, end, joints in list(entries)[-args.dump:]:
                self.log('DIVE', f'    {stamp(wall)} t={mono:8.3f} '
                                 f'xyz=({end[0]:+.4f} {end[1]:+.4f} {end[2]:+.4f}) '
                                 f'j=({" ".join(f"{v:+.3f}" for v in joints[:6])})')

        def report(self):
            counted = ', '.join(f'{t}={n}' for t, n in sorted(self.counts.items()))
            self.log('probe', f'{self.dives} dive(s) so far; messages: {counted or "<none yet>"}')

    return rclpy, DiveProbe()


def main(args):
    setup_loader(ROOT)
    rclpy, node = build_node(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log('probe', 'probe stopping')
        node.destroy_node()
        # A signal-driven stop has already shut the context down, and shutting
        # it down twice raises over the top of the real exit.
        try:
            rclpy.shutdown()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir',
                        default=os.path.join('/tmp', 'dive_probe_'
                                             + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                        help='directory for events.log and the CSVs')
    parser.add_argument('--arm-cmd-topics', nargs='*',
                        default=['/ARX_VR_L_filtered', '/ARX_VR_R_filtered'],
                        help='the topics the arms actually subscribe to')
    parser.add_argument('--vr-topics', nargs='*',
                        default=['/ARX_VR_L', '/ARX_VR_R'],
                        help='the raw headset stream, before filtering')
    parser.add_argument('--status-topics', nargs='*',
                        default=['/arm_l_status_full', '/arm_r_status_full'],
                        help='where each arm reports its end-effector pose')
    parser.add_argument('--joy-topic', default='/arx_joy')
    parser.add_argument('--body-topic', default='/body_information')
    parser.add_argument('--node-name', default='dive_probe')
    parser.add_argument('--dive-drop', type=float, default=0.05,
                        help='fall in end_pos z, in metres, that counts as a dive')
    parser.add_argument('--dive-window', type=float, default=0.5,
                        help='seconds the fall has to happen within')
    parser.add_argument('--body-step', type=float, default=0.05,
                        help='lift height change worth a line in events.log')
    parser.add_argument('--context', type=int, default=400,
                        help='messages kept per topic for a dive dump')
    parser.add_argument('--dump', type=int, default=25,
                        help='messages printed per topic when a dive is seen')
    parser.add_argument('--publisher-poll', type=float, default=0.5)
    parser.add_argument('--report-period', type=float, default=5.0)

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    import rclpy
    rclpy.init()
    args = parse_args()
    main(args)
