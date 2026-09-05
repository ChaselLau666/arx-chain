import array
import importlib.util
import io
import json
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'source ROS environment first')
class ProbeLogTests(unittest.TestCase):
    def test_labels_and_full_messages_are_preserved(self):
        path = Path(__file__).resolve().parents[1] / 'tools/probe_vr_buttons.py'
        spec = importlib.util.spec_from_file_location('probe_vr_buttons', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stream = io.StringIO()
        log = module.SessionLog(stream)
        message = module.PosCmd()
        message.mode1 = 1
        message.temp_int_data = array.array('i', [1, 2, 3, 4, 5, 6])
        log.set_label(' right_A\n')
        log.sample('left', message)
        log.set_label('')
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[0]['label'], 'baseline')
        self.assertEqual(rows[2]['label'], 'right_A')
        self.assertEqual(rows[2]['hand'], 'left')
        self.assertEqual(rows[2]['fields']['mode1'], 1)
        self.assertEqual(rows[2]['fields']['temp_int_data'], [1, 2, 3, 4, 5, 6])
        self.assertEqual(set(rows[2]['fields']), set(message.get_fields_and_field_types()))
        self.assertEqual(rows[3]['label'], 'baseline')
        self.assertEqual(sorted(r['monotonic_ns'] for r in rows),
                         [r['monotonic_ns'] for r in rows])


if __name__ == '__main__':
    unittest.main()
