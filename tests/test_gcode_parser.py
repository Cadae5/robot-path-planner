import unittest
import os
from gcode_processor.parser import load_gcode

class TestGcodeParser(unittest.TestCase):

    def setUp(self):
        self.test_gcode_file = "test_temp.gcode"

    def tearDown(self):
        if os.path.exists(self.test_gcode_file):
            os.remove(self.test_gcode_file)

    def write_gcode_file(self, content):
        with open(self.test_gcode_file, 'w') as f:
            f.write(content)

    def test_simple_g0_g1(self):
        self.write_gcode_file("G0 X10 Y20.5 Z-5\nG1 X15")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 10.0, 'y': 20.5, 'z': -5.0})
        self.assertEqual(commands[1], {'command': 'G1', 'x': 15.0})

    def test_comments_and_empty_lines(self):
        self.write_gcode_file("; This is a full line comment\n\nG0 X1 Y1 ; inline comment\n  \nG1 Z5")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 1.0, 'y': 1.0})
        self.assertEqual(commands[1], {'command': 'G1', 'z': 5.0})

    def test_parameter_order_independence(self):
        self.write_gcode_file("G0 Z30 Y20 X10\nG1 Y5 X15 Z25")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 10.0, 'y': 20.0, 'z': 30.0})
        self.assertEqual(commands[1], {'command': 'G1', 'x': 15.0, 'y': 5.0, 'z': 25.0})

    def test_case_insensitivity(self):
        self.write_gcode_file("g0 x10 y20 z30\nG1 X15 Y5 Z25")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 10.0, 'y': 20.0, 'z': 30.0})
        self.assertEqual(commands[1], {'command': 'G1', 'x': 15.0, 'y': 5.0, 'z': 25.0})

    def test_ignore_other_commands(self):
        self.write_gcode_file("G0 X1\nM30\nG28\nG1 Y2")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 1.0})
        self.assertEqual(commands[1], {'command': 'G1', 'y': 2.0})

    def test_file_not_found(self):
        # Suppress print output for this test
        # Note: load_gcode currently prints to stdout on error.
        # For cleaner tests, it might be better to raise an exception.
        # For now, we just check it returns an empty list.
        commands = load_gcode("non_existent_file.gcode")
        self.assertEqual(commands, [])

    def test_no_space_params(self):
        self.write_gcode_file("G0X1Y2Z3\nG1X10.5Y20.5Z30.5")
        commands = load_gcode(self.test_gcode_file)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], {'command': 'G0', 'x': 1.0, 'y': 2.0, 'z': 3.0})
        self.assertEqual(commands[1], {'command': 'G1', 'x': 10.5, 'y': 20.5, 'z': 30.5})
