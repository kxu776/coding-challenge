import unittest
import numpy as np
from src.part_2 import Part2

part2 = Part2()

class TestMatrixInitialisation(unittest.TestCase):
    def test_light_matrix_initialises_correctly(self):
        light_matrix = part2.light_matrix
        self.assertEqual(light_matrix.shape, (1000,1000))
        self.assertEqual(light_matrix.dtype, int)
        self.assertEqual(light_matrix.sum(), 0)

class TestApplyInstruction(unittest.TestCase):
    def test_invalid_instruction_throws(self):
        # given
        invalid_instruction = 'invalid_instruction'
        invalid_instruction_error = 'Unrecognised instruction input'
        with self.assertRaises(ValueError) as raised_error:
            # when
            part2.apply_instruction(invalid_instruction, np.full((3, 3), False).view())

        # then
        self.assertEqual(str(raised_error.exception), invalid_instruction_error)

    def test_valid_turn_on_instruction_updates_matrix(self):
        # given
        turn_on_instruction = 'turn on'
        mock_update_range = np.full((3,3), 0)
        self.assertEqual(mock_update_range.sum(), 0)
        expected_final_state = np.full((3,3), 1)

        # when
        part2.apply_instruction(turn_on_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 9)

    def test_valid_turn_off_instruction_updates_matrix(self):
        # given
        turn_off_instruction = 'turn off'
        mock_update_range = np.full((3,3), 1)
        self.assertEqual(mock_update_range.sum(), 9)
        expected_final_state = np.full((3,3), 0)

        # when
        part2.apply_instruction(turn_off_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 0)

    def test_turn_off_instruction_does_not_decrement_below_zero(self):
        # given
        turn_off_instruction = 'turn off'
        mock_update_range = np.full((3, 3), 0)
        self.assertEqual(mock_update_range.sum(), 0)
        expected_final_state = mock_update_range.copy()

        # when
        part2.apply_instruction(turn_off_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 0)

    def test_valid_toggle_instruction_increments_matrix_by_two(self):
        # given
        toggle_instruction = 'toggle'
        mock_update_range = np.full((3, 3), 0)
        self.assertEqual(mock_update_range.sum(), 0)
        expected_final_state = np.full((3, 3), 2)

        # when
        part2.apply_instruction(toggle_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 18)

class TestApplyMultipleInstructionRows(unittest.TestCase):
    def test_main(self):
        # given
        part2_main = Part2()
        part2_main.INPUT_FILEPATH = 'test.txt'

        # when
        part2_main.main()

        # then
        self.assertEqual(part2_main.light_matrix.sum(), 1003996)

if __name__ == '__main__':
    unittest.main()
