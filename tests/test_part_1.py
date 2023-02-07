import unittest
import pandas as pd
import numpy as np
from src.part_1 import Part1

part1 = Part1()

class TestMatrixInitialisation(unittest.TestCase):
    def test_light_matrix_initialises_correctly(self):
        light_matrix = part1.light_matrix
        self.assertEqual(light_matrix.shape, (1000,1000))
        self.assertEqual(light_matrix.dtype, bool)
        self.assertEqual(light_matrix.sum(), 0)

class TestReadFile(unittest.TestCase):
    def test_invalid_filepath_throws(self):
        invalid_filepath = 'invalid_filepath.txt'
        with self.assertRaises(FileNotFoundError):
            part1.read_file_to_dataframe(invalid_filepath)

    def test_valid_filepath_returns_dataframe(self):
        # given
        valid_filepath = 'test.txt'
        with open(valid_filepath, 'r') as testfile:
            expected_length = len(testfile.readlines())

        # when
        result = part1.read_file_to_dataframe(valid_filepath)

        # then
        self.assertIs(type(result), pd.DataFrame)
        self.assertEqual(result.columns.values.tolist(), ['instruction', 'start_coords', 'end_coords'])
        self.assertEqual(len(result.index), expected_length)

class TestParseInputRowToDict(unittest.TestCase):
    def test_valid_input_row_returns_dictionary(self):
        # given
        valid_input_row = 'turn on 0,0 through 999,999'
        expected_result = {'instruction': 'turn on', 'start_coords': (0,0), 'end_coords': (999,999)}

        # when
        result = part1.parse_input_row_to_dict(valid_input_row)

        # then
        self.assertEqual(result, expected_result)

    def test_invalid_input_rows_throw(self):
        # given
        invalid_input_strings = ['', '111,222 through 333,444', 'turn on through', 'toggle through 333,444', 'turn off 111,222 through','turn on 1111,0 through 0,0', 'turn on 0,1111 through 0,0', 'turn on 0,0 through 1111,0', 'turn on 0,0 through 0,1111']
        misformatted_input_error = 'Misformatted input file row - Expected format: INSTRUCTION COORDINATE,COORDINATE through COORDINATE,COORDINATE'

        for invalid_input in invalid_input_strings:
            # when
            with self.assertRaises(ValueError) as raised_error:
                part1.parse_input_row_to_dict(invalid_input)

            # then
            self.assertEqual(str(raised_error.exception), misformatted_input_error)

class TestApplyInstructionRowToMatrix(unittest.TestCase):
    def test_updates_correct_matrix_range_for_input(self):
            # given
            mock_light_matrix = np.full((3, 3), False)
            instruction_row = {'instruction': 'turn on', 'start_coords': [0, 0], 'end_coords': [0, 1]}
            expected_matrix_state = np.array([[True, False, False], [True, False, False], [False, False, False]])

            # when
            part1.apply_instruction_row_to_matrix(instruction_row, mock_light_matrix)

            # then
            self.assertEqual(mock_light_matrix.all(), expected_matrix_state.all())
            self.assertEqual(mock_light_matrix.sum(), 2)

class TestApplyInstruction(unittest.TestCase):
    def test_invalid_instruction_throws(self):
        # given
        invalid_instruction = 'invalid_instruction'
        invalid_instruction_error = 'Unrecognised instruction input'
        with self.assertRaises(ValueError) as raised_error:
            # when
            part1.apply_instruction(invalid_instruction, np.full((3,3), False).view())

        # then
        self.assertEqual(str(raised_error.exception), invalid_instruction_error)

    def test_valid_turn_on_instruction_updates_matrix(self):
        # given
        turn_on_instruction = 'turn on'
        mock_update_range = np.full((3,3), False)
        self.assertEqual(mock_update_range.sum(), 0)
        expected_final_state = np.full((3,3), True)

        # when
        part1.apply_instruction(turn_on_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 9)

    def test_valid_turn_off_instruction_updates_matrix(self):
        # given
        turn_off_instruction = 'turn off'
        mock_update_range = np.full((3,3), True)
        self.assertEqual(mock_update_range.sum(), 9)
        expected_final_state = np.full((3,3), False)

        # when
        part1.apply_instruction(turn_off_instruction, mock_update_range.view())

        # then
        self.assertEqual(mock_update_range.all(), expected_final_state.all())
        self.assertEqual(mock_update_range.sum(), 0)

    def test_valid_toggle_instruction_updates_matrix(self):
        toggle_instruction = 'toggle'
        matrix_state_assertions = [{'initial_light_value': False, 'initial_matrix_sum': 0, 'expected_final_matrix_sum': 9},{'initial_light_value': True, 'initial_matrix_sum': 9, 'expected_final_matrix_sum': 0}]

        for assertion in matrix_state_assertions:
            # given
            mock_update_range = np.full((3,3), assertion['initial_light_value'])
            self.assertEqual(mock_update_range.sum(), assertion['initial_matrix_sum'])
            expected_final_state = np.full((3,3), not assertion['initial_light_value'])

            # when
            part1.apply_instruction(toggle_instruction, mock_update_range)

            # then
            self.assertEqual(mock_update_range.all(), expected_final_state.all())
            self.assertEqual(mock_update_range.sum(), assertion['expected_final_matrix_sum'])

class TestApplyMultipleInstructionRows(unittest.TestCase):
    def test_main(self):
        # given
        part1_main = Part1()
        part1_main.INPUT_FILEPATH = 'test.txt'

        # when
        part1_main.main()

        # then
        self.assertEqual(part1_main.light_matrix.sum(), 998004)

if __name__ == '__main__':
    unittest.main()
