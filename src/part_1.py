import pandas as pd
import re
import numpy as np

class Part1:
    INPUT_FILEPATH = '../coding_challenge_input.txt'

    light_matrix = None

    def __init__(self):
        self.light_matrix = np.zeros((1000, 1000), dtype=bool)

    @staticmethod
    def parse_input_row_to_dict(row_string):
        instruction = re.compile('\d').split(row_string, 1)[0].strip()
        coord_pairs = re.compile('\s(\d{1,3},\d{1,3})(?:\s|$)').findall(row_string) # parses valid coordinates up to 3 digits
        if not instruction or len(coord_pairs) != 2:
            raise ValueError(
                'Misformatted input file row - Expected format: INSTRUCTION COORDINATE,COORDINATE through COORDINATE,COORDINATE')
        start_coords, end_coords = [tuple(map(int, coord.split(',', 1))) for coord in
                                        coord_pairs]
        return {'instruction': instruction, 'start_coords': start_coords, 'end_coords': end_coords}

    def read_file_to_dataframe(self, filepath):
        with open(filepath, 'r') as input_file:
            file_data = input_file.readlines()
        instructions = []
        for row in file_data:
            instructions.append(self.parse_input_row_to_dict(row))
        return pd.DataFrame(instructions)

    def apply_instruction(self, instruction, update_range):
        match instruction:
            case 'turn on':
                update_range[:] = True
            case 'turn off':
                update_range[:] = False
            case 'toggle':
                update_range[:] = np.bitwise_not(update_range[:])
            case _:
                raise ValueError('Unrecognised instruction input')

    def apply_instruction_row_to_matrix(self,instruction_row, matrix):
        start_x, start_y = instruction_row['start_coords']
        end_x, end_y = instruction_row['end_coords']
        update_range = matrix[start_x:end_x + 1, start_y:end_y + 1]
        self.apply_instruction(instruction_row['instruction'], update_range)

    def main(self):
        input_instructions = self.read_file_to_dataframe(self.INPUT_FILEPATH)
        for i, instruction_row in input_instructions.iterrows():
            self.apply_instruction_row_to_matrix(instruction_row, self.light_matrix.view())
        return self.light_matrix.sum()

if __name__ == '__main__':
    part1 = Part1()
    print(part1.main())
