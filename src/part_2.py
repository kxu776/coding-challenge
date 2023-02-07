import numpy as np
from src import part_1


class Part2(part_1.Part1):
    def __init__(self):
        self.light_matrix = np.zeros((1000, 1000), dtype=int)

    def apply_instruction(self, instruction, update_range):
        match instruction:
                case 'turn on':
                    update_range[:] += 1
                case 'turn off':
                    update_range[update_range > 0] -= 1
                case 'toggle':
                    update_range[:] += 2
                case _:
                    raise ValueError('Unrecognised instruction input')

if __name__ == '__main__':
    part2 = Part2()
    print(part2.main())
