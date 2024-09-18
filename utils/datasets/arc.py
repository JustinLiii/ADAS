# modified from https://github.com/jennyzzt/LLM_debate_on_ARC
# prompt also inspired by https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/master/arc_solve/prompting.py   

import concurrent.futures
import numpy as np

from ..utils import list_to_string, get_percentage_match

TASK_OVERVIEW = """You will be given some number of paired example inputs and outputs grids. The outputs were produced by applying a transformation rule to the input grids. In addition to the paired example inputs and outputs, there is also one test input without a known output.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color. 0 is black.
Your task is to determine the transformation rule from examples and find out the answer, involving determining the size of the output grid for the test and correctly filling each cell of the grid with the appropriate color or number.

The transformation only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the grid visually, and try to find the pattern.
"""

def format_arc_data(arc_data, direct=False):
    task_str = TASK_OVERVIEW

    task_demo_str = ''
    # Get task demo string
    task_demo_str += '## Examples:\n\n'
    for i, demo in enumerate(arc_data['train']):
        task_demo_str += f'### Example {i}:\n'
        task_demo_str += f'input = {list_to_string(demo["input"])}\n'
        task_demo_str += f'output = {list_to_string(demo["output"])}\n\n'

    # Get task test string
    task_test_str = ''
    for testcase in arc_data['test']:
        task_test_str += '## Test Problem:\n'
        task_test_str += f'Given input:\n {list_to_string(testcase["input"])}\n\n'
        task_test_str += f'Analyze the transformation rules based on the provided Examples and determine what the output should be for the Test Problem.'

    task_str += task_demo_str + task_test_str

    return task_str, arc_data['train'], arc_data['test'][0]['input']

def eval_solution(output, arc_data, soft_eval=False):
    if not output:
        return 0

    solution = arc_data['test'][0]['output']
    if soft_eval:
        score = get_percentage_match(solution, output)
    else:
        score = 1 if output == solution else 0
    return score

def eval_algo(solve_fn, arc_data, soft_eval=False):
    # Calculate percentage of test cases done correctly
    testcases = arc_data['test']
    scores = []
    for testcase in testcases:
        input = testcase['input']
        output = testcase['output']
        gen_output = None
        # Run solve_fn with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(solve_fn, input)
                try:
                    gen_output = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    future.cancel()
            except:  # if the function does not work
                continue
        # Check if correct output
        if soft_eval:
            score = get_percentage_match(output, gen_output)
        else:
            score = 1 if output == gen_output else 0
        scores.append(score)
    return np.mean(scores)