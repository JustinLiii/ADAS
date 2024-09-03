import argparse
import copy
import json
import os
import pickle
import logging
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt

# TODO: 为什么把这个初始化写在这里 （怒
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print('===============================================================')
    print('Cannot initialize OpenAI client for following reason:')
    print(f' - {e}')
    print('Trying other clients')
    print('===============================================================')

from utils import random_id, format_arc_data, eval_solution, list_to_string, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\nDO NOT add ```json or ```\nDO NOT write code outside of the json field"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    
    
    logger = logging.getLogger(__name__)
    logger.debug('REQUEST\n' + str([
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ]))
    logger.debug('RESPONSE\n' + str(content))
    
    # somehow glm tend to do this
    if content is not None:
        content = content.lstrip('```json').rstrip('```')
        
    try:
        json_dict = json.loads(content) #type: ignore
    except:
        json_dict = {}
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    
    logger = logging.getLogger(__name__)
    logger.debug('REQUEST\n' + str(msg_list))
    logger.debug('RESPONSE\n' + str(content))
    
    # somehow glm tend to do this
    if content is not None:
        content = content.lstrip('```json').rstrip('```')
    
    try:
        json_dict = json.loads(content) #type: ignore
    except:
        json_dict = {}
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='glm-4-flash', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> tuple[str, str]:
        code_output = False

        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `transform` function. DO NOT return anything else. (It will be tested later.)"
                code_output = True
        system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue

            if isinstance(content, list):
                try:
                    content = list_to_string(content)
                except:
                    pass

            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            if len(response_json) != len(self.output_fields):
                # try to fill in the missing field
                for key in self.output_fields:
                    if not key in response_json and len(response_json) < len(self.output_fields):
                        response_json[key] = ''
                for key in copy.deepcopy(list(response_json.keys())):
                    if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                        del response_json[key]
        except Exception as e:
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            else:
                raise e
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except Exception as e:
                return gen_output("Error during function execution: {e}"), correct_examples, wrong_examples

            if transformed_grid == output_grid:
                feedback += f"Your transform function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    transformed_grid = list_to_string(transformed_grid)
                except:
                    pass
                feedback += f"Your transform function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {transformed_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        return gen_output(feedback), correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
        except Exception as e:
            return gen_output("Error during function execution: {e}")

        return gen_output(transform_output)


def search(args):
    # get agent archieve
    file_path = os.path.join(args.expr_home_dir, "run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            # don't eval what had been evaluated
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            raise e

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            logger = logging.getLogger(__name__)
            logger.exception(e)
            continue

        acc_list = []
        # eval next solution
        # debug at most debug_max times for current agent
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                logger = logging.getLogger(__name__)
                logger.exception(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            # don't save result if no successful code runs had been made
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    file_path = os.path.join(args.expr_home_dir, "run_archive.json")
    eval_file_path = file_path.strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            # print(e)
            # continue
            raise e
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1


def evaluate_forward_fn(args, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    if SEARCHING_MODE:
        arc_dir = args.val_data_path
    else:
        arc_dir = args.test_data_path
    print(arc_dir)
    with open(arc_dir, 'rb') as pickle_file:
        arc_data_queue = pickle.load(pickle_file)

    print(f"problem length: {len(arc_data_queue) * args.n_repreat}")
    max_workers = min(len(arc_data_queue) * args.n_repreat, args.max_workers) if args.multiprocessing else 1

    agent_task_queue = []
    for arc_data in arc_data_queue:
        task_str, examples, test_input = format_arc_data(arc_data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data)] * args.n_repreat)

    def call_forward(agent_task_queue):
        agent, taskInfo, arc_data = agent_task_queue
        res = agent.forward(taskInfo)
        origin_res = res
        try:
            if isinstance(res, Info):
                res = res.content
            if isinstance(res, str):
                res = eval(res)
            hard_score = eval_solution(res, arc_data, soft_eval=False)
            return hard_score
        except Exception as e:
            print(e)
            logger = logging.getLogger(__name__)
            logger.exception(e) 
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acc_list = list(tqdm(executor.map(call_forward, agent_task_queue), total=len(agent_task_queue)))

    print("acc:", bootstrap_confidence_interval(acc_list))
    return acc_list

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, default='dataset/sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='dataset/sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='temp/')
    parser.add_argument('--expr_name', type=str, default=None)
    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='glm-4-flash',
                        choices=['gpt-4o-mini-2024-07-18',
                                 'gpt-4-turbo-2024-04-09', 
                                 'gpt-3.5-turbo-0125', 
                                 'gpt-4o-2024-05-13',
                                 'gemini-1.5-flash',
                                 'glm-4-flash',
                                 'glm-4-plus'
                                 ])

    args = parser.parse_args()
    if args.expr_name is None:
        args.expr_name = "arc_" + args.model + "_results"
        
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.expr_home_dir = os.path.join(args.save_dir, args.expr_name)
    if not os.path.exists(args.expr_home_dir):
        os.mkdir(args.expr_home_dir)
        
    # Save args to a JSON file
    args_file = os.path.join(args.expr_home_dir, 'config.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    return args

def config_logger(args):
    def filter(record):
        assert isinstance(record, logging.LogRecord)
        return record.name == '__main__'
    
    
    file_handler = logging.FileHandler(
        os.path.join(args.expr_home_dir, "experiment.log"), 
        encoding="utf-8",
        mode='w'
    )
    file_handler.addFilter(filter)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler],
    )

if __name__ == "__main__":
    args = setup_args()
        
    config_logger(args)
    
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
