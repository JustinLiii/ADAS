import argparse
import copy
import json
import os
import pickle
import random
import logging
import contextlib
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
import multiprocessing

import backoff
import numpy as np
import openai
from tqdm import tqdm

# from arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt
from exec import _set_memory_limit, suppress_output

# TODO: 为什么把这个初始化写在这里 （怒
ZHIPU_API_KEY = os.environ["ZHIPU_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
zhipu_client = openai.OpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
openai_client = openai.OpenAI(base_url="https://api.openai.com/v1", api_key=OPENAI_API_KEY)

import utils.datasets as dataset_utils
from utils import random_id, list_to_string, bootstrap_confidence_interval

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\nUSE DOUBLE QUOTES " INSTEAD OF SINGLE QUOTES ' FOR PROPERTY KEYS AND VALUES\nDO NOT write code outside of the json field\n"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

input_tokens = 0
output_tokens = 0

def count_tokens(response):
    input = response.usage.prompt_tokens
    output = response.usage.completion_tokens
    return input, output


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        keys,
        system_message,
        temperature=0.5
) -> dict:
    global input_tokens
    global output_tokens
    
    if 'gpt' in model:
        client = openai_client
    elif 'glm' in model:
        client = zhipu_client
    else:
        raise AttributeError("Unsupported model")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(e)
        raise e
        
    content = response.choices[0].message.content
    i, o = count_tokens(response)
    input_tokens += i
    output_tokens += o
    
    
    logger = logging.getLogger(__name__)
    logger.debug('REQUEST\n' + str([
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ]))
    logger.debug('RESPONSE\n' + str(content))
    logger.debug('TOKEN_COUNT\n' + f'input: {input_tokens}, output： {output_tokens}')
    
    # somehow glm tend to do this
    if content is not None:
        content = content.lstrip('```json').rstrip('```')
        
    try:
        json_dict = json.loads(content) #type: ignore
    except json.JSONDecodeError:
        try:
            response = client.chat.completions.create(
                model='glm-4-flash',
                messages=[
                    {"role": "system", "content": f"rewrite to JSON object with and only with keys {keys}, response should only be a JSON object"},
                    {"role": "user", "content": content}, # type:ignore
                ],
                temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception(e)
            raise e
            
        content = response.choices[0].message.content
        i, o = count_tokens(response)
        input_tokens += i
        output_tokens += o
        logger.debug('REFORMAT\n' + str(content))
        logger.debug('TOKEN_COUNT\n' + f'input: {input_tokens}, output： {output_tokens}')
        if content is not None:
            content = content.lstrip('```json').rstrip('```')
        
    try:
        json_dict = json.loads(content) #type: ignore
    except:
        json_dict = {}
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, [openai.RateLimitError, openai.BadRequestError], max_time=120)
def get_json_response_from_gpt_reflect(
        msg_list,
        keys,
        model,
        temperature=0.8
) -> dict:
    global input_tokens
    global output_tokens
    
    if 'gpt' in model:
        client = openai_client
    elif 'glm' in model:
        client = zhipu_client
    else:
        raise AttributeError("Unsupported model")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=msg_list,
            temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(e)
        raise e
    content = response.choices[0].message.content
    i, o = count_tokens(response)
    input_tokens += i
    output_tokens += o
    
    logger = logging.getLogger(__name__)
    logger.debug('REQUEST\n' + str(msg_list))
    logger.debug('RESPONSE\n' + str(content))
    logger.debug('TOKEN_COUNT\n' + f'input: {input_tokens}, output： {output_tokens}')
    
    # somehow glm tend to do this
    if content is not None:
        content = content.lstrip('```json').rstrip('```')
    try:
        json_dict = json.loads(content) #type: ignore
    except json.JSONDecodeError:
        try:
            response = client.chat.completions.create(
                model='glm-4-flash',
                messages=[
                    {"role": "system", "content": f"rewrite to JSON object with and only with keys {keys}, response should only be a JSON object"},
                    {"role": "user", "content": content}, # type:ignore
                ],
                temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception(e)
            raise e
        content = response.choices[0].message.content
        i, o = count_tokens(response)
        input_tokens += i
        output_tokens += o
        logger.debug('REFORMAT\n' + str(content))
        logger.debug('TOKEN_COUNT\n' + f'input: {input_tokens}, output： {output_tokens}')
        if content is not None:
            content = content.lstrip('```json').rstrip('```')
    
    try:
        json_dict = json.loads(content) #type: ignore
    except json.JSONDecodeError as e:
        logger = logging.getLogger(__name__)
        logger.exception(e)
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
            response_json = get_json_response_from_gpt(prompt, self.model, self.output_fields, system_prompt, self.temperature)
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
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            return gen_output(f"Error during code execution: {repr(e)}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                return gen_output(f"Error during function execution: {repr(e)}"), correct_examples, wrong_examples

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
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            return gen_output(f"Error during code execution: {repr(e)}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            return gen_output(f"Error during function execution: {repr(e)}")

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
        with set_logger_output(n+1):
            output_fields = ["thought", "name", "code"]
            system_prompt, prompt = get_prompt(archive)
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                next_solution = get_json_response_from_gpt_reflect(msg_list, output_fields, args.model)

                Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
                # Reflexion 1
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_1})
                next_solution = get_json_response_from_gpt_reflect(msg_list, output_fields, args.model)
                # Reflexion 2
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": Reflexion_prompt_2})
                next_solution = get_json_response_from_gpt_reflect(msg_list, output_fields, args.model)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print("During LLM generate new solution:")
                print(repr(e))
                logger = logging.getLogger(__name__)
                logger.exception(e)
                n -= 1
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
                    print(repr(e))
                    logger = logging.getLogger(__name__)
                    logger.exception(e)
                    msg_list.append({"role": "assistant", "content": str(next_solution)})
                    msg_list.append({"role": "user", "content": f"Error during evaluation:\n{repr(e)}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                    try:
                        next_solution = get_json_response_from_gpt_reflect(msg_list, output_fields, args.model)
                    except Exception as e:
                        print("During LLM generate new solution:")
                        print(repr(e))
                        logger = logging.getLogger(__name__)
                        logger.exception(e)
                        continue
                    continue
            if not acc_list:
                # don't save result if no successful code runs had been made
                n -= 1
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
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception(e)
            continue
            # raise e
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1

def call_forward(agent_task):
    _set_memory_limit(512 * 1024 * 1024)
    with suppress_output():
        try:
            agent, taskInfo, arc_data = agent_task
            res = agent.forward(taskInfo)
        except Exception as e:
            # if exception in forward, raise
            logger = multiprocessing.get_logger()
            logger.exception(e)
            raise e
        
        try:
            origin_res = res
            if isinstance(res, Info):
                res = res.content
            if isinstance(res, str):
                res = eval(res)
            hard_score = dataset_utils.arc.eval_solution(res, arc_data, soft_eval=False)
            return hard_score
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logger = multiprocessing.get_logger()
            logger.exception(e) 
            return 0

def evaluate_forward_fn(args, forward_str):
    global debug_log_lock
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

    if args.task == 'arc':
        if SEARCHING_MODE:
            arc_dir = args.val_data_path
        else:
            arc_dir = args.test_data_path

        print(arc_dir)
        with open(arc_dir, 'rb') as pickle_file:
            arc_data_queue = pickle.load(pickle_file)
        task_queue = []
        for arc_data in arc_data_queue:
            task_str, examples, test_input = dataset_utils.arc.format_arc_data(arc_data)
            taskInfo = Info('task', 'User', task_str, -1)
            task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data)] * args.n_repeat)
    
    elif args.task == 'drop':
        examples = dataset_utils.drop.load_drop(args.data_filename)[1:-1]  # first one and the last one is for few-shot examples
        random.seed(args.shuffle_seed)
        random.shuffle(examples)

        if SEARCHING_MODE:
            examples = examples[:args.valid_size] * args.n_repreat
        else:
            examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

        questions = [example['inputs'] for example in examples]
        answers = [example['targets'] for example in examples]
        
        task_queue = []
        for q in questions:
            taskInfo = Info('task', 'User', q, -1)
            task_queue.append(taskInfo)

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    logger = logging.getLogger(__name__)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if args.task == 'arc':
            futures = [executor.submit(call_forward, task) for task in task_queue]
        else:
            agentSystem = AgentSystem()
            futures = [executor.submit(agentSystem.forward, task) for task in task_queue]
        acc_list = []
        
        timeout = False
        for future in tqdm(futures):
            try:
                acc_list.append(future.result(timeout=60))
            except TimeoutError:
                logger.error(f"Task timed out")
                acc_list.append(0)
                timeout = True
            except BrokenProcessPool as e:
                # with debug_log_lock:
                #     logger.exception(e)
                # acc_list.append(0)
                for process in list(executor._processes.values()):
                    process.kill()
                executor.shutdown(cancel_futures=True)
                raise e
            except Exception as e:
                logger.exception(e)
                print('Vital/Unexpected exception, shutting down ProcessPool')
                # kill process pool
                for process in list(executor._processes.values()):
                    process.kill()
                executor.shutdown(cancel_futures=True)
                raise e # catch forward() exception and other exception
        
        # if there's timeout process, force exit process pool
        if timeout:
            for process in list(executor._processes.values()):
                process.kill()
            print("ProcessPool killed for timeout process")
                    
    print("acc:", bootstrap_confidence_interval(acc_list))
    return acc_list



def config_logger(args):
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(
        os.path.join(args.expr_home_dir, "experiment.log"), 
        encoding="utf-8",
        mode='a',
    )
    file_handler.addFilter(lambda record: record.name == '__main__')
    file_handler.setFormatter(logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s'))
    
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)
    logger.addHandler(file_handler)
    
    multi_logger = multiprocessing.get_logger()
    multi_logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)
    
    for handler in copy.copy(multi_logger.handlers):
        multi_logger.removeHandler(handler)
    
    multi_logger.addHandler(file_handler)
    

@contextlib.contextmanager
def set_logger_output(n: int):
    logger = logging.getLogger(__name__)
    multi_logger = multiprocessing.get_logger()
    
    file_handler = logging.FileHandler(
        os.path.join(args.expr_home_dir, f"generation_{n}.log"), 
        encoding="utf-8",
        mode='w',
    )
    file_handler.addFilter(lambda record: record.name == '__main__')
    file_handler.setFormatter(logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s'))
    logger.removeHandler(logger.handlers[0])
    logger.addHandler(file_handler)
    multi_logger.removeHandler(multi_logger.handlers[0])
    multi_logger.addHandler(file_handler)
    
    try:
        yield
    finally:
        file_handler = logging.FileHandler(
            os.path.join(args.expr_home_dir, "experiment.log"), 
            encoding="utf-8",
            mode='a',
        )
        file_handler.addFilter(lambda record: record.name == '__main__')
        file_handler.setFormatter(logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s'))

        logger.removeHandler(logger.handlers[0])
        logger.addHandler(file_handler)
        multi_logger.removeHandler(multi_logger.handlers[0])
        multi_logger.addHandler(file_handler)
    

def setup_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--val_data_path', type=str, default='dataset/sampled_arc_val_data.pkl')
    # parser.add_argument('--test_data_path', type=str, default='dataset/sampled_arc_test_data.pkl')
    parser.add_argument('--task', type=str, default='arc') 
    parser.add_argument('--n_repeat', type=int, default=5) # drop 1
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=24)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='temp/')
    parser.add_argument('--expr_name', type=str, default=None)
    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-mini-2024-07-18',
                        choices=['gpt-4o-mini-2024-07-18',
                                 'gpt-4-turbo-2024-04-09', 
                                 'gpt-3.5-turbo-0125', 
                                 'gpt-4o-2024-05-13',
                                 'gemini-1.5-flash',
                                 'glm-4-flash',
                                 'glm-4-plus'])

    args = parser.parse_args()
    # format task data path
    if args.task == 'arc':
        args.test_data_path = 'dataset/sampled_arc_test_data.pkl'
        args.val_data_path = 'dataset/sampled_arc_val_data.pkl'
    elif args.task == 'drop':
        args.data_filename = 'dataset/drop_v0_dev.jsonl.gz'
        args.valid_size = 128
        args.test_size = 800
        args.shuffle_seed = 0
    
    if args.expr_name is None:
        args.expr_name = args.task + "_" + args.model + "_results"
        
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

if __name__ == "__main__":
    args = setup_args()
    
    if args.task == 'arc':
        from _arc.arc_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    elif args.task == 'drop':
        from _drop.drop_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    else:
        raise ValueError(f"不支持的任务: {args.task}")
        
    config_logger(args)
    
    # search
    SEARCHING_MODE = True
    search(args)

    # evaluate
    SEARCHING_MODE = False
    evaluate(args)
