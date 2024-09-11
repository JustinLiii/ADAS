import sys
import os
import ast
import logging
import copy
from typing import Any
from concurrent.futures import ProcessPoolExecutor, TimeoutError, CancelledError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Process, Pipe
from tqdm import tqdm
# import multiprocessing
import contextlib

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_output(yes=True):
    if not yes:
        try:
            yield
        finally:
            return
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def _validate_code(code):
    """Basic code validation to prevent obvious harmful operations."""
    try:
        parsed = ast.parse(code)
        for node in ast.walk(parsed):
            # if isinstance(node, (ast.Import, ast.ImportFrom)):
            #     raise ValueError("Import statements are not allowed")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['system', 'popen', 'exec', 'eval']:
                    raise ValueError(f"Forbidden function call: {node.func.attr}")
    except SyntaxError as e:
        raise e

def _set_memory_limit(limit):
    """Set memory limit for the current process (Linux only)"""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    
def kill_executor(executor):
    for _, process in executor._processes.items():
        process.kill()
    
def _eval_main(code, memory_limit = None, suppress = False):
    if memory_limit is not None:
        _set_memory_limit(memory_limit)
    _validate_code(code) # if invalid, raise ValueError and SyntaxError
    with suppress_output(suppress):
        return eval(code) # may raise any error/exceptions but it's own SyntaxError
    
def save_eval(code, memory_limit = 512*1024*1024, timeout = 5, suppress_output = True) -> Any | Exception:
    # start a new process
    recv_end, send_end = Pipe(False)
    p = Process(target=lambda code, memory_limit, suppress_output, pipe:
                            pipe.send(_eval_main(code, memory_limit, suppress_output)),
                args=(code, memory_limit, suppress_output, send_end))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        p.close()
        return TimeoutError("Task timed out, this may be due to an infinite loop")
    else:
        ret = recv_end.recv()
        recv_end.close()
        p.close()
        return ret
        
        
def save_eval_batch(codes: list[str], memory_limit = 512*1024*1024, timeout = 5, suppress_output = True, max_workers=None) -> list[Any | Exception]:
    global logger
    
    ret = []
    
    # BE CAREFUL: shutdown the pool for all possible program exits
    executor =  ProcessPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(_eval_main, code, memory_limit, suppress_output) : code for code in codes}
    
    for future in tqdm(copy.copy(futures.keys()), total=len(futures)):
        try:
            ret.append(future.result(timeout=timeout))
            futures.pop(future)
        except TimeoutError as e:
            logger.error("Task timed out, this may be due to an infinite loop")
            futures.pop(future)
            ret.append(TimeoutError("Task timed out, this may be due to an infinite loop"))
            
            # Kill executor and restart it
            kill_executor(executor)
            executor.shutdown(cancel_futures=True)
            ret.extend(save_eval_batch(list(futures.values()), memory_limit, timeout, suppress_output, max_workers))
        except CancelledError as e:
            logger.error("Task cancelled")
            futures.pop(future)
            ret.append(e)
        except BrokenProcessPool as e:
            logger.error(f"BrokenProcessPool: {e}, This may be due to out of memory")
            futures.pop(future)
            ret.append(BrokenProcessPool("BrokenProcessPool, This may be due to out of memory"))
            
            # Kill executor and restart it
            kill_executor(executor)
            executor.shutdown(cancel_futures=True)
            ret.extend(save_eval_batch(list(futures.values()), memory_limit, timeout, suppress_output, max_workers))
            break
        except Exception as e:
            futures.pop(future)
            ret.append(e)
    executor.shutdown(cancel_futures=True)
    return ret

# Run tasks with ProcessPoolExecutor
# def run_tasks(agent_task_queue, max_workers):
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         future_to_task = {executor.submit(call_forward_safe, task): task for task in agent_task_queue}

#         acc_list = []
#         for future in tqdm(as_completed(future_to_task), total=len(agent_task_queue)):
#             task = future_to_task[future]
#             try:
#                 acc_list.append(future.result(timeout=120))
#             except TimeoutError:
#                 logger.error(f"Task {task} timed out")
#             except Exception as e:
#                 logger.error(f"Task {task} raised an error: {e}")
    
#     return acc_list

# Example usage
# acc_list = run_tasks(agent_task_queue, max_workers=4)
