import sys
import os
import ast
import logging
import traceback
from typing import Any
from concurrent.futures import ProcessPoolExecutor, TimeoutError, CancelledError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from tqdm import tqdm
import contextlib

logger = logging.getLogger(__name__)

class ExecutionException(Exception):
    """Exception raised for errors during the execution of the generated code.
       exception messages are the repr of the original exception."""
    pass

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
    try:
        with suppress_output(suppress):
            return eval(code) # may raise any error/exceptions but it's own SyntaxError, ValueError and MemoryError
    except MemoryError as e:
        raise e
    except Exception as e:
        if e.__traceback__ is not None:
            # skip the first frame of the traceback, only show the evaled code
            tb = e.__traceback__.tb_next
        else:
            tb = None
        raise ExecutionException(''.join(traceback.format_exception(type(e), e, tb))) from None
    
def _call_main(fn, arg, memory_limit = None, suppress = False):
    if memory_limit is not None:
        _set_memory_limit(memory_limit)
    try:
        with suppress_output(suppress):
            return fn(arg) # may raise any error/exceptions but it's own SyntaxError, ValueError and MemoryError
    except MemoryError as e:
        raise e
    except Exception as e:
        if e.__traceback__ is not None:
            # skip the first frame of the traceback, only show the evaled code
            tb = e.__traceback__.tb_next
        else:
            tb = None
        raise ExecutionException(''.join(traceback.format_exception(type(e), e, tb))) from None
    
def _eval_wrapper(code, memory_limit, suppress_output, pipe: Connection):
    try:
        pipe.send(_eval_main(code, memory_limit, suppress_output))
    except Exception as e:
        e_str = traceback.format_exception(type(e), e, e.__traceback__)
        pipe.send(e_str)
        # pipe.send(e)
        raise e
    
def save_eval(code, memory_limit = 512*1024*1024, timeout = 5, suppress_output = True) -> Any | Exception:
    # start a new process
    recv_end, send_end = Pipe(False)
    p = Process(target=_eval_wrapper,
                args=(code, memory_limit, suppress_output, send_end))
    p.start()
    p.join(timeout)
    if p.is_alive():
        recv_end.close()
        p.kill()
        p.join()
        p.close()
        return TimeoutError("Task timed out, this may be due to an infinite loop")
    elif p.exitcode != 0:
        if recv_end.poll():
            e_str= recv_end.recv()
            e_str = ''.join(e_str)
            e = ExecutionException(e_str)
        recv_end.close()
        p.kill()
        p.join()
        p.close()
        if e:
            raise e
    else:
        if not recv_end.poll():
            recv_end.close()
            p.close()
            raise Exception("No return value from the process")
        
        ret = recv_end.recv()
        recv_end.close()
        p.close()
        return ret
        
        
def save_eval_batch(codes: list[str], 
                    memory_limit = 512*1024*1024, 
                    timeout = 5, 
                    suppress_output = True, 
                    max_workers=None) -> list[Any | Exception]:
    return save_call_batch(_eval_main, codes, memory_limit, timeout, suppress_output, max_workers)

def save_call_batch(fn, 
                    args, 
                    memory_limit = 512*1024*1024, 
                    timeout = 5, 
                    suppress_output = True, 
                    max_workers=None) -> list[Any | Exception]:
    """
    Executes a batch of function calls in parallel using a process pool executor.
    Args:
        fn (callable): The function to be called for each argument in the batch.
        args (Iterable): The arguments to be passed to the function.
        memory_limit (int, optional): The memory limit for each process in bytes. Defaults to 512*1024*1024.
        timeout (int, optional): The maximum time in seconds to wait for each function call to complete. Defaults to 5.
        suppress_output (bool, optional): Whether to suppress the output of each function call. Defaults to True.
        max_workers (int, optional): The maximum number of worker processes to use. Defaults to None, applies to the creation of ProcessPoolExecutor.
    Returns:
        list[Any | Exception]: A list containing the return values or exceptions raised by each function call.
    Raises:
        Exception: If an unexpected error occurs during the execution of a function call.
    """
    
    ret = []
    args_dict = {i: arg for i, arg in enumerate(args)}
    
    with tqdm(total=len(args_dict)) as pbar:
        while len(args_dict) > 0:
            timeout_accured = False
            # BE CAREFUL: shutdown the pool for all possible program exits
            executor =  ProcessPoolExecutor(max_workers=max_workers)
            futures = {executor.submit(_call_main, fn, arg, memory_limit, suppress_output) : i for i, arg in args_dict.items()}
            for future in futures.keys():
                try:
                    ret.append(future.result(timeout=timeout))
                    args_dict.pop(futures[future])
                    pbar.update(1)
                except (ExecutionException, ValueError, SyntaxError, MemoryError) as e:
                    args_dict.pop(futures[future])
                    pbar.update(1)
                    ret.append(e)
                except TimeoutError as e:
                    args_dict.pop(futures[future])
                    pbar.update(1)
                    ret.append(TimeoutError("Task timed out, this may be due to an infinite loop"))
                    # print("TimeoutError")
                    
                    # Kill executor and restart it
                    # for f in futures.keys():
                    #     f.cancel()
                    # kill_executor(executor)
                    # break
                    # IF TOO MANY TIMEOUTS, REPEATLY KILLING THE EXECUTOR WILL TAKE TOO LONG
                    timeout_accured = True
                except CancelledError as e:
                    args_dict.pop(futures[future])
                    pbar.update(1)
                    ret.append(e)
                except BrokenProcessPool as e:
                    # BrkenProcessPool is raised when the executor is shutdown
                    # most likely due to MemoryError from other tasks in the pool
                    print("BrokenProcessPool, restarting...")
                    # TODO: This is not guranteed to work, need to find a better way to handle this
                    for f in futures.keys():
                        f.cancel()
                    kill_executor(executor)
                    break
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    kill_executor(executor)
                    executor.shutdown(cancel_futures=True)
                    raise e
            if timeout_accured:
                kill_executor(executor)
            executor.shutdown(cancel_futures=True)
    return ret

# def bomb():
#     a = [0]
#     while True:
#         a.extend(a*len(a))
        
# def wait():
#     import time
#     while True:
#         time.sleep(1)
        
# def wrong():
#     a = [0,1,2]
#     return a[3]

# if __name__ == '__main__':
#     code = '1+1'
#     code2 = 'bomb()'
#     import time
#     code3 = 'wait()'
#     code4 = 'wrong()'
    
#     print(save_eval_batch([code3, code, code3, code3], 1024*1024))
#     ret = save_eval_batch([code, code3, code], 1024*1024)