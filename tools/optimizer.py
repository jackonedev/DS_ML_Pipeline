"""
## [MULTIPROCESSING]

Ejemplo de uso:
```
# Solo se necesita importar la función one_func_many_args
func = text_normalizer.normalize_corpus
input_data = [(X_train, stop_words), (X_test, stop_words)]
results = one_func_many_args(func, input_data)

norm_train_reviews, norm_test_reviews = results
```
"""

import multiprocessing as mp
from typing import Callable, Iterable, List, Tuple


def run_func(func: Callable, args: Tuple, results: List, index: int):
    """
    Run a function with given arguments and store the result in the shared list.

    Args:
        func (Callable): The function to be executed.
        args (Iterable): The arguments to be passed to the function.
        results (List): The shared list to store the results.
        index (int): The index at which to store the result.
    """
    print(f'Initializing "{func.__name__}" execution with index [{index}]')
    result = func(*args)
    results.append((index, result))


def sort_results(original_list):
    """
    Sort results by the index.

    Args:
        results (List): A list of tuples containing index and result.

    Returns:
        List: A sorted list of results.
    """
    sorted_list = sorted(original_list, key=lambda x: x[0])
    ordered_results = [result[1] for result in sorted_list]
    return ordered_results


def one_func_many_args(
    func: Callable, args: List[Tuple], num_workers: int = mp.cpu_count()
):
    """
    Multiprocessing, parallelism: Execute the same function with many different arguments.

    Args:
        func (Callable): The function to be executed.
        args (List[Tuple]): A list of tuples containing the arguments for each function call.
        num_workers (int, optional): The number of worker processes to use. Defaults to the number of CPUs.

    Returns:
        List: A sorted list of results from each function call.
    """

    assert isinstance(args, list), "args must be a list of arguments"
    assert isinstance(args[0], tuple), "args must be a list of tuples"

    with mp.Manager() as manager:
        results = manager.list()
        with mp.Pool(num_workers) as pool:
            for i, arg in enumerate(args):
                pool.apply_async(run_func, args=(func, arg, results, i))
            pool.close()
            pool.join()

        return sort_results(list(results))


def many_funcs_one_arg(
    funcs: List[Callable], *args: Iterable, num_workers: int = mp.cpu_count()
):
    """
    Multiprocessing, parallelism: Execute many functions with the same arguments.

    Args:
        funcs (List[Callable]): A list of functions to be executed.
        *args (Iterable): The arguments to be passed to the functions.
        num_workers (int, optional): The number of worker processes to use. Defaults to the number of CPUs.

    Returns:
        List: A sorted list of the results from executing the functions.
    """

    if len(funcs) > 4:
        print("This function will execute a maximum of 4 functions")
        raise SystemError

    with mp.Manager() as manager:
        results = manager.list()
        with mp.Pool(num_workers) as pool:
            for i, func in enumerate(funcs):
                pool.apply_async(run_func, args=(func, args, results, i))
            pool.close()
            pool.join()

        return sort_results(list(results))
