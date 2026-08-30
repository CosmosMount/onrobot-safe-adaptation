from os import sep as slash
from rlx.algorithm import Algorithm
from rlx.environment import Environment


_algorithms = {}


def extract_algorithm_name_from_file(file_name):
    return file_name.split(f"algorithms{slash}")[1].split(f"{slash}__init__.py")[0].replace(slash, ".")


def register_algorithm(name, get_default_config, get_model_class, general_properties):
    _algorithms[name] = Algorithm(name, get_default_config, get_model_class, general_properties)


def get_algorithm_config(algorithm_name):
    return _algorithms[algorithm_name].get_default_config(algorithm_name)


def get_algorithm_model_class(algorithm_name):
    return _algorithms[algorithm_name].get_model_class


def get_algorithm_general_properties(algorithm_name):
    return _algorithms[algorithm_name].general_properties

_environments = {}


def extract_environment_name_from_file(file_name):
    return file_name.split(f"environments{slash}")[1].split(f"{slash}__init__.py")[0].replace(slash, ".")


def register_environment(name, get_default_config, create_train_and_eval_env, general_properties):
    _environments[name] = Environment(name, get_default_config, create_train_and_eval_env, general_properties)


def get_environment_config(environment_name):
    return _environments[environment_name].get_default_config(environment_name)


def get_environment_create_train_and_eval_env(environment_name):
    return _environments[environment_name].create_train_and_eval_env


def get_environment_general_properties(environment_name):
    return _environments[environment_name].general_properties