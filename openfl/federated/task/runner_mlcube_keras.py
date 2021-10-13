import os
import subprocess
import json
import yaml

import numpy as np

from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey
from .runner_keras import KerasTaskRunner


class MLCubeKerasTaskRunner(KerasTaskRunner):
    """A wrapper for MLCube Tensorflow Tasks."""

    def __init__(self,
                 mlcube_dir='.',
                 mlcube_runner_type='docker',
                 mlcube_model_init_path=None,
                 mlcube_model_in_path=None,
                 mlcube_model_out_path=None,
                 parameters=None,
                 parameters_path=None,
                 training_sample_count_key='training_samples',
                 evaluation_sample_count_key='evaluation_samples',
                 **kwargs):
        """Initialize.

        Args:
            TBD
        """
        super().__init__(**kwargs)

        self.opt_treatment = 'RESET'

        self.training_sample_count_key = training_sample_count_key
        self.evaluation_sample_count_key = evaluation_sample_count_key

        self.mlcube_dir = mlcube_dir
        self.mlcube_runner_type = mlcube_runner_type

        self.mlcube_model_init_path = os.path.join(mlcube_dir, mlcube_model_init_path)
        self.mlcube_model_in_path   = os.path.join(mlcube_dir, mlcube_model_in_path)
        self.mlcube_model_out_path  = os.path.join(mlcube_dir, mlcube_model_out_path)

        # Need to call dummy train task to load initial model
#         self.dummy_train()

        self.parameters_path = os.path.join(mlcube_dir, parameters_path)
        self.parameters = parameters
        
        self.load_native(self.mlcube_model_init_path)

        self.required_tensorkeys_for_function = {}
        self.initialize_tensorkeys_for_functions()
        
    def train(self, col_name, round_num, input_tensor_dict, epochs, **kwargs):
        """Perform training for a specified number of epochs."""
#         if 'metrics' not in kwargs:
#             raise KeyError('metrics must be included in kwargs')
#         param_metrics = kwargs['metrics']

        self.rebuild_model(round_num, input_tensor_dict)

        # 1. Save model in native format
        self.save_native(self.mlcube_model_in_path)

        # update the "epochs" parameter value
        self.parameters['epochs'] = epochs

        # Write the paramaters.yaml file
        with open(self.parameters_path, 'w') as f:
            yaml.dump(self.parameters, f)

        # 2. Call MLCube train task
        platform_yaml = os.path.join(self.mlcube_dir, 'platforms', '{}.yaml'.format(self.mlcube_runner_type))
        task_yaml = os.path.join(self.mlcube_dir, 'run', 'train.yaml')
        proc = subprocess.run(["mlcube_docker",
                               "run",
                               "--mlcube={}".format(self.mlcube_dir),
                               "--platform={}".format(platform_yaml),
                               "--task={}".format(task_yaml)])

        # 3. Load model from native format
        self.load_native(self.mlcube_model_out_path)

        metrics = self.load_metrics(os.path.join(self.mlcube_dir, 'workspace', 'metrics', 'train_metrics.json'))

        # set the training data size
        sample_count = int(metrics.pop(self.training_sample_count_key))
        self.data_loader.set_train_data_size(sample_count)

        # 5. Convert to tensorkeys

        # output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric_name, origin, round_num, True, ('metric',)
            ): np.array(
                    metrics[metric_name]
                ) for metric_name in metrics}

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }
        # the train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # this ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        #  roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator
        # because these are only created after training occurs.
        # A work around could involve doing a single epoch of training
        # on random data to get the optimizer names, and then throwing away
        # the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

    def validate(self, col_name, round_num, input_tensor_dict, **kwargs):
        """
        Run the trained model on validation data; report results.

        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc,
         precision, f1_score, etc.)
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        # 1. Save model in native format
        self.save_native(self.mlcube_model_in_path)

        # 2. Call MLCube validate task
        platform_yaml = os.path.join(self.mlcube_dir, 'platforms', '{}.yaml'.format(self.mlcube_runner_type))
        task_yaml = os.path.join(self.mlcube_dir, 'run', 'evaluate.yaml')
        proc = subprocess.run(["mlcube_docker",
                               "run",
                               "--mlcube={}".format(self.mlcube_dir),
                               "--platform={}".format(platform_yaml),
                               "--task={}".format(task_yaml)])

        # 3. Load any metrics
        metrics = self.load_metrics(os.path.join(self.mlcube_dir, 'workspace', 'metrics', 'evaluate_metrics.json'))

        # set the validation data size
        sample_count = int(metrics.pop(self.evaluation_sample_count_key))
        self.data_loader.set_valid_data_size(sample_count)

        # 4. Convert to tensorkeys
    
        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        output_tensor_dict = {
            TensorKey(
                metric_name, origin, round_num, True, tags
            ): np.array(metrics[metric_name])
            for metric_name in metrics
        }

        return output_tensor_dict, {}

    def load_metrics(self, filepath):
        """
        Load metrics from JSON file
        """
        ### 
        with open(filepath) as json_file:
            metrics = json.load(json_file)
        return metrics
