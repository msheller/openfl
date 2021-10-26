# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GaNDLFTaskRunner module."""

from functools import partialmethod
from copy import deepcopy
from typing import Iterator
from typing import Tuple

import numpy as np
import torch as pt
import torch.nn as nn
import tqdm

from .runner_pt import set_tensor_dict, get_tensor_dict, reset_opt_vars, train_results_tensorkey_dicts

from openfl.utilities import Metric
from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey

from .runner import TaskRunner
from openfl.protocols.utils import global_local_tensorkey_dicts

from GANDLF.cli.main_run import main_run
from GANDLF.parseConfig import parseConfig


class GaNDLFTaskRunner(TaskRunner):
    """GaNDLF OpenFL wrapper for Federated Learning."""

    def __init__(
            self,
            data_csv: str = None,
            config_file: str = None,
            output_dir: str = None,
            train_mode: bool = None,
            device: str = None,
            reset_prev: bool = None,
            **kwargs
    ):
        """Initialize.

        Args:

            data_csv (str): The CSV file of the training data.
            config_file (str): The YAML file of the training configuration.
            output_dir (str): The output directory.
            train_mode (bool): Whether to train or infer.
            device (str): The device type.
            reset_prev (bool): Whether the previous run will be reset or not.
            **kwargs: Additional parameters to pass to the functions
        """
        super().__init__(self, **kwargs)

        # TODO: Currently I get back end_epoch and best_loss from gandlf main_run, but do nothing
        # with them. We need to find a DB solution to persist these.

        # TODO: Complete the functionality of the crossfold trainval/test evaluation. Currently
        # Need to put starting model for training on disk before calling main_run.

        self.data_csv       = data_csv
        self.config_file    = config_file
        self.output_dir     = output_dir
        self.train_mode     = train_mode
        self.device         = device
        self.reset_prev      = reset_prev

        # This is a map of all the required tensors for each of the public
        # functions in PyTorchTaskRunner
        self.required_tensorkeys_for_function = {}

        self.training_round_completed = False

        # overwrite attribute to account for one optimizer param (in every
        # child model that does not overwrite get and set tensordict) that is
        # not a numpy array
        self.tensor_dict_split_fn_kwargs.update({
            'holdout_tensor_names': ['__opt_state_needed']
        })

        # parse config
        parameters = parseConfig(config_file)

        # check config for unsupported parameters
        self._check_config(parameters)
        # raise NotImplementedError("OpenFL doesn't yet support this amazing GaNDLF feature <feature>. Isn't Micah the worst? Sarthak, have a 666")

        # run a zero epoch "train" to create the model and optimizer objects
        gandlf_results_dict = main_run(data_csv=self.data_csv, 
                                       config_file=self.config_file, 
                                       output_dir=self.output_dir, 
                                       train_mode=self.train_mode, 
                                       device=self.device, 
                                       reset_prev=self.reset_prev, 
                                       epochs=0, 
                                       start_epoch=0,
                                       best_loss=np.inf,
                                       do_train=False,
                                       do_val=False,
                                       do_test=False)

        self.model = gandlf_results_dict['state']['model']
        self.optimizer = gandlf_results_dict['state']['optimizer']

        # TODO: For now we store some model state using class attributes but these are not
        # currently persisted for crash recover. Changes needed in OpenFL for this.
        self.best_loss = gandlf_results_dict['state']['best_loss']
        self.end_epoch = gandlf_results_dict['state']['end_epoch']

        # TODO: I would like to test and ensure best_loss is 1e7 and end_epoch is -1
        # then raise an exception otherwise that use of a best model other than
        # ones with these values for initialization is not advised, since such an initialization 
        # will be written over by the aggregator innitial model. However, I don't want 
        # to rely on 1e7 staying the default. What is best here?

    def _check_config(self, parameters):
        """
        Inspect the parameters and raise NotImplementedError if any are not supported.
        Args:
            parameters:   parameters read from the GaNDLF YAML config
            
        Returns:
            None
        """

        # TODO: Fill in additional unsupported params below
        unsupported_parameters = ["parallel_compute_command"]
        unsupported_parameters_found = []
        for key in parameters:
            if key in unsupported_parameters:
                unsupported_parameters_found.append(key)

        if ("nested_training" in parameters) and \
            (parameters["nested_training"]["validation"] > 1) or \
                (parameters["nested_training"]["testing"] > 1):
                raise ValueError(f'Cross fold validation config found in {self.__repr__()} self.config_file. Cross fold training is only supported via a train_batches call (see code for specifying config via parameters in that case.')

        if len(unsupported_parameters_found) > 0:
            raise NotImplementedError(f'Parameters: {unsupported_parameters_found} are not currently supported in OpenFL.')

    def rebuild_model(self, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """

        if self.opt_treatment == 'RESET':
            reset_opt_vars(model=self.model)
            set_tensor_dict(self.model, input_tensor_dict, with_opt_vars=False)
        elif (self.training_round_completed
              and self.opt_treatment == 'CONTINUE_GLOBAL' and not validation):
            set_tensor_dict(self.model, input_tensor_dict, with_opt_vars=True)
        else:
            set_tensor_dict(self.model, input_tensor_dict, with_opt_vars=False)

    def validate(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            kwargs:              Key word arguments passed to GaNDLF main_run

        Returns:
            global_output_dict:   Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.rebuild_model(input_tensor_dict=input_tensor_dict, validation=True)
        
        gandlf_results_dict = main_run(data_csv=self.data_csv, 
                                       config_file=self.config_file, 
                                       output_dir=self.output_dir, 
                                       train_mode=self.train_mode, 
                                       device=self.device, 
                                       epochs=1,
                                       model=self.model,
                                       optimizer=self.model.optimizer, 
                                       do_train=False,
                                       do_val=True,
                                       do_test=False 
                                       **kwargs
                                       )


        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        
        global_local_w_kwargs = ({}, {'acc': np.array(gandlf_results_dict['scores']['epoch_valid_metric'])}, {'origin': col_name,
                                                                                                              'round_num': round_num,
                                                                                                              'report': True, 
                                                                                                              'tags': tags})
        return global_local_tensorkey_dicts(global_local_w_kwargs)

    def train_batches(self, 
                      col_name, 
                      round_num, 
                      input_tensor_dict,
                      use_tqdm=False, 
                      epochs=1,
                      crossfold_test=False,
                      crossfold_test_data_csv=None,
                      crossfold_val_n=None,
                      crossfold_test_n=None, 
                      **kwargs):
        """Train batches. Here epochs are specified as opposed to batches.

        Args:
            col_name                : Name of the collaborator
            round_num               : What round is it
            input_tensor_dict       : Required input tensors (for model)
            use_tqdm (bool)         : Use tqdm to print a progress bar (Default=True)
            epochs                  : The number of epochs to train
            crossfold_test          : Whether or not to use cross fold trainval/test
                                    to evaluate the quality of the model under fine tuning
                                    (this uses a separate prameter to pass in the data and 
                                    config used)
            crossfold_test_data_csv : Data csv used to define data used in crossfold test.
                                      This csv does not itself define the folds, just
                                      defines the total data to be used.
            crossfold_val_n         : number of folds to use for the train,val level of the nested crossfold.
            corssfold_test_n        : number of folds to use for the trainval,test level of the nested crossfold.
            kwargs                  : Key word arguments passed to GaNDLF main_run

        Returns:
            global_output_dict      : Tensors to send back to the aggregator
            local_output_dict       : Tensors to maintain in the local TensorDB
        """

        
        
        # invoke GaNDLF main run
        # TODO: Here the values used for start_epoch and best_loss are taken from class attributes
        # and not persisted for crash recover. We need to change this once we have a way for OpenFL to track this state. 
        if crossfold_test:
            # TODO: We will need to use model checkpoints on disk here rather than passing model, as otherwise it will keep one fold of traininng and
            # pass it as init for the next fold of training. So we'll want to use the file system (ie pass model=None) in this
            # case: store all results to disk and return no model (user can collect results from file). However the first model
            # will need to be manually put on disk before calling main_run?
            if (crossfold_val_n is None) or (crossfold_test_n is None) or (crossfold_test_data_csv is None):
                raise ValueError('When crossfold_test is True, all of crossfold_test_data_csv, crossfold_val_n, and crossfold_test_n need to be provided.')
        
            #NOTE: Here the return is None, as all outputs go to disk (not even a model is returned)
            
            gandlf_results_dict = main_run(data_csv=crossfold_test_data_csv, 
                                        config_file=self.config_file, 
                                        output_dir=self.output_dir, 
                                        train_mode=self.train_mode, 
                                        device=self.device, 
                                        epochs=epochs,
                                        model=self.model,
                                        start_epoch=0,
                                        best_loss=1e7,
                                        optimizer=self.model.optimizer, 
                                        do_train=True,
                                        do_val=True,
                                        do_test=True,
                                        crossfold_val_n=crossfold_val_n,
                                        corssfold_test_n=crossfold_test_n, 
                                        **kwargs
                                        )
            return
        else:
            gandlf_results_dict = main_run(data_csv=self.data_csv, 
                                        config_file=self.config_file, 
                                        output_dir=self.output_dir, 
                                        train_mode=self.train_mode, 
                                        device=self.device, 
                                        epochs=epochs,
                                        model=self.model,
                                        optimizer=self.model.optimizer, 
                                        do_train=True,
                                        do_val=False,
                                        do_test=False, 
                                        **kwargs
                                        )
            self.model = gandlf_results_dict['state']['model']
            self.optimizer = gandlf_results_dict['state']['optimizer']
            self.end_epoch = gandlf_results_dict['state']['end_epoch']
            self.best_loss = gandlf_results_dict['state']['best_loss']
            
            # Output metric tensors (scalar)
            output_metric_dict = {'last_epoch_train_loss': np.array(gandlf_results_dict['scores']['epoch_train_loss'])}
            
            # output model tensors (Doesn't include TensorKey)
            output_model_dict = self.model.get_tensor_dict(with_opt_vars=True)
            
            global_tensorkey_dict, local_tensorkey_dict = train_results_tensorkey_dicts(output_model_dict=output_model_dict, 
                                                                                        output_metric_dict=output_metric_dict, 
                                                                                        logger=self.logger, 
                                                                                        tensor_dict_split_fn_kwargs=self.tensor_dict_split_fn_kwargs, 
                                                                                        col_name=col_name, 
                                                                                        round_num=round_num)
            # Update the required tensors if they need to be pulled from the
            # aggregator
            # TODO this logic can break if different collaborators have different
            # roles between rounds.
            # For example, if a collaborator only performs validation in the first
            # round but training in the second, it has no way of knowing the
            # optimizer state tensor names to request from the aggregator because
            # these are only created after training occurs. A work around could
            # involve doing a single epoch of training on random data to get the
            # optimizer names, and then throwing away the model.
            if self.opt_treatment == 'CONTINUE_GLOBAL':
                self.initialize_tensorkeys_for_functions(with_opt_vars=True)

            # This will signal that the optimizer values are now present,
            # and can be loaded when the model is rebuilt
            self.train_round_completed = True

            # Return global_tensor_dict, local_tensor_dict
            return global_tensorkey_dict, local_tensorkey_dict

    def _get_weights_names(self, with_opt_vars=False):
        # Gets information regarding tensor model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or
        # simple assignment is better
        # for now, state dict gives us names which is good
        # FIXME: do both and sanity check each time?

        state = self.model.state_dict().keys()

        if with_opt_vars:
            opt_state = _get_optimizer_state(self.model.optimizer)
            state += opt_state.keys()

        return state

    def get_optimizer(self):
        """Get the optimizer of this instance."""
        return self.optimizer

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called \
        as part of a task. By default, this is just all of the layers and \
        optimizer of the model.

        Args:
            func_name

        Returns:
            list : [TensorKey]
        """
        if func_name == 'validate':
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """Set the required tensors for all publicly accessible task methods.

        By default, this is just all of the layers and optimizer of the model.
        Custom tensors should be added to this function.

        Args:
            None

        Returns:
            None
        """
        # TODO there should be a way to programmatically iterate through
        #  all of the methods in the class and declare the tensors.
        # For now this is done manually

        output_model_dict = get_tensor_dict(model=self.model, with_opt_vars=with_opt_vars)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )
        if not with_opt_vars:
            global_model_dict_val = global_model_dict
            local_model_dict_val = local_model_dict
        else:
            output_model_dict = get_tensor_dict(model=self.model, with_opt_vars=False)
            global_model_dict_val, local_model_dict_val = split_tensor_dict_for_holdouts(
                self.logger,
                output_model_dict,
                **self.tensor_dict_split_fn_kwargs
            )

        self.required_tensorkeys_for_function['train_batches'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict]
        self.required_tensorkeys_for_function['train_batches'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict]

        self.required_tensorkeys_for_function['train'] = [
            TensorKey(
                tensor_name, 'GLOBAL', 0, False, ('model',)
            ) for tensor_name in global_model_dict
        ]
        self.required_tensorkeys_for_function['train'] += [
            TensorKey(
                tensor_name, 'LOCAL', 0, False, ('model',)
            ) for tensor_name in local_model_dict
        ]

        # Validation may be performed on local or aggregated (global) model,
        # so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        # TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function['validate']['apply=local'] = [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('trained',))
            for tensor_name in {
                **global_model_dict_val,
                **local_model_dict_val
            }]
        self.required_tensorkeys_for_function['validate']['apply=global'] = [
            TensorKey(tensor_name, 'GLOBAL', 0, False, ('model',))
            for tensor_name in global_model_dict_val
        ]
        self.required_tensorkeys_for_function['validate']['apply=global'] += [
            TensorKey(tensor_name, 'LOCAL', 0, False, ('model',))
            for tensor_name in local_model_dict_val
        ]

    def load_native(self, filepath, model_state_dict_key='model_state_dict',
                    optimizer_state_dict_key='optimizer_state_dict', **kwargs):
        """
        Load model and optimizer states from a pickled file specified by \
        filepath. model_/optimizer_state_dict args can be specified if needed. \
        Uses pt.load().

        Args:
            filepath (string)                 : Path to pickle file created
                                                by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state dict
                                                in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = pt.load(filepath)
        self.model.load_state_dict(pickle_dict[model_state_dict_key])
        self.model.optimizer.load_state_dict(pickle_dict[optimizer_state_dict_key])

    def save_native(self, filepath, model_state_dict_key='model_state_dict',
                    optimizer_state_dict_key='optimizer_state_dict', **kwargs):
        """
        Save model and optimizer states in a picked file specified by the \
        filepath. model_/optimizer_state_dicts are stored in the keys provided. \
        Uses pt.save().

        Args:
            filepath (string)                 : Path to pickle file to be
                                                created by pt.save().
            model_state_dict_key (string)     : key for model state dict
                                                in pickled file.
            optimizer_state_dict_key (string) : key for optimizer state
                                                dict in picked file.
            kwargs                            : unused

        Returns:
            None
        """
        pickle_dict = {
            model_state_dict_key: self.model.state_dict(),
            optimizer_state_dict_key: self.model.optimizer.state_dict()
        }
        pt.save(pickle_dict, filepath)

def _derive_opt_state_dict(opt_state_dict):
    """Separate optimizer tensors from the tensor dictionary.

    Flattens the optimizer state dict so as to have key, value pairs with
    values as numpy arrays.
    The keys have sufficient info to restore opt_state_dict using
    expand_derived_opt_state_dict.

    Args:
        opt_state_dict: The optimizer state dictionary

    """
    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict['state']) == 0:
        derived_opt_state_dict['__opt_state_needed'] = 'false'
        return derived_opt_state_dict

    derived_opt_state_dict['__opt_state_needed'] = 'true'

    # Using one example state key, we collect keys for the corresponding
    # dictionary value.
    example_state_key = opt_state_dict['param_groups'][0]['params'][0]
    example_state_subkeys = set(
        opt_state_dict['state'][example_state_key].keys()
    )

    # We assume that the state collected for all params in all param groups is
    # the same.
    # We also assume that whether or not the associated values to these state
    # subkeys is a tensor depends only on the subkey.
    # Using assert statements to break the routine if these assumptions are
    # incorrect.
    for state_key in opt_state_dict['state'].keys():
        assert example_state_subkeys == set(opt_state_dict['state'][state_key].keys())
        for state_subkey in example_state_subkeys:
            assert (isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                pt.Tensor)
                == isinstance(
                    opt_state_dict['state'][state_key][state_subkey],
                    pt.Tensor))

    state_subkeys = list(opt_state_dict['state'][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a
    # tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(
                opt_state_dict['state'][example_state_key][state_subkey],
                pt.Tensor
        ):
            state_subkey_tags.append('istensor')
        else:
            state_subkey_tags.append('')
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))

    # Forming the flattened dict, using a concatenation of group index,
    # subindex, tag, and subkey inserted into the flattened dict key -
    # needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict['param_groups']):
        for idx, param_id in enumerate(group['params']):
            for subkey, tag in state_subkeys_and_tags:
                if tag == 'istensor':
                    new_v = opt_state_dict['state'][param_id][
                        subkey].cpu().numpy()
                else:
                    new_v = np.array(
                        [opt_state_dict['state'][param_id][subkey]]
                    )
                derived_opt_state_dict[f'__opt_state_{group_idx}_{idx}_{tag}_{subkey}'] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing
    # original opt_state_dict structure
    derived_opt_state_dict['__opt_group_lengths'] = np.array(
        nb_params_per_group
    )

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    """Expand the optimizer state dictionary.

    Takes a derived opt_state_dict and creates an opt_state_dict suitable as
    input for load_state_dict for restoring optimizer state.

    Reconstructing state_subkeys_and_tags using the example key
    prefix, "__opt_state_0_0_", certain to be present.

    Args:
        derived_opt_state_dict: Optimizer state dictionary

    Returns:
        dict: Optimizer state dictionary
    """
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith('__opt_state_0_0_'):
            stripped_key = key[16:]
            if stripped_key.startswith('istensor_'):
                this_tag = 'istensor'
                subkey = stripped_key[9:]
            else:
                this_tag = ''
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {'param_groups': [], 'state': {}}
    nb_params_per_group = list(
        derived_opt_state_dict.pop('__opt_group_lengths').astype(np.int)
    )

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = [f'{group_idx}_{idx}' for idx in range(nb_params)]
        opt_state_dict['param_groups'].append({'params': these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict['state'][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = f'__opt_state_{this_id}_{tag}_{subkey}'
                if tag == 'istensor':
                    new_v = pt.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currrently supported optimizers) the subkey
                    # should be 'step' and the length of array should be one.
                    assert subkey == 'step'
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict['state'][this_id][subkey] = new_v

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict


def _get_optimizer_state(optimizer):
    """Return the optimizer state.

    Args:
        optimizer
    """
    opt_state_dict = deepcopy(optimizer.state_dict())

    # Optimizer state might not have some parts representing frozen parameters
    # So we do not synchronize them
    param_keys_with_state = set(opt_state_dict['state'].keys())
    for group in opt_state_dict['param_groups']:
        local_param_set = set(group['params'])
        params_to_sync = local_param_set & param_keys_with_state
        group['params'] = sorted(params_to_sync)

    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _set_optimizer_state(optimizer, device, derived_opt_state_dict):
    """Set the optimizer state.

    Args:
        optimizer:
        device:
        derived_opt_state_dict:

    """
    temp_state_dict = expand_derived_opt_state_dict(
        derived_opt_state_dict, device)

    # FIXME: Figure out whether or not this breaks learning rate
    #  scheduling and the like.
    # Setting default values.
    # All optimizer.defaults are considered as not changing over course of
    # training.
    for group in temp_state_dict['param_groups']:
        for k, v in optimizer.defaults.items():
            group[k] = v

    optimizer.load_state_dict(temp_state_dict)


def to_cpu_numpy(state):
    """Send data to CPU as Numpy array.

    Args:
        state

    """
    # deep copy so as to decouple from active model
    state = deepcopy(state)

    for k, v in state.items():
        # When restoring, we currently assume all values are tensors.
        if not pt.is_tensor(v):
            raise ValueError('We do not currently support non-tensors '
                             'coming from model.state_dict()')
        # get as a numpy array, making sure is on cpu
        state[k] = v.cpu().numpy()
    return state
