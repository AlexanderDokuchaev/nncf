"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import tempfile
from functools import partial
from pathlib import Path
import pytest
import tensorflow as tf

from tests.common.helpers import TEST_ROOT
from tests.tensorflow.helpers import get_coco_dataset_builders
from tests.tensorflow.test_models import SequentialModel, SequentialModelNoInput

from examples.tensorflow.classification import main as cls_main
from examples.tensorflow.object_detection import main as od_main
from examples.tensorflow.segmentation import train as seg_train
from examples.tensorflow.segmentation import evaluation as seg_eval
from examples.tensorflow.common.model_loader import AVAILABLE_MODELS
from examples.tensorflow.common.prepare_checkpoint import main as prepare_checkpoint_main

od_main.get_dataset_builders = partial(get_coco_dataset_builders, train=True, validation=True)
seg_train.get_dataset_builders = partial(get_coco_dataset_builders, train=True, calibration=True)
seg_eval.get_dataset_builders = partial(get_coco_dataset_builders, validation=True)

AVAILABLE_MODELS.update({
    'SequentialModel': SequentialModel,
    'SequentialModelNoInput': SequentialModelNoInput
})


class ConfigFactory:
    """Allows to modify config file before test run"""

    def __init__(self, base_config, config_path):
        self.config = base_config
        self.config_path = str(config_path)

    def serialize(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)
        return self.config_path

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value


def convert_to_argv(args):
    return ' '.join(key if val is None else '{} {}'.format(key, val) for key, val in args.items()).split()


SAMPLE_TYPES = [
    'classification',
    'object_detection',
    'segmentation',
]


SAMPLES = {
    'classification': {
        'train-test-export': cls_main.main
    },
    'object_detection': {
        'train-test-export': od_main.main
    },
    'segmentation': {
        'train': seg_train.main,
        'test-export': seg_eval.main
    },
}


DATASETS = {
    'classification': [('cifar10', 'tfds'), ('cifar10', 'tfds'), ('cifar10', 'tfds')],
    'object_detection': [('coco2017', 'tfrecords')],
    'segmentation': [('coco2017', 'tfrecords')],
}


TEST_CONFIG_ROOT = TEST_ROOT.joinpath('tensorflow', 'data', 'configs')
CONFIGS = {
    'classification': [
        TEST_CONFIG_ROOT.joinpath('resnet50_cifar10_magnitude_sparsity_int8.json'),
        TEST_CONFIG_ROOT.joinpath('sequential_model_cifar10_magnitude_sparsity_int8.json'),
        TEST_CONFIG_ROOT.joinpath('sequential_model_no_input_cifar10_magnitude_sparsity_int8.json'),
    ],
    'object_detection': [
        TEST_CONFIG_ROOT.joinpath('retinanet_coco2017_magnitude_sparsity_int8.json'),
    ],
    'segmentation': [
        TEST_CONFIG_ROOT.joinpath('mask_rcnn_coco2017_magnitude_sparsity_int8.json'),
    ],
}


BATCH_SIZE_PER_GPU = {
    'classification': [32, 32, 32],
    'object_detection': [1],
    'segmentation': [1],
}


def get_global_batch_size():
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    coeff = num_gpus if num_gpus else 1
    global_batch_size = {}
    for sample_type, batch_sizes in BATCH_SIZE_PER_GPU.items():
        global_batch_size[sample_type] = [coeff * bs for bs in batch_sizes]
    return global_batch_size


GLOBAL_BATCH_SIZE = get_global_batch_size()


DATASET_PATHS = {
    'classification': {
        x: lambda dataset_root, dataset_name=x:
        os.path.join(dataset_root, dataset_name) if dataset_root else
        os.path.join(tempfile.gettempdir(), dataset_name)
        for x, _ in DATASETS['classification']
    },
    'object_detection': {
        'coco2017': lambda dataset_root: TEST_ROOT.joinpath('tensorflow', 'data', 'mock_datasets', 'coco2017')
    },
    'segmentation': {
        'coco2017': lambda dataset_root: TEST_ROOT.joinpath('tensorflow', 'data', 'mock_datasets', 'coco2017')
    },
}


def get_sample_fn(sample_type, modes):
    variants = []
    for key in SAMPLES[sample_type].keys():
        supported_modes = set(key.split('-'))
        if set(modes).issubset(supported_modes):
            variants.append(key)

    if len(variants) != 1:
        raise Exception('Can not choose a function for given arguments')

    return SAMPLES[sample_type][variants[0]]


def generate_config_params():
    config_params = []
    for sample_id, sample_type in enumerate(SAMPLE_TYPES):
        config_paths, batch_sizes = CONFIGS[sample_type], GLOBAL_BATCH_SIZE[sample_type]
        dataset_names, dataset_types = zip(*DATASETS[sample_type])

        for params_id, params in enumerate(zip(config_paths, dataset_names, dataset_types, batch_sizes)):
            config_params.append((sample_type, *params, '{}_{}'.format(sample_id, params_id)))
    return config_params


def generate_id(value):
    sample_type, config_path, dataset_name, dataset_type, batch_size, _ = value
    filename = config_path.name
    return '-'.join([sample_type, filename, dataset_name, dataset_type, str(batch_size)])


CONFIG_PARAMS = generate_config_params()


@pytest.fixture(params=CONFIG_PARAMS, ids=generate_id)
def _config(request, dataset_dir):
    sample_type, config_path, dataset_name, dataset_type, batch_size, tid = request.param
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)

    with config_path.open() as f:
        jconfig = json.load(f)

    if 'checkpoint_save_dir' in jconfig.keys():
        del jconfig['checkpoint_save_dir']

    jconfig['dataset'] = dataset_name
    jconfig['dataset_type'] = dataset_type

    return {
        'sample_type': sample_type,
        'nncf_config': jconfig,
        'model_name': jconfig['model'],
        'dataset_path': dataset_path,
        'batch_size': batch_size,
        'tid': tid
    }


@pytest.fixture(scope='module')
def _case_common_dirs(tmp_path_factory):
    return {
        'checkpoint_save_dir': str(tmp_path_factory.mktemp('models'))
    }


def test_model_eval(_config, tmp_path):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    args = {
        '--mode': 'test',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size']
    }
    main = get_sample_fn(_config['sample_type'], modes=['test'])
    main(convert_to_argv(args))


@pytest.mark.dependency(name='tf_test_model_train')
def test_model_train(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    args = {
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 1,
        '--checkpoint-save-dir': checkpoint_save_dir
    }

    if _config['sample_type'] != 'segmentation':
        args['--mode'] = 'train'

    main = get_sample_fn(_config['sample_type'], modes=['train'])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.dependency(depends=['tf_test_model_train'])
def test_trained_model_eval(_config, tmp_path, _case_common_dirs):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])
    args = {
        '--mode': 'test',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--resume': ckpt_path
    }

    main = get_sample_fn(_config['sample_type'], modes=['test'])
    main(convert_to_argv(args))


@pytest.mark.dependency(depends=['tf_test_model_train'])
def test_resume(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(str(tmp_path), 'models')
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    args = {
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 2,
        '--checkpoint-save-dir': checkpoint_save_dir,
        '--resume': ckpt_path
    }

    if _config['sample_type'] != 'segmentation':
        args['--mode'] = 'train'

    main = get_sample_fn(_config['sample_type'], modes=['train'])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.dependency(depends=['tf_test_model_train'])
def test_trained_model_resume_train_test_export_last_ckpt(_config, tmp_path, _case_common_dirs):
    if _config['sample_type'] == 'segmentation':
        pytest.skip()

    checkpoint_save_dir = os.path.join(str(tmp_path), 'models')
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    export_path = os.path.join(str(tmp_path), 'model.pb')
    args = {
        '--mode': 'train test export',
        '--data': _config['dataset_path'],
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--batch-size': _config['batch_size'],
        '--epochs': 2,
        '--checkpoint-save-dir': checkpoint_save_dir,
        '--resume': ckpt_path,
        '--to-frozen-graph': export_path
    }

    main = get_sample_fn(_config['sample_type'], modes=['train', 'test', 'export'])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)
    assert os.path.exists(export_path)


FORMATS = [
    'frozen-graph',
    'saved-model',
    'h5'
]


def get_export_model_name(export_format):
    model_name = 'model'
    if export_format == 'frozen-graph':
        model_name = 'model.pb'
    elif export_format == 'h5':
        model_name = 'model.h5'
    return model_name


@pytest.mark.dependency(depends=['tf_test_model_train'])
@pytest.mark.parametrize('export_format', FORMATS, ids=FORMATS)
def test_export_with_resume(_config, tmp_path, export_format, _case_common_dirs):
    config_factory = ConfigFactory(_config['nncf_config'], tmp_path / 'config.json')
    ckpt_path = os.path.join(_case_common_dirs['checkpoint_save_dir'], _config['tid'])

    if export_format == 'saved-model':
        compression_config = _config['nncf_config'].get('compression', {})
        if isinstance(compression_config, dict):
            compression_config = [compression_config]
        for config in compression_config:
            if config.get('algorithm', '') == 'quantization':
                pytest.skip()

    if _config['sample_type'] == 'segmentation' and export_format == 'h5':
        pytest.skip('The {} sample does not support export to {} format.'.format(_config['sample_type'],
                                                                                 export_format))

    export_path = os.path.join(str(tmp_path), get_export_model_name(export_format))
    args = {
        '--mode': 'export',
        '--config': config_factory.serialize(),
        '--log-dir': tmp_path,
        '--resume': ckpt_path,
        '--to-{}'.format(export_format): export_path,
    }

    main = get_sample_fn(_config['sample_type'], modes=['export'])
    main(convert_to_argv(args))

    model_path = os.path.join(export_path, 'saved_model.pb') \
        if export_format == 'saved-model' else export_path
    assert os.path.exists(model_path)


def get_prepare_checkpoint_configs():
    supported_model_types = ['object_detection', 'segmentation']
    config_params = []
    for sample_type in supported_model_types:
        config_paths, batch_sizes = CONFIGS[sample_type], GLOBAL_BATCH_SIZE[sample_type]
        dataset_names, dataset_types = zip(*DATASETS[sample_type])

        for config_path, dataset_name, dataset_type, batch_size in \
                zip(config_paths, dataset_names, dataset_types, batch_sizes):
            dataset_path = DATASET_PATHS[sample_type][dataset_name](None)

            with config_path.open() as f:
                jconfig = json.load(f)

            if 'checkpoint_save_dir' in jconfig.keys():
                del jconfig['checkpoint_save_dir']

            jconfig['dataset'] = dataset_name
            jconfig['dataset_type'] = dataset_type
            config_params.append((sample_type, config_path, jconfig, dataset_path, batch_size))
    return config_params


@pytest.mark.parametrize('sample_type,config_path,config_eval,dataset_path,batch_size',
                         get_prepare_checkpoint_configs(),
                         ids=[x[0] for x in get_prepare_checkpoint_configs()])
def test_prepare_checkpoint(sample_type, config_path, config_eval, dataset_path, batch_size, tmp_path):
    # Keep default soft_device_placement state
    default_soft_device_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(True)
    checkpoint_save_dir = tmp_path
    log_dir = tempfile.mkdtemp()
    args = {
        '--model-type': sample_type,
        '--config': config_path,
        '--checkpoint-save-dir': checkpoint_save_dir,
        '--resume': tempfile.mkdtemp(),
    }

    prepare_checkpoint_main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)
    config_factory = ConfigFactory(config_eval, Path(tempfile.gettempdir()) / 'config.json')
    args = {
        '--mode': 'test',
        '--data': dataset_path,
        '--config': config_factory.serialize(),
        '--log-dir': log_dir,
        '--batch-size': batch_size,
        '--resume': checkpoint_save_dir
    }

    main = get_sample_fn(sample_type, modes=['test'])
    main(convert_to_argv(args))
    # Restore default soft_device_placement state
    tf.config.set_soft_device_placement(default_soft_device_placement)
