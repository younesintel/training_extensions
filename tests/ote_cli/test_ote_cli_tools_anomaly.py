import pytest

from ote_cli.registry import Registry

registry = Registry('external')

args = {
        '--train-ann-file': 'tmp/anomalib/train.json',
        '--train-data-roots': 'tmp/anomalib/dataset/',
        '--val-ann-file': 'tmp/anomalib/val.json',
        '--val-data-roots': 'tmp/anomalib/dataset/',
        '--test-ann-files': 'tmp/anomalib/test.json',
        '--test-data-roots': 'tmp/anomalib/dataset/',
    }

for template in registry.filter(task_type='ANOMALY').templates:
    def test_ote_train(template=template):
        