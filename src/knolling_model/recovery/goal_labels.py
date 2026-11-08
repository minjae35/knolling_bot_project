"""Source/instruction-only typed goal supervision over immutable v3 records."""
import torch

from knolling_dataset.schema import digest

SCHEMA = 'recovery_goal_labels_v1'
VOCAB = dict(axis=('horizontal', 'vertical'), region=('full', 'left', 'right'),
             gap=(.015, .020, .030), group=('none', 'color'), sort=('asc', 'desc'))
FIELDS = dict(axis='axis_alignment', region='region', gap='gap_m', group='group_by', sort='sort_direction')
CONTRACT = dict(schema=SCHEMA, vocabulary=VOCAB)


def goal_labels(record, object_ids):
    ins = record['instruction']
    keep = ins['keep_all']
    if type(keep) is not bool or not set(ins['selected_ids']).issubset(object_ids):
        raise ValueError('invalid goal selection')
    result = {'goal_keep': torch.tensor(float(keep)),
              'goal_selected': torch.tensor([not keep and i in ins['selected_ids'] for i in object_ids], dtype=torch.float32)}
    for key, values in VOCAB.items():
        value = ins[FIELDS[key]]
        if not keep and value not in values:
            raise ValueError(f'unsupported goal {key}: {value}')
        result['goal_'+key] = torch.tensor(-100 if keep else values.index(value))
    return result


class GoalDataset(torch.utils.data.Dataset):
    """Labels stay in supervision; never added to inference inputs or records."""
    def __init__(self, dataset):
        self.dataset, self.records, self.pairs = dataset, dataset.records, dataset.pairs
        self.root = dataset.root
        self.fingerprint = digest([dataset.fingerprint, CONTRACT])
        for record in self.records:
            goal_labels(record, sorted(o['object_id'] for o in record['source']['objects']))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample['supervision'] = dict(sample['supervision'], **goal_labels(self.records[index], sample['metadata']['object_ids']))
        return sample
