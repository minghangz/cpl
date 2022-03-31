import h5py
import numpy as np

from datasets.base import BaseDataset, build_collate_data


class ActivityNet(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])
        
    def _load_frame_features(self, vid):
        with h5py.File(self.args['feature_path'], 'r') as fr:
            return np.asarray(fr[vid]['c3d_features']).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)
