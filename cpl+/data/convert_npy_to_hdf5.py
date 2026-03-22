import numpy as np
import h5py
from tqdm import tqdm
import glob

if __name__ == '__main__':
    feat_path = './i3d_finetuned/*'
    h5_file = h5py.File('./i3d_features.hdf5', 'w')
    
    for vid_path in tqdm(glob.glob(feat_path)):
        vid = vid_path.split('.npy')[0].split('/')[-1]
        feats = np.load(vid_path)
        h5_file.create_dataset(vid, data=feats.squeeze(), compression="gzip")
    h5_file.close()
