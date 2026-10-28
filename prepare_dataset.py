import os
import io
import re
import shutil
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import cv2
import torch

ZIP_PATH = './data/Driving_Risk_Dataset.zip'
DATA_DIR = './data'
BACKUP_DIR = './data/toy_backup'
PRETRAINED_DIR = './pretrained'
PRETRAINED_URL = 'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch4_window7_224.pth'
PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, 'swin_base_patch4_window7_224.pth')

def backup_toy_data():
    """Backup existing toy dataset if present."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    toy_files = [
        'kine_train.pkl', 'kine_val.pkl',
        'label_train.pkl', 'label_val.pkl',
        'video_train.pkl', 'video_val.pkl'
    ]
    for tf in toy_files:
        src = os.path.join(DATA_DIR, tf)
        dst = os.path.join(BACKUP_DIR, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Backing up {tf} -> {dst}")
            shutil.copy2(src, dst)

def extract_and_prepare_kinematics(z, df):
    """Extract full kinematics dataset and split into train / val / test."""
    print("\n--- 1. Processing Kinematics Data (2,000 samples) ---")
    npz_bytes = io.BytesIO(z.read('kinematics_tensors.npz'))
    npz = np.load(npz_bytes, allow_pickle=True)
    
    kinematics = npz['kinematics'] # shape (2000, 3, 51)
    labels = npz['labels']         # shape (2000,)
    event_ids = list(npz['event_ids'])
    
    # Map each event_id to its split
    split_map = dict(zip(df['event_id'], df['split']))
    splits = np.array([split_map.get(eid, 'train') for eid in event_ids])
    
    train_mask = (splits == 'train')
    val_mask = (splits == 'val')
    test_mask = (splits == 'test')
    
    kine_train = kinematics[train_mask]
    label_train = labels[train_mask]
    kine_val = kinematics[val_mask]
    label_val = labels[val_mask]
    kine_test = kinematics[test_mask]
    label_test = labels[test_mask]
    
    print(f"Train samples: {len(kine_train)} | Shape: {kine_train.shape} | Labels: {np.bincount(label_train)}")
    print(f"Val samples:   {len(kine_val)} | Shape: {kine_val.shape} | Labels: {np.bincount(label_val)}")
    print(f"Test samples:  {len(kine_test)} | Shape: {kine_test.shape} | Labels: {np.bincount(label_test)}")
    
    # Save standard training & validation kinematics files
    pd.to_pickle(kine_train.astype(np.float32), os.path.join(DATA_DIR, 'kine_train.pkl'))
    pd.to_pickle(kine_val.astype(np.float32), os.path.join(DATA_DIR, 'kine_val.pkl'))
    pd.to_pickle(kine_test.astype(np.float32), os.path.join(DATA_DIR, 'kine_test.pkl'))
    
    pd.to_pickle(label_train.astype(np.int64), os.path.join(DATA_DIR, 'label_train.pkl'))
    pd.to_pickle(label_val.astype(np.int64), os.path.join(DATA_DIR, 'label_val.pkl'))
    pd.to_pickle(label_test.astype(np.int64), os.path.join(DATA_DIR, 'label_test.pkl'))
    
    print("Saved kinematics .pkl files to data/ successfully.")
    return event_ids, kinematics, labels, split_map

def extract_and_prepare_video(z, df, event_ids, kinematics, labels, split_map, num_frames=77, img_size=(224, 224)):
    """
    Extract video frames for available events (100 events, 4 frames each),
    interpolate to num_frames (default 77), and save matched video + kinematics.
    """
    print(f"\n--- 2. Processing Video Data (Interpolating to {num_frames} frames) ---")
    vframe_dirs = set([f.split('/')[1] for f in z.namelist() if f.startswith('video_frames/') and len(f.split('/')) > 1])
    
    norm_to_vdir = {}
    for vd in vframe_dirs:
        m = re.match(r'EVENT_([A-Za-z_]+)_(\d+)', vd)
        if m:
            cname, num = m.groups()
            norm_id = f'EVENT_{cname}_{int(num):04d}'
            norm_to_vdir[norm_id] = vd
            
    eid_to_idx = {eid: i for i, eid in enumerate(event_ids)}
    valid_eids = [eid for eid in norm_to_vdir if eid in eid_to_idx]
    
    print(f"Total matched video events: {len(valid_eids)}")
    
    # Sort for deterministic ordering
    valid_eids.sort()
    
    video_list = []
    kine_list = []
    label_list = []
    split_list = []
    
    # Interpolation frame indices
    sample_indices = np.round(np.linspace(0, 3, num_frames)).astype(int)
    
    for eid in valid_eids:
        raw_vdir = norm_to_vdir[eid]
        # Load 4 frames: frame_01.jpg .. frame_04.jpg
        frames = []
        for i in range(1, 5):
            fname = f'video_frames/{raw_vdir}/frame_{i:02d}.jpg'
            try:
                img_data = z.read(fname)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if (img.shape[0], img.shape[1]) != img_size:
                    img = cv2.resize(img, img_size)
                # Normalize to [0, 1] float32
                frames.append(img.astype(np.float32) / 255.0)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                break
                
        if len(frames) != 4:
            continue
            
        frames = np.stack(frames, axis=0) # (4, H, W, C)
        # Resample / interpolate along time dimension
        seq = frames[sample_indices]      # (num_frames, H, W, C)
        # Permute to (C, D, H, W) as expected by SwinTransformer3D: (3, num_frames, 224, 224)
        seq = np.transpose(seq, (3, 0, 1, 2))
        
        orig_idx = eid_to_idx[eid]
        kine_item = kinematics[orig_idx]
        lbl_item = labels[orig_idx]
        s_item = split_map.get(eid, 'train')
        
        video_list.append(seq)
        kine_list.append(kine_item)
        label_list.append(lbl_item)
        split_list.append(s_item)
        
    video_arr = np.stack(video_list, axis=0) # (N, 3, num_frames, 224, 224)
    kine_arr = np.stack(kine_list, axis=0)   # (N, 3, 51)
    label_arr = np.array(label_list, dtype=np.int64)
    split_arr = np.array(split_list)
    
    train_mask = (split_arr == 'train')
    # Use val + test for validation on video (67 train, 33 val)
    val_mask = ~train_mask
    
    vid_train = video_arr[train_mask]
    vid_val = video_arr[val_mask]
    vlabel_train = label_arr[train_mask]
    vlabel_val = label_arr[val_mask]
    
    kine_vtrain = kine_arr[train_mask]
    kine_vval = kine_arr[val_mask]
    
    print(f"Video Train: {vid_train.shape} | Labels: {np.bincount(vlabel_train)}")
    print(f"Video Val:   {vid_val.shape} | Labels: {np.bincount(vlabel_val)}")
    
    # Save video dataset files
    pd.to_pickle(vid_train.astype(np.float32), os.path.join(DATA_DIR, 'video_train.pkl'))
    pd.to_pickle(vid_val.astype(np.float32), os.path.join(DATA_DIR, 'video_val.pkl'))
    pd.to_pickle(vlabel_train.astype(np.int64), os.path.join(DATA_DIR, 'video_label_train.pkl'))
    pd.to_pickle(vlabel_val.astype(np.int64), os.path.join(DATA_DIR, 'video_label_val.pkl'))
    
    # Also save matched kinematics for the multimodal events
    pd.to_pickle(kine_vtrain.astype(np.float32), os.path.join(DATA_DIR, 'kine_video_train.pkl'))
    pd.to_pickle(kine_vval.astype(np.float32), os.path.join(DATA_DIR, 'kine_video_val.pkl'))
    
    print("Saved video .pkl files to data/ successfully.")

def download_pretrained_weights():
    """Download OpenMMLab Video Swin pretrained checkpoint if not already present."""
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    if os.path.exists(PRETRAINED_PATH):
        print(f"\nPretrained checkpoint already exists at {PRETRAINED_PATH} (size: {os.path.getsize(PRETRAINED_PATH)/(1024**2):.1f} MB)")
        return
        
    print(f"\n--- 3. Downloading Pretrained Weights ---")
    print(f"Downloading from {PRETRAINED_URL} to {PRETRAINED_PATH}...")
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if count % 200 == 0 or percent == 100:
                print(f"\rProgress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
                
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED_PATH, reporthook=reporthook)
    print(f"\nDownloaded checkpoint to {PRETRAINED_PATH} successfully!")

def verify_and_patch_checkpoint():
    """Verify downloaded checkpoint keys for compatibility with video_swin_transformer."""
    if not os.path.exists(PRETRAINED_PATH):
        return
    print("\nVerifying checkpoint structure...")
    ckpt = torch.load(PRETRAINED_PATH, map_location='cpu')
    keys = list(ckpt.keys())
    print("Checkpoint keys:", keys)
    # If checkpoint has 'state_dict' but not 'model', alias it so video_swin_transformer's checkpoint['model'] works!
    if 'state_dict' in ckpt and 'model' not in ckpt:
        print("Aliasing ckpt['state_dict'] -> ckpt['model'] for compatibility...")
        ckpt['model'] = ckpt['state_dict']
        torch.save(ckpt, PRETRAINED_PATH)
        print("Checkpoint saved with 'model' key.")
    print("Checkpoint verified successfully.")

def main():
    print("==================================================")
    print("  DUST Dataset Preparation & Checkpoint Setup")
    print("==================================================")
    
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Cannot find {ZIP_PATH}!")
        
    backup_toy_data()
    
    with zipfile.ZipFile(ZIP_PATH) as z:
        df = pd.read_csv(io.BytesIO(z.read('metadata.csv')))
        event_ids, kinematics, labels, split_map = extract_and_prepare_kinematics(z, df)
        extract_and_prepare_video(z, df, event_ids, kinematics, labels, split_map, num_frames=77)
        
    download_pretrained_weights()
    verify_and_patch_checkpoint()
    
    print("\n==================================================")
    print("  Dataset preparation completed successfully!")
    print("==================================================")

if __name__ == '__main__':
    main()
