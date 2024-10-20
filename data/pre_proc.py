import os
import time
import subprocess
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='./',
                    help='the dir of the video data')
parser.add_argument('--fps', type=int, default=24,
                    help='output fps for videos')
parser.add_argument('--n_workers', type=int, default=6,
                    help='number of workers')

args = parser.parse_args()

def video_to_frames(ytid, save_dir='./',fps=24):
    save_path = os.path.join(save_dir)
    video_file = os.path.join(save_path, ytid[:-4] + '.mp4')
    frames_dir = os.path.join(save_path, 'frames/', ytid[:-4])
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
    command = ['ffmpeg', '-i', video_file, '-vf', f'fps={fps}', '-q:v', '5', f'{frames_dir}/%08d.jpg']
    status = subprocess.call(command)
    return save_path, frames_dir, video_file, status


def video_to_audio(ytid, save_dir='./'):
    save_path = os.path.join(save_dir)
    video_file = os.path.join(save_path, ytid[:-4] + '.mp4')
    audio_dir = os.path.join(save_path, 'audio/', ytid[:-4])

    command = ['ffmpeg', '-i', video_file, audio_dir + '.mp3']
    status = subprocess.call(command)
    return save_path, audio_dir, video_file, status


def meta_to_data(meta, save_dir='./',fps=12, transcript_only=False):

    save_path, frames_dir, video_file, status_v = video_to_frames(meta['link'],save_dir=save_dir, fps=fps)
    status_a, _, _, _ = video_to_audio(meta['link'], save_dir=save_dir)
    if status_v != 0:
        print('video to frames status:', status_v)
        return meta, False
    if status_a != 0:
        print('video to frames status:', status_a)
        return meta, False

    return meta, True

if __name__ == '__main__':
    # '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/MELD.Raw/train_splits'
    # '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/MELD.Raw/dev_splits_complete'
    # '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/MELD.Raw/output_repeated_splits_test'

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir,  'frames/')):
        os.mkdir(os.path.join(args.save_dir,  'frames/'))
    if not os.path.exists(os.path.join(args.save_dir,  'audio/')):
        os.mkdir(os.path.join(args.save_dir,  'audio/'))

    playlist_movies = [{'link': link} for link in os.listdir(args.save_dir)]

    global progress_bar
    progress_bar = tqdm(total=len(playlist_movies))

    def update_progress_bar(_):
        progress_bar.update()
    
    pool = mp.Pool(args.n_workers)

    for meta in playlist_movies:
        time.sleep(0.2)
        pool.apply_async(partial(meta_to_data, meta, save_dir=args.save_dir, fps=args.fps, transcript_only=True), callback=update_progress_bar)

    pool.close()
    pool.join()