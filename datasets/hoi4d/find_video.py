import os
import argparse
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--action", type=str)
    return parser.parse_args()

def main():
    args = parse_args()

    video_root_dir = "data/HOI4D_release"
    video_path_list = glob(f"{video_root_dir}/*/*/*/*/*/*/*/*/image.mp4")
    output_dir = f"{args.action}_output_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{args.action}_seq_names.txt", 'w') as f:
        for video_idx in tqdm(range(len(video_path_list))):
            video_path = video_path_list[video_idx]
            object_name = video_path.split('/')[4]
            task_name = video_path.split('/')[8]
            seq_name = '-'.join(video_path.split('/')[2:-2])
        
            if args.action == "pour":
                if object_name == "C2" and task_name == "T5":
                    os.system(f'cp {video_path} {output_dir}/{seq_name}.mp4')
                    f.writelines(seq_name + '\n')

if __name__ == '__main__':
    main()
