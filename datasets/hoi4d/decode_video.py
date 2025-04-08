import os
import subprocess
import sys
from tqdm import tqdm

def decode_video(root):
    with open('./pour_seq_names.txt', 'r') as f:
        rgb_list = [os.path.join(root, 'HOI4D_release', i.strip().replace('-', '/'), 'align_rgb') for i in f.readlines()]

    for rgb in tqdm(rgb_list):
        depth = rgb.replace('align_rgb','align_depth').replace('HOI4D_release', 'HOI4D_depth_video')
        rgb_video = os.path.join(rgb, "image.mp4")
        depth_video = os.path.join(depth, "depth_video.avi")

        cmd =  """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15 -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(rgb_video, rgb, "jpg")

        print(cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            log.info(err.decode())

        cmd = """ ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15 -qscale:v 2 {}/%05d.{} -loglevel quiet """.format(depth_video, depth, "png")
        print(cmd)

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            log.info(err.decode())

if __name__ == '__main__':
    decode_video('./data')
