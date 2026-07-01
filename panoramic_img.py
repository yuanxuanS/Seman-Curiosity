import os

import cv2
import numpy as np
import quaternion

from src.finetune.dataset_utils import SampleLoader


DATA_PTH = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/graph/episodes_data"
OUTPUT_DIR = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/panorama"
PANORAMA_SIZE = (1024, 512)

RGB_TYPES = [
       "rgb", "rgb_30", "rgb_60", "rgb_90", "rgb_120", "rgb_150",
       "rgb_180", "rgb_210", "rgb_240", "rgb_270", "rgb_300", "rgb_330",
]
DEPTH_TYPES = [
       "depth", "depth_30", "depth_60", "depth_90", "depth_120", "depth_150",
       "depth_180", "depth_210", "depth_240", "depth_270", "depth_300", "depth_330",
]
OTHER_TYPES = ["position", ]


def _prepare_rgb_images(sample_data):
       images = []
       for rgb_key in RGB_TYPES:
              image = np.array(sample_data[rgb_key].data, copy=True)  # Convert BGR to RGB
            #   if image.ndim == 3 and image.shape[2] == 3:
            #          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #   elif image.ndim == 2:
            #          image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
              images.append(image)
       return images


def _prepare_depth_images(sample_data):
       images = []
       for depth_key in DEPTH_TYPES:
              depth = np.array(sample_data[depth_key].data, copy=True)
              depth = np.squeeze(depth)
              if depth.dtype != np.uint8:
                     depth_min = float(np.min(depth))
                     depth_max = float(np.max(depth))
                     if depth_max > depth_min:
                            depth = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
                     else:
                            depth = np.zeros_like(depth, dtype=np.uint8)
              depth_color = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
              images.append(depth_color)
       return images


def _stitch_images(images):
       stitcher = cv2.Stitcher_create()
       status, pano = stitcher.stitch(images)
       if status == cv2.Stitcher_OK:
              return pano
       raise RuntimeError(f"Stitching failed with status {status}")


def _concat_images_horizontally(images):
       if not images:
              raise ValueError("No images provided for panorama construction")

       base_height = images[0].shape[0]
       resized_images = []
       for image in images:
              if image.shape[0] != base_height:
                     scale = base_height / float(image.shape[0])
                     new_width = max(1, int(round(image.shape[1] * scale)))
                     image = cv2.resize(image, (new_width, base_height), interpolation=cv2.INTER_AREA)
              resized_images.append(image)

       return np.concatenate(resized_images, axis=1)


def _build_panorama(images):
       try:
              return _stitch_images(images)
       except cv2.error:
              return _concat_images_horizontally(images)
       except RuntimeError:
              return _concat_images_horizontally(images)


def _resize_panorama(pano, size):
       if size is None:
              return pano
       width, height = size
       return cv2.resize(pano, (width, height), interpolation=cv2.INTER_AREA)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sampler = SampleLoader(DATA_PTH, glbstep=True)
    inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  

    #    env = 0
    #    episode = 1
    #    frame = 1
    #    glbstep = 0
    pos = {}
    for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):

        sample_data = sampler.get_sample_multimodality(
                env, episode, step, RGB_TYPES + DEPTH_TYPES + OTHER_TYPES, glbstep
        )

        rgb_pano = _resize_panorama(_build_panorama(_prepare_rgb_images(sample_data)), PANORAMA_SIZE)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"rgb_env{env}_epi{episode}_glb{glbstep}_step{step}.png"), rgb_pano)

        depth_images = _prepare_depth_images(sample_data)
        try:
               depth_pano_raw = _stitch_images(depth_images)
        except Exception as e:
               print(f"Depth stitching failed for env{env} epi{episode} glb{glbstep} step{step}: {e}")
        else:
               depth_pano = _resize_panorama(depth_pano_raw, PANORAMA_SIZE)
               cv2.imwrite(os.path.join(OUTPUT_DIR, f"depth_env{env}_epi{episode}_glb{glbstep}_step{step}.png"), depth_pano)

        # print(f"Saved RGB panorama to {os.path.join(OUTPUT_DIR, 'rgb_panorama.png')}")
        # print(f"Saved depth panorama to {os.path.join(OUTPUT_DIR, 'depth_panorama.png')}")

        # pos[(env, episode, glbstep, step)] = sample_data['position'].position
        
    # write positions to text file
    # out_pth = os.path.join(OUTPUT_DIR, "positions.txt")
    # print(f"Writing positions to {out_pth}")
    # with open(out_pth, "w") as f:
    #     f.write("env,episode,glbstep,step,px,py,pz,qw,qx,qy,qz\n")
    #     for (env, episode, glbstep, step), pose in pos.items():
    #         position = np.asarray(pose)
    #         # orientation is stored in the sample_data for the same key; reload to get orientation
    #         line_vals = [env, episode, glbstep, step] + position.tolist()
    #         f.write(",".join([str(x) for x in line_vals]) + "\n")

if __name__ == "__main__":
       main()