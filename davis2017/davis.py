import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image

class VOSDataset(object):

    VOID_LABEL = 255

    def __init__(self, root, img_folder='JPEGImages', mask_folder='Annotations', imagesets_path=None,  sequences='all'):

        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        """

        self.root = root  # ./datasets/DAVIS
        self.img_path = os.path.join(self.root, img_folder)
        self.mask_path = os.path.join(self.root, mask_folder)

        if imagesets_path is not None:
            self.imagesets_path = os.path.join(self.root, imagesets_path)
        else:
            self.imagesets_path = None

        self._check_directories()

        if sequences == "all":

            if imagesets_path is not None:

                print("Using imageset file: " , self.imagesets_path)
                with open(self.imagesets_path, 'r') as f:
                    tmp = f.readlines()
                sequences_names = [x.strip() for x in tmp]

            else:

                print("No imageset file found! Including all sequences inside this dataset.")
                sequences_names = os.listdir(self.mask_path)


        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]


        self.sequences = defaultdict(dict)

        # Load masks and images
        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'VOS dataset not found in the specified directory: ', self.root)
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f'Image dir are not found in the specified directory: ', self.img_path)
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'GT annotations dir are not found in the specified directory: ', self.mask_path)
        if self.imagesets_path is not None:
            if not os.path.exists(self.imagesets_path):
                raise FileNotFoundError(f'Imageset file is not in the specified directory: ', self.imagesets_path)

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq