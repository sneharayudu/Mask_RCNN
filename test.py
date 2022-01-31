from PIL import Image
import numpy as np
import tensorflow as tf
import  cv2
import keras
import skimage
import io
import traceback

classes = ["printer", "computer monitor", "computer keyboard"]
import warnings
warnings.filterwarnings('ignore')

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn import visualize

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "cfg_new"
    # number of classes (background + Inlet)
    NUM_CLASSES = len(classes) + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = PredictionConfig()
config.display()

#intializing config info
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# # load model weights
model.load_weights("./printer_monitor_keyboard/weights/weights.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


# def display_instances(image, boxes, masks, ids, names, scores):
#     n_instances = boxes.shape[0]
#     colors = visualize.random_colors(n_instances)
#     if not n_instances:
#         print('NO INSTANCES TO DISPLAY')
#     else:
#         assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
#     for i, color in enumerate(colors):
#         if not np.any(boxes[i]):
#             continue
#         y1, x1, y2, x2 = boxes[i]
#         print('++++++++++++++++++++++++++++++++++++++++++++++++')
#         print(i)
#         print(ids)
#         print(scores)
#         label = names[ids[i]]
#         score = scores[i] if scores is not None else None
#         caption = '{} {:.2f}'.format(label, score) if score else label
#         mask = masks[:, :, i]
#
#         image = visualize.apply_mask(image, mask, color)
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#         image = cv2.putText(
#             image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
#         )
#
#     return image

def detect(decoded):
    try:
        class_names = ["no data","printer", "computer monitor", "computer keyboard"]
        print("decoaded",decoded)
        r = model.detect([decoded], verbose=1)[0]
        print(r['class_ids'])
        masked_frame = display_instances(decoded, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'])
        try:
            cv2.imwrite("filename.jpg", masked_frame)
        except Exception as e:
            print("img write err"+str(e))
        # # You may need to convert the color.
        # masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(masked_frame)
        return masked_frame
    except Exception as e:
        print("======================")
        print(traceback.format_exc())
        return str(traceback.format_exc())