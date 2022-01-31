<<<<<<< HEAD
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import  cv2
import keras
import skimage


import io
import traceback
app = Flask(__name__)

global classes
classes = ["printer", "computer monitor", "computer keyboard"]
import warnings
warnings.filterwarnings('ignore')

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances


"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


from PIL import Image
import numpy as np



@app.route('/')
def upload_form():
    return render_template("singlefile_index.html")

@app.route('/display_result_imgs', methods=['POST'])
def PredictB64():
    try:
        #save it to tmp dir
        from test import detect
        files = request.files.getlist("files")
        base64_l = []
        print(files)
        for f in files:
            print("f",f)
            file = f.read()

            npimg = np.fromstring(file, np.uint8)
            # convert numpy array to image
            decoded = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            print("before decode",decoded)
            frame = detect(decoded)
            print(frame)
            print(type(frame))

            rawBytes = io.BytesIO()
            print("the image",frame)
            frame.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
            print(type(img_base64))
            img_str = img_base64.decode('utf-8')

            base64_l.append(img_str)

        return render_template("display_result.html",data=base64_l)
    except Exception as e:
        print("+++++++++++++++++++++++++++")
        print(traceback.format_exc())
        return "err"



if __name__ == '__main__':
=======
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import  cv2
import keras
import skimage


import io
import traceback
app = Flask(__name__)

global classes
classes = ["printer", "computer monitor", "computer keyboard"]
import warnings
warnings.filterwarnings('ignore')

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances


"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


from PIL import Image
import numpy as np



@app.route('/')
def upload_form():
    return render_template("singlefile_index.html")

@app.route('/display_result_imgs', methods=['POST'])
def PredictB64():
    try:
        #save it to tmp dir
        from test import detect
        files = request.files.getlist("files")
        base64_l = []
        print(files)
        for f in files:
            print("f",f)
            file = f.read()

            npimg = np.fromstring(file, np.uint8)
            # convert numpy array to image
            decoded = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            print("before decode",decoded)
            frame = detect(decoded)
            print(frame)
            print(type(frame))

            rawBytes = io.BytesIO()
            print("the image",frame)
            frame.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
            print(type(img_base64))
            img_str = img_base64.decode('utf-8')

            base64_l.append(img_str)

        return render_template("display_result.html",data=base64_l)
    except Exception as e:
        print("+++++++++++++++++++++++++++")
        print(traceback.format_exc())
        return "err"



if __name__ == '__main__':
>>>>>>> 39c9c927bd4b42f0ffe9f28acd15dcaa8c3888b8
    app.run()