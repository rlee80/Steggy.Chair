from flask import Flask, request, jsonify
import os

app = Flask(__name__)
upload_folder = "images"

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


from models import *
from utils.utils import *
from utils.datasets import *

import datetime
import sys
from PIL import Image



import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
@app.route("/")
def index():
    return "Test"

@app.route("/image_proc")
def image_proc():
    # File is known as image
    upload_folder = "data/samples"
    file = request.files['image']

    path = os.path.join(upload_folder, file.filename)
    print(path)
    file.save(path)

    data = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet("config/yolov3.cfg", img_size=416).to(device)

    if "weights/yolov3.weights".endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights("weights/yolov3.weights")
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load("weights/yolov3.weights"))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder("data/samples", img_size=416),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    classes = load_classes("data/coco.names")  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.8, 0.4)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, 416, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:


                print("Prediction is " + classes[int(cls_pred)])
                prediction = classes[int(cls_pred)]

                data['prediction'] = prediction

                if prediction == "person":
                    data['x1'] = float(x1)
                    data['y1'] = float(y1)

                    data['x2'] = float(x2)
                    data['y2'] = float(y2)

                    data['midline'] = float((x1+x2)/2)
                    data['image_midline'] = len(img[0])/2


            os.remove(os.path.join(upload_folder, file.filename))
            return jsonify(data)



if __name__ == "__main__":
    app.run()
