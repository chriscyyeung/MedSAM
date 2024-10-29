"""
Implements an OpenIGTLink client that receives ultrasound (pyigtl.ImageMessage) and sends prediction/segmentation (pyigtl.ImageMessage).
Transform messages (pyigtl.TransformMessage) are also received and sent to the server, but the device name is changed by replacing Image to Prediction.
This is done to ensure that the prediction is visualized in the same position as the ultrasound image.

Arguments:
    model: Path to the torchscript file you intend to use for segmentation. The model must be a torchscript model that takes a single image as input and returns a single image as output.
    input device name: This is the device name the client is listening to
    output device name: The device name the client outputs to
    host: Server's IP the client connects to.
    input port: Port used for receiving data from the PLUS server over OpenIGTLink
    output port: Port used for sending data to Slicer over OpenIGTLink
"""

import argparse
import time
import yaml
import logging
import cv2
import pyigtl
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from torchvision.transforms import Normalize

from sam2.build_sam import build_sam2

ROOT = Path(__file__).parent.resolve()

PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                # points=None, 
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


# Parse command line arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model checkpoint.")
    parser.add_argument("--model-cfg", type=str, help="Path to SAM2 config.")
    parser.add_argument("--pretrain-model", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=1024, help="Model input size.")
    parser.add_argument("--scanconversion-config", type=str, help="Path to scan conversion config (.yaml) file. Optional.")
    parser.add_argument("--input-device-name", type=str, default="Image_Image")
    parser.add_argument("--output-device-name", type=str, default="Prediction")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=18944)
    parser.add_argument("--output-port", type=int, default=18945)
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file. Optional.")
    return parser


def preprocess_input(image, input_size, scanconversion_config=None, x_cart=None, y_cart=None):
    if scanconversion_config is not None:
        # Scan convert image from curvilinear to linear
        num_samples = scanconversion_config["num_samples_along_lines"]
        num_lines = scanconversion_config["num_lines"]
        converted_image = np.zeros((1, num_lines, num_samples))
        converted_image[0, :, :] = map_coordinates(image[0, :, :], [x_cart, y_cart], order=1, mode='constant', cval=0.0)
    
    # resize to model input size
    converted_image = cv2.resize(converted_image[0, :, :], (input_size, input_size))  # default is bilinear
    converted_image = np.repeat(converted_image[np.newaxis, ...], 3, axis=0)  # (3, 1024, 1024)
    converted_image = torch.from_numpy(converted_image).unsqueeze(0)  # add batch dimension

    # normalize pixel values
    converted_image = converted_image / 255.0
    converted_image = Normalize(PIXEL_MEAN, PIXEL_STD)(converted_image)
    converted_image = converted_image.float()
    print(torch.min(converted_image), torch.max(converted_image))

    # generate random bounding box coordinates (act as no prompt)
    H, W = converted_image.shape[2:]
    box_width = np.random.randint(10, W // 2)
    box_height = np.random.randint(10, H // 2)
    x_min = np.random.randint(0, W - box_width)
    y_min = np.random.randint(0, H - box_height)
    x_max = x_min + box_width
    y_max = y_min + box_height
    bboxes = np.array([x_min, y_min, x_max, y_max])
    bboxes = bboxes[np.newaxis, ...]
    
    return converted_image, bboxes


def postprocess_prediction(prediction, original_size, scanconversion_config=None, vertices=None, weights=None, mask_array=None):
    prediction = torch.sigmoid(prediction)
    prediction = prediction.squeeze().detach().cpu().numpy() * 255
    
    if scanconversion_config is not None:
        # resize to scan conversion size first
        num_samples = scanconversion_config["num_samples_along_lines"]
        num_lines = scanconversion_config["num_lines"]
        prediction = cv2.resize(prediction, (num_lines, num_samples))
        # Scan convert prediction from linear to curvilinear
        prediction = scan_convert(prediction, scanconversion_config, vertices, weights)
        if mask_array is not None:
            prediction = prediction * mask_array
    else:
        prediction = cv2.resize(prediction, (original_size[2], original_size[1]))

    prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    return prediction


def scan_conversion_inverse(scanconversion_config):
    """
    Compute cartesian coordianates for inverse scan conversion.
    Mapping from curvilinear image to a rectancular image of scan lines as columns.
    The returned cartesian coordinates can be used to map the curvilinear image to a rectangular image using scipy.ndimage.map_coordinates.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Rerturns:
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.

    Example:
        >>> x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        >>> scan_converted_image = map_coordinates(ultrasound_data[0, :, :, 0], [x_cart, y_cart], order=3, mode="nearest")
        >>> scan_converted_segmentation = map_coordinates(segmentation_data[0, :, :, 0], [x_cart, y_cart], order=0, mode="nearest")
    """

    # Create sampling points in polar coordinates

    initial_radius = np.deg2rad(scanconversion_config["angle_min_degrees"])
    final_radius = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_start_px = scanconversion_config["radius_start_pixels"]
    radius_end_px = scanconversion_config["radius_end_pixels"]

    theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, scanconversion_config["num_samples_along_lines"]),
                           np.linspace(radius_start_px, radius_end_px, scanconversion_config["num_lines"]))

    # Convert the polar coordinates to cartesian coordinates

    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][0]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][1]

    return x_cart, y_cart


def scan_interpolation_weights(scanconversion_config):
    image_size = scanconversion_config["curvilinear_image_size"]

    x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
    triangulation = Delaunay(np.vstack((x_cart.flatten(), y_cart.flatten())).T)

    grid_x, grid_y = np.mgrid[0:image_size, 0:image_size]
    simplices = triangulation.find_simplex(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
    vertices = triangulation.simplices[simplices]

    X = triangulation.transform[simplices, :2]
    Y = np.vstack((grid_x.flatten(), grid_y.flatten())).T - triangulation.transform[simplices, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]

    return vertices, weights


def scan_convert(linear_data, scanconversion_config, vertices, weights):
    """
    Scan convert a linear image to a curvilinear image.

    Args:
        linear_data (np.ndarray): Linear image to be scan converted.
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        scan_converted_image (np.ndarray): Scan converted image.
    """
    
    z = linear_data.flatten()
    zi = np.einsum('ij,ij->i', np.take(z, vertices), weights)

    image_size = scanconversion_config["curvilinear_image_size"]
    return zi.reshape(image_size, image_size)


def curvilinear_mask(scanconversion_config):
    """
    Generate a binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        mask_array (np.ndarray): Binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.
    """
    angle1 = 90.0 + (scanconversion_config["angle_min_degrees"])
    angle2 = 90.0 + (scanconversion_config["angle_max_degrees"])
    center_rows_px = scanconversion_config["center_coordinate_pixel"][0]
    center_cols_px = scanconversion_config["center_coordinate_pixel"][1]
    radius1 = scanconversion_config["radius_start_pixels"]
    radius2 = scanconversion_config["radius_end_pixels"]
    image_size = scanconversion_config["curvilinear_image_size"]

    mask_array = np.zeros((image_size, image_size), dtype=np.int8)
    mask_array = cv2.ellipse(mask_array, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle1, angle2, 1, -1)
    mask_array = cv2.circle(mask_array, (center_cols_px, center_rows_px), radius1, 0, -1)
    # Convert mask_array to uint8
    mask_array = mask_array.astype(np.uint8)

    # Repaint the borders of the mask to zero to allow erosion from all sides
    mask_array[0, :] = 0
    mask_array[:, 0] = 0
    mask_array[-1, :] = 0
    mask_array[:, -1] = 0
    
    # Erode mask by 10 percent of the image size to remove artifacts on the edges
    erosion_size = int(0.1 * image_size)
    mask_array = cv2.erode(mask_array, np.ones((erosion_size, erosion_size), np.uint8), iterations=1)
    
    return mask_array


def main(args):
    """
    Runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
    the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
    """
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    input_client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.input_port)
    output_server = pyigtl.OpenIGTLinkServer(port=args.output_port, local_server=False)
    model = None

    # Initialize timer and counters for profiling

    start_time = time.perf_counter()
    preprocess_counter = 0
    preprocess_total_time = 0
    inference_counter = 0
    inference_total_time = 0
    postprocess_counter = 0
    postprocess_total_time = 0
    total_counter = 0
    total_time = 0
    image_message_counter = 0
    transform_message_counter = 0

    # Load pytorch model
    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model if Path(args.model).is_absolute() else f'{str(ROOT)}/{args.model}'
    sam2_model = build_sam2(args.model_cfg, args.pretrain_model, device=device, apply_postprocessing=True)
    model = MedSAM2(model=sam2_model)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    logging.info("Model loaded")

    # If scan conversion is enabled, compute x_cart, y_cart, vertices, and weights for conversion and interpolation
    if args.scanconversion_config:
        logging.info("Loading scan conversion config...")
        with open(args.scanconversion_config, "r") as f:
            scanconversion_config = yaml.safe_load(f)
        x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        logging.info("Scan conversion config loaded")
    else:
        scanconversion_config = None
        x_cart = None
        y_cart = None
        logging.info("Scan conversion config not found")

    if x_cart is not None and y_cart is not None:
        vertices, weights = scan_interpolation_weights(scanconversion_config)
        # mask_array = curvilinear_mask(scanconversion_config)
        mask_array = None
    else:
        vertices = None
        weights = None
        mask_array = None

    while True:
        # Print average inference time
        if time.perf_counter() - start_time > 1.0:
            logging.info("--------------------------------------------------")
            logging.info(f"Image messages received:   {image_message_counter}")
            logging.info(f"Transform messages received:   {transform_message_counter}")
            if preprocess_counter > 0:
                avg_preprocess_time = round((preprocess_total_time / preprocess_counter) * 1000, 1)
                logging.info(f"Average preprocess time:  {avg_preprocess_time} ms")
            if inference_counter > 0:
                avg_inference_time = round((inference_total_time / inference_counter) * 1000, 1)
                logging.info(f"Average inference time:   {avg_inference_time} ms")
            if postprocess_counter > 0:
                avg_postprocess_time = round((postprocess_total_time / postprocess_counter) * 1000, 1)
                logging.info(f"Average postprocess time: {avg_postprocess_time} ms")
            if total_counter > 0:
                avg_total_time = round((total_time / total_counter) * 1000, 1)
                logging.info(f"Average total time:       {avg_total_time} ms")
            start_time = time.perf_counter()
            preprocess_counter = 0
            preprocess_total_time = 0
            inference_counter = 0
            inference_total_time = 0
            postprocess_counter = 0
            postprocess_total_time = 0
            total_counter = 0
            total_time = 0
            image_message_counter = 0
            transform_message_counter = 0
        
        # Receive messages from server
        messages = input_client.get_latest_messages()
        for message in messages:
            if message.device_name == args.input_device_name:  # Image message
                image_message_counter += 1
                total_start_time = time.perf_counter()
        
                if model is None:
                    logging.error("Model not loaded. Exiting...")
                    break
                
                # Resize image to model input size
                orig_img_size = message.image.shape

                # Preprocess input
                preprocess_start_time = time.perf_counter()
                image, bboxes = preprocess_input(message.image, args.image_size, scanconversion_config, x_cart, y_cart)
                image = image.to(device)
                preprocess_total_time += time.perf_counter() - preprocess_start_time
                preprocess_counter += 1

                # Run inference
                inference_start_time = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(image, bboxes)
                inference_total_time += time.perf_counter() - inference_start_time
                inference_counter += 1

                # Postprocess prediction
                postprocess_start_time = time.perf_counter()
                prediction = postprocess_prediction(prediction, orig_img_size, scanconversion_config, vertices, weights, mask_array)
                postprocess_total_time += time.perf_counter() - postprocess_start_time
                postprocess_counter += 1

                image_message = pyigtl.ImageMessage(prediction, device_name=args.output_device_name)
                output_server.send_message(image_message, wait=True)
                
                total_time += time.perf_counter() - total_start_time
                total_counter += 1

            if message.message_type == "TRANSFORM" and "Image" in message.device_name:  # Image transform message
                transform_message_counter += 1
                output_tfm_name = message.device_name.replace("Image", "Pred")
                tfm_message = pyigtl.TransformMessage(message.matrix, device_name=output_tfm_name)
                output_server.send_message(tfm_message, wait=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
