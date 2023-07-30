import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from PIL import Image
from sklearn.metrics import confusion_matrix

mpl.use('agg')
def sobel_detection(img, kSize):
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kSize)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kSize)
  grad = np.sqrt(sobelx**2 + sobely**2)
  # Combine x and y edges
  grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

  return grad_norm


def apply_otsu_threshold(gray_image):
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def overlay_images(rgb_image, grayscale_image, color):
  # Convert images to numpy arrays if they are not already
  if isinstance(rgb_image, Image.Image):
      rgb_array = np.array(rgb_image)
  else:
      rgb_array = rgb_image
  if len(grayscale_image.shape) == 2 or grayscale_image.shape[2] == 1:
    # Convert image to 8-bit depth
    img_8bit = cv2.convertScaleAbs(grayscale_image)
    # Convert grayscale image to RGB image
    grayscale_array_rgb  = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

    overlaid_image = grayscale_array_rgb.copy()
  else:
    overlaid_image = grayscale_image.copy()

  mask = (rgb_array > [0,0,0]).any(axis=-1)
  overlaid_image[mask] = color

  return overlaid_image

def color_contour(grad_norm, blended_img2):

  # Threshold the edges
  thresh = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  # Perform contour detection
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  mask_green_contour = np.zeros(blended_img2.shape)
  # Create polygon from contour points if contours are found
  if contours:
    # Create polygon from contour points
    polygon = np.concatenate(contours)

    # Check if each point is inside the polygon
    for y in range(blended_img2.shape[0]):
        for x in range(blended_img2.shape[1]):
            point = (x, y)
            pixel_value = blended_img2[y, x]

            if cv2.pointPolygonTest(polygon, point, False) >= 0 and np.array_equal(pixel_value, [255, 0, 0]):
                # Color point and polygon
                cv2.drawContours(mask_green_contour, contours, -1, (0, 255, 0), 1)

    # Check if the mask has any non-zero pixels within the contour region
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      roi = mask_green_contour[y:y+h, x:x+w]
      if np.count_nonzero(roi):
        cv2.fillPoly(mask_green_contour, contours, (0, 255, 0))
      else:
        continue

    return mask_green_contour


def calculate_metrics(painted_img, mask,name_file, save_dir=f"{os.getenv('STORAGE_PATH')}/train_again_try/output_sobel_model_landsat8_testGOES18/"):
  # Extract the green band from the painted image
  painted_green = painted_img[:, :, 1].flatten()
  painted_green[painted_green == 255] = 1
  # Flatten the mask
  mask = mask.flatten()

  # Calculate True Positives, True Negatives, False Positives, False Negatives
  tn, fp, fn, tp = confusion_matrix(mask, painted_green, labels=[0, 1]).ravel()
  # Calculate accuracy
  accuracy = (tp + tn) / (tp + tn + fp + fn)

  # Calculate precision
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = 0.0

  # Calculate recall (sensitivity)
  if tp + fn > 0:
    recall = tp / (tp + fn)
  else:
    recall = 0.0

  # Calculate F1 score
  if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
  else:
    f1_score = 0.0

  # Calculate specificity
  if tn + fp > 0:
    specificity = tn / (tn + fp)
  else:
    specificity = 0.0

  # Calculate balanced accuracy
  balanced_accuracy = (recall + specificity) / 2

  # Create the confusion matrix as a heatmap
  cm = np.array([[tn, fp], [fn, tp]])
  ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    # Add titles to the axis
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  ax.set_xticklabels(['Not Cloud', 'Cloud'])
  ax.set_yticklabels(['Not Cloud', 'Cloud'])

  # Save the confusion matrix plot if save_dir is provided
  # if save_dir is not None:
  plt.savefig(save_dir+name_file+"confusion_matrix.png")

  # Create a DataFrame for the metrics
  metrics = pd.DataFrame({
      'Metrics': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'Specificity', 'Balanced Accuracy'],
      'Values': [accuracy, precision, recall, f1_score, specificity, balanced_accuracy]
  })

  # Save the metrics as a CSV file if save_dir is provided
  # if save_dir is not None:
  metrics.to_csv(save_dir+name_file + "metrics.csv", index=False)

  return metrics


def norm(band):
  band = np.nan_to_num(band)
  band_min, band_max = band.min(), band.max()
  return (band - band_min) / (band_max - band_min)

def print_tiff_image(images, names, cmapp, file_names):
  # Display the result
  count = 0
  for i, image in enumerate(images):
    plt.figure(figsize=(6, 6))
    if cmapp[count]:
      plt.imshow(image, cmap=cmapp[i])
    else:
      plt.imshow(image[count])
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.title(names[i])
    plot_path = os.path.join(f"{file_names[i]}.jpg")
    plt.savefig(plot_path)
    plt.close()

def load_product(file_path):
    im = tiff.imread(file_path)

    width, height, _ = im.shape
    # Create RGB image
    img_rgb = np.zeros((width, height, 4))
    # Load the img to be used for prediction
    img = np.zeros((width, height, 3), dtype=np.float32)
    img[:, :, :] = np.nan_to_num(tiff.imread(file_path)[:, :, :3])
    # Create the overlapping pixels mask (Band 1-8 are overlapping, but 9, 10, 11 vary at the border regions)
    return img

def load_lansat(file_path):
    r = norm(tiff.imread(file_path + "_B4.tif"))
    g = norm(tiff.imread(file_path + "_B3.tif"))
    b = norm(tiff.imread(file_path + "_B2.tif"))
    return np.stack([r, g, b], axis=2)

def predict_edge(data_path, path_prefix, image_source = "lansat"):
    # print(product)
    # print(folder)
    if image_source == "landsat":
        background_rgb = load_lansat(data_path + "/" + path_prefix)
    else:
        background_rgb = load_product(data_path + "/" + path_prefix +"_4bands.tif")
    large_tiff = Image.fromarray(np.uint8(background_rgb * 255))

    # Convert from BGR to RGB color space
    large_tiff = np.array(large_tiff)
    large_tiff = cv2.cvtColor(large_tiff, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(large_tiff, cv2.COLOR_RGB2GRAY)

    (thresh, im_bw) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = apply_otsu_threshold(gray)
    grad_norm = sobel_detection(im_bw, 5)

    # Read the third image
    img_prediction = tiff.imread(data_path + "/" + path_prefix + '_predicted.tif')
    img_prediction = cv2.cvtColor(img_prediction, cv2.COLOR_BGR2RGB)
    red = [255, 0, 0]

    blended_img2 = overlay_images(img_prediction, grad_norm, red)

    imggggg = color_contour(grad_norm, blended_img2)

    imggggg = np.uint8(imggggg)
    # Split the images into their individual color channels
    b1, g1, r1 = cv2.split(imggggg)
    b2, g2, r2 = cv2.split(blended_img2)

    # Replace the green channel in img2 with the green channel from img1

    # Copy the green channel from img1 to img2 if g = 255, otherwise use the green channel from img2
    for i in range(g1.shape[0]):
        for j in range(g1.shape[1]):
            if g1[i, j] == 255:
                g2[i, j] = g1[i, j]
                b2[i, j] = 0
                r2[i, j] = 0

    # Merge the color channels into a single image
    merged = cv2.merge((b2, g2, r2))
    green = (0, 255, 0)
    blended_img3 = overlay_images(img_prediction, large_tiff, red)

    blended_img1 = overlay_images(imggggg, blended_img3, green)

    names_edge_blur = ['Sobel edge detection',
                       'Edge detection result on Original Image']
    cmapp = ['gray', None]
    images = [grad_norm, blended_img1]
    files_names = [f"{data_path}/{path_prefix}_sobel_edge_detection", f"{data_path}/{path_prefix}_edge_detection"]
    print_tiff_image(images, names_edge_blur, cmapp, files_names)

    return f"{data_path}/{path_prefix}_edge_detection.jpg"
