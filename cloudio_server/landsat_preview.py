import os
import matplotlib.pyplot as plt
import tifffile as tiff
import matplotlib as mpl

mpl.use('agg')

class BandImageExporter:
    def __init__(self):
        self.export_directory = None

    def set_export_directory(self, export_directory):
        self.export_directory = export_directory

    def plot_band_image(self, image_path, band_name):
        band = tiff.imread(self.export_directory + image_path + "_" + band_name + ".tif")
        plt.figure(figsize=(6, 6))
        plt.imshow(band, cmap='gray')
        plt.axis('off')
        plt.title(f'{band_name} Band')
        preview_dir = os.path.join(self.export_directory, 'preview')
        os.makedirs(preview_dir, exist_ok=True)
        plot_path = os.path.join(preview_dir, f'{band_name}_plot.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def plot_geos_image(self, image_path):
        preview_dir = os.path.join(self.export_directory, 'preview')
        os.makedirs(preview_dir, exist_ok=True)
        plot_path = os.path.join(preview_dir, f'4bands_plot.png')
        # Define the TIFF image file path

        # Read the TIFF image
        image = tiff.imread(self.export_directory + image_path + "_4bands.tif")

        # Plot the image for each channel
        channel_image = image[:, :, 3]
        plt.figure(figsize=(6, 6))
        plt.imshow(channel_image, cmap='gray')
        plt.title("Geos 18 Image")
        plt.axis('off')
        plt.savefig(plot_path)
        plt.close()
        return plot_path
