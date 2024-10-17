#import image
#this is for my project here idk lol need to get intensity of the image
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector
import argparse
import copy
import matplotlib.patches as patches 

#notes - review if self 

class Plume:
    def __init__(self, image_path: str = None, image_array: np.ndarray = None):


        

        if image_path is not None:
            image_paths = ['test-image-for-tripathi.jpeg']
            image = Image.open(image_path)

            self.image_array = asarray(image)

            print(f"Successfully loaded image from {image_path} with shape: {self.image_array.shape}")
        elif image_array is not None:
            self.image_array = image_array  
            print(f"image array is {self.image_array}")
        else:
            raise RuntimeError("Constructor must provide either an image path or image array")
    
        
        self.image_height = len(self.image_array)
        self.image_width = len(self.image_array[0])
        print(f"image shape is {self.image_array.shape}")
        print(f"image height is {self.image_height}")
        print(f"image width is {self.image_width}")
        #look up what self. in front of parameters means

        if self.image_height < 2 or self.image_width < 2:
            raise ValueError("Image provided must be at least 2x2 pixels")
     # return image_array

    def to_grayscale(self):
        grayscale_image = Image.fromarray(self.image_array).convert('L')
        self.image_array = np.array(grayscale_image)
        print("image converted to grayscale")

    def select_roi(self, image_path):
        image = Image.open(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)

        roi_selector = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(roi_selector)
        print("Select ROI by clicking and dragging. Press Enter to confirm.")

        self.selected_roi = None

        xmin, xmax, ymin, ymax = None, None, None, None

        def onselect(xmin_val, xmax_val):
            nonlocal xmin, xmax  # Use nonlocal to modify the outer scope variables
            xmin, xmax = xmin_val, xmax_val
            print(f"Selected X Range: {xmin}, {xmax}")

        def onselect_vertical(ymin_val, ymax_val):
            nonlocal ymin, ymax  # Use nonlocal to modify the outer scope variables
            ymin, ymax = ymin_val, ymax_val
            print(f"Selected Y Range: {ymin}, {ymax}")

        # def onselect(xmin, xmax):
        #    print(xmin, xmax)
           
        span_horizontal = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        props=dict(alpha=0.5, facecolor='red'))
        
        span_vertical = SpanSelector(ax, onselect_vertical, 'vertical', useblit=True,
                        props=dict(alpha=0.5, facecolor='blue'))

        def on_key(event):
            if event.key == 'enter':
                plt.close(fig)  # Close the figure on Enter key

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            self.selected_roi = (int(xmin), int(ymin), int(xmax), int(ymax))
            print(f"Final Selected ROI: {self.selected_roi}")
        else:
            raise RuntimeError("No region of interest selected.")

        # if self.selected_roi is None:
        #     raise RuntimeError("No region of interest selected.")
        plt.close()
        return self.selected_roi
    

    def get_region_of_interest(self, region):
        x1, y1, x2, y2 = region
        return self.image_array[y1:y2, x1:x2]

    def compute_mean_intensity(self, roi):
        intensity = np.mean(roi)
        return intensity

    def intensity_to_concentration(self, intensity, calibration_factor):
        return intensity * calibration_factor #replace with the actual calibration factor

    def analyze_image(self, calibration_factor, region):
        self.to_grayscale()
        roi = self.get_region_of_interest(region)
        mean_intensity = self.compute_mean_intensity(roi)
        #check this line of code
        concentration = self.intensity_to_concentration(mean_intensity, calibration_factor)
        return concentration
    

    @staticmethod
    def analyze_image_series(image_paths, region, calibration_factor):
        concentrations = []
        for image_path in image_paths:
            analyzer = Plume(image_path=image_path)
            concentration = analyzer.analyze_image(calibration_factor, region)
            concentration = concentration.item()
            concentrations.append(concentration)
        return concentrations

    @staticmethod
    def plot_concentrations(concentrations):
        x_values = range(len(concentrations))
        print(x_values)
        plt.plot(x_values, concentrations, marker='o')
        plt.xticks(x_values)
        plt.xlabel('Time (image number)')
        plt.ylabel('Dye Concentration (mg/unit)')
        plt.title('Dye Concentration Over Time')
        plt.show()

if __name__ == "__main__":
        print("Starting...")
        image_paths = ['test-image-for-tripathi.jpeg']  # Replace with your image paths
        print(f"Image paths: {image_paths}")

       
        try:
            print(f"Creating plume instance...")
            plume_instance = Plume(image_path=image_paths[0])

            print(f"Selecting ROI...")
            region = plume_instance.select_roi(image_paths[0])  # Define the region of interest (x1, y1, x2, y2)
            print (f"Final selected region: {region}")

            calibration_factor = 35  # Define the calibration factor to convert intensity to concentration
            print (f"Calibration factor: {calibration_factor}")

            print (f"analyzing image series...")
            concentrations = Plume.analyze_image_series(image_paths, region, calibration_factor)
        #put plume in front of plot concentrations
            print(f"Concentrations: {concentrations}")

            print("Plotting concentrations...")
            Plume.plot_concentrations(concentrations)
            print("Program completed successfully.")

        except RuntimeError as e:
            print(f"Error: {e}")
