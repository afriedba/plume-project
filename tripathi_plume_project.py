"""
Method for plume concentrations project.

TO USE:
1. Open your terminal. Make sure you are in the plume-project folder. If not, 
    use command cd to open it.
2. Run the file! You can either press the triangle in the top right corner of 
    VSCode, or do it using the python command in terminal.
3. It will prompt you in the terminal to enter the iamge name that you want to 
    use for ROI selection. Make sure you include the type (ex. image.jpeg)
4. Click and drag to select your region. Enter to confirm.
5. The program will finish running. Your concentration plot will save as 
    concentration_plot.png in the plume-project-folder. Please keep in mind that
    you will replace the plot each time you run the program. If you want to 
    keep a certain plot, rename it to something other that concentration_plot.

FOR YOUR IMPLEMENTATIONS:
1. Images right now are just samples. They will be entered alphanumerically, so
    please ensure that they are correctly ordered in the images folder. The
    image you choose to select the ROI for will also be in the alphanumeric 
    order from the folder when plotted on the graph, so if that is meant to be
    in spot 0, you must make it first in the folder.
2. Calibration factor is a random # right now. To adjust, go down to the main 
    method, and give a value to CALIBRATION_FACTOR.
3. To adjust axis titles or any changes to the graph, change the 
    plot_concentrations method. There's a note there for the x-axis labels 
    specifically assuming uniform time differences. If not the case, you'll have 
    to manually enter the x_values in a list ex. [0, 12, 14, ...].
"""

import os
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import matplotlib.patches as patches

class Plume:
    """Main class for project. Each plume represents a single imag and its arguments"""
    def __init__(self, image_path: str = None, image_array: np.ndarray = None):
        """Initializes all of the arguments"""

        if image_path is not None: # checks to make sure image_path exists
            image = Image.open(image_path)

            self.image_array = asarray(image)
            print(f"Successfully loaded image from {image_path} with shape: {self.image_array.shape}")

        # converts to an image array
        elif image_array is not None:
            self.image_array = image_array  
            print(f"image array is {self.image_array}")
        else:
            raise RuntimeError("Constructor must provide either an image path or image array")
    
        # printing the dimensions
        self.image_height = len(self.image_array)
        self.image_width = len(self.image_array[0])
        # print(f"image shape is {self.image_array.shape}")
        # print(f"image height is {self.image_height}")
        # print(f"image width is {self.image_width}")
        # ^ those are helpful for debugging, but don't need to be printed every time

        if self.image_height < 2 or self.image_width < 2:
            raise ValueError("Image provided must be at least 2x2 pixels")
     # returning image_array

        self.selected_roi = None

    def to_grayscale(self):
        """Converts an image to grayscale"""
        grayscale_image = Image.fromarray(self.image_array).convert('L')
        self.image_array = np.array(grayscale_image)
        # print("image converted to grayscale")

    def select_roi(self, image_path):
        """Selects the ROI for a given image"""
        image = Image.open(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)

        roi_selector = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(roi_selector)
        print("Select ROI by clicking and dragging. Press Enter to confirm.")

        xmin, xmax, ymin, ymax = None, None, None, None

        def onselect(xmin_val, xmax_val):
            nonlocal xmin, xmax  # Use nonlocal to modify the outer scope variables
            xmin, xmax = xmin_val, xmax_val
            # print(f"Selected X Range: {xmin}, {xmax}")

        def onselect_vertical(ymin_val, ymax_val):
            nonlocal ymin, ymax  # Use nonlocal to modify the outer scope variables
            ymin, ymax = ymin_val, ymax_val
            # print(f"Selected Y Range: {ymin}, {ymax}")

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
            # print(f"Final Selected ROI: {self.selected_roi}")
        else:
            raise RuntimeError("No region of interest selected.")

        # if self.selected_roi is None:
        #     raise RuntimeError("No region of interest selected.")
        plt.close()
        return self.selected_roi
    
    def get_region_of_interest(self, regions):
        """Returns the image array of the ROI"""
        x1, y1, x2, y2 = regions
        return self.image_array[y1:y2, x1:x2]

    def compute_mean_intensity(self, roi):
        """Calculates the mean intensity"""
        intensity = np.mean(roi)
        return intensity

    def intensity_to_concentration(self, intensity, calibration_factors):
        """Uses the intensity * calibration factor to return the concentration"""
        return intensity * calibration_factors

    def analyze_image(self, calibration_factors, regions):
        """Takes in a plume, and returns the concentration for the defined region """
        self.to_grayscale()
        roi = self.get_region_of_interest(regions)
        mean_intensity = self.compute_mean_intensity(roi)
        concentration = self.intensity_to_concentration(mean_intensity, calibration_factors)
        return concentration
    
def analyze_image_series(image_path_list, regions, calibration_factors):
    """Does the concentration analysis for each image"""
    concentration_list = []
    for image_path in image_path_list:
        analyzer = Plume(image_path=image_path)
        concentration = analyzer.analyze_image(calibration_factors, regions)
        concentration = concentration.item()
        concentration_list.append(concentration)
    return concentration_list

def plot_concentrations(concentration_list):
    """Plots the concentrations"""
    x_values = range(len(concentration_list)) 
    # starts at 0 and adds 1 for each image funneled through
    # if you want to change axis labels, this is where
    # for example, changing to 20 (min/values/wahtever label says) apart
    # x_values = [x * 20 for x in x_values]
    plt.plot(x_values, concentration_list, marker='o')
    plt.xticks(x_values)
    plt.xlabel('Time (image number)')
    plt.ylabel('Dye Concentration (mg/unit)')
    plt.title('Dye Concentration Over Time')
    plt.savefig('concentration_plot.png', format='png')
    plt.show(block=False)

def get_image_paths_from_folder(img_folder_path):
    """ Return all of the images in a list format from the folder path. """
    # Get all image file paths in the folder
    img_paths = sorted([
        os.path.join(img_folder_path, filename)
        for filename in os.listdir(img_folder_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    ])
    return img_paths


if __name__ == "__main__":
    print("Starting...")

    FOLDER_PATH = "images"
    image_paths = get_image_paths_from_folder(FOLDER_PATH)

    if not image_paths:
        raise RuntimeError(f"No images found in folder {FOLDER_PATH}")
       
    try:
        selected_image_path = "images/" + input(f"Enter the name of the image for ROI selection from {FOLDER_PATH}: ")
            
        if not os.path.isfile(selected_image_path):
            raise ValueError("The provided file path is not valid.")

        print("Creating plume instance...")
        plume_instance = Plume(image_path=selected_image_path)

        print("Selecting ROI...")
        region = plume_instance.select_roi(selected_image_path)
        print (f"Final selected region: {region}")

        CALIBRATION_FACTOR = 35  # Define the calibration factor to convert intensity to concentration
        print (f"Calibration factor: {CALIBRATION_FACTOR}")

        print ("analyzing image series...")
        concentrations = analyze_image_series(image_paths, region, CALIBRATION_FACTOR)

        print(f"Concentrations: {concentrations}")

        print("Plotting concentrations...")
        plot_concentrations(concentrations)
        print("Program completed successfully. View plot in the concentration_plot.png file")

    except RuntimeError as e:
        print(f"Error: {e}")