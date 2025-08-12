import cv2
import numpy as np


class StructuralPattern:
    def __init__(self, image):
        self.image = image

    @staticmethod
    def Change_Neighbour_Pixel_as_Center(N, C):
        # Initialize the new value to 0
        new_value = 0
        try:
            # If the center pixel value C is greater than the neighbor pixel value N
            if C > N:
                # Set the new value to 1
                new_value = 1
        except:
            # Pass in case of any exception (though this exception handling is quite broad and might need refinement)
            pass
        # Return the new value
        return new_value

    @staticmethod
    def get_pixel(image, center, x, y):
        # Initialize the new value to 0
        new_value = 0
        try:
            # Check if the pixel at position (x, y) is greater than the center pixel value
            if image[x][y] > center:
                # Set the new value to 1 if the condition is met
                new_value = 1
        except:
            # If an exception occurs (e.g., index out of range), just pass
            pass
        # Return the new value (either 0 or 1)
        return new_value

    def check_neighbouring_pixel_with_center(self, image, x, y):
        # Get the center pixel value
        center = image[x][y]

        # Create an array of values by checking neighboring pixels relative to the center pixel
        val_ar = [
            self.get_pixel(image, center, x - 1, y - 1),  # Top-left
            self.get_pixel(image, center, x - 1, y),  # Top
            self.get_pixel(image, center, x - 1, y + 1),  # Top-right
            self.get_pixel(image, center, x, y + 1),  # Right
            self.get_pixel(image, center, x + 1, y + 1),  # Bottom-right
            self.get_pixel(image, center, x + 1, y),  # Bottom
            self.get_pixel(image, center, x + 1, y - 1),  # Bottom-left
            self.get_pixel(image, center, x, y - 1)  # Left
        ]

        # Define the power values for each position
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]

        # Initialize the final value to 0
        val = 0

        # Calculate the final value by summing the products of val_ar and power_val
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]

        # Return the calculated value
        return val

    @staticmethod
    def binary_to_decimal(val_ar):
        # Define the power values corresponding to each bit position
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]

        # Initialize the decimal value to 0
        val = 0

        # Loop through each bit in the input binary array
        for i in range(len(val_ar)):
            # Add the product of the binary value and its corresponding power value to the decimal value
            val += val_ar[i] * power_val[i]

        # Return the calculated decimal value
        return val

    def get_structural_pattern(self):
        # Convert the image to grayscale if it is not already
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Get the height and width of the image
        height, width = self.image.shape[0], self.image.shape[1]

        # Initialize eight matrices to store the binary patterns
        b = np.zeros((height, width), np.uint8)
        b1 = np.zeros((height, width), np.uint8)
        b2 = np.zeros((height, width), np.uint8)
        b3 = np.zeros((height, width), np.uint8)
        b4 = np.zeros((height, width), np.uint8)
        b5 = np.zeros((height, width), np.uint8)
        b6 = np.zeros((height, width), np.uint8)
        b7 = np.zeros((height, width), np.uint8)
        b8 = np.zeros((height, width), np.uint8)

        # Loop through each pixel in the image (excluding the borders)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Check the neighboring pixels for the current pixel
                b[i, j] = self.check_neighbouring_pixel_with_center(self.image, i, j)

        # Loop through each pixel again to calculate the ternary patterns
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                t = 0
                # Initialize arrays to store ternary patterns for 8 directions
                z1, z2, z3, z4, z5, z6, z7, z8 = (
                    np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8),
                    np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
                )
                # Loop through a 3x3 neighborhood
                for k in range(-2, 1):
                    for l in range(-2, 1):
                        # Calculate the ternary patterns for the 8 directions
                        z1[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i + 1, j])
                        z2[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i - 1, j])
                        z3[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i, j + 1])
                        z4[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i, j - 1])
                        z5[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i + 1, j + 1])
                        z6[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i - 1, j - 1])
                        z7[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i + 1, j - 1])
                        z8[t] = self.Change_Neighbour_Pixel_as_Center(self.image[i + k, j + l], self.image[i - 1, j + 1])
                # Convert the binary patterns to decimal values
                b1[i, j] = self.binary_to_decimal(z1)
                b2[i, j] = self.binary_to_decimal(z2)
                b3[i, j] = self.binary_to_decimal(z3)
                b4[i, j] = self.binary_to_decimal(z4)
                b5[i, j] = self.binary_to_decimal(z5)
                b6[i, j] = self.binary_to_decimal(z6)
                b7[i, j] = self.binary_to_decimal(z7)
                b8[i, j] = self.binary_to_decimal(z8)

        # Calculate the average of the binary patterns to get the final structural pattern
        BB = (b + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8) // 9

        # Return the final structural pattern
        return BB