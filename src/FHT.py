import numpy as np
import cv2
from PIL import Image

# =============================== INTERPOLATIONS ================================

def NearestNeighborInterpolation(image: np.array, result: np.array, x: int, 
                                 y: int, x_found: float, y_found: float):
    ''' Find values by the closest pixel in picture '''
    x_found = np.clip(int(np.round(x_found)), 0, image.shape[0] - 1)
    y_found = np.clip(int(np.round(y_found)), 0, image.shape[1] - 1)
    result[x, y, :] = image[x_found, y_found, :]

def BilinearInterpolation(image: np.array, result: np.array, x: int, 
                          y: int, x_found: float, y_found: float):
    ''' Find values by interpolation neighbours' values '''

    # get closest lower. If it is already integer, get lower one
    y_lower = (int(y_found) - 1) if int(y_found) == np.ceil(y_found) else int(y_found)
    y_lower = np.clip(y_lower, 0, image.shape[1] - 1)
    y_upper = np.clip(y_lower + 1, 0, image.shape[1] - 1)

    # same as previous
    x_lower = (int(x_found) - 1) if int(x_found) == np.ceil(x_found) else int(x_found)
    x_lower = np.clip(x_lower, 0, image.shape[0] - 1)
    x_upper = np.clip(x_lower + 1, 0, image.shape[0] - 1)

    # get cells that are near our cell
    left_down = image[x_lower, y_lower, :]
    left_up = image[x_upper, y_lower, :]
    right_down = image[x_lower, y_upper, :]
    right_up = image[x_upper, y_upper, :]

    # calculate linear interpolation
    first_diag = left_down + (y_found - y_lower) * (right_down - left_down)
    second_diag = left_up + (y_found - y_lower) * (right_up - left_up)

    # calculate bilinear interpolation
    result_values = first_diag + (x_found - x_lower) * (second_diag - first_diag)

    # save clipped result
    result[x, y, :] = np.clip(result_values, 0, 255)


# ====================================IMAGE=PROCESSOR==================================================
class ImageProcessor:
    ''' 
    Image processor is a class with FHT realisation
    
    PreprocessImage - makes gradients to determine bounds. It can be called from any image and it is a static method.

    GetFHTForImage - calculates Fast Hough Transform for image (just array for further work)

    RotateCurreтеImage - rotate image that was put into processor
    '''
    def __init__(self, image_path=None):
        self._image_path = image_path
        if self._image_path is not None:
            self._image = Image.open(self._image_path)
        else:
            self._image = None

    @staticmethod
    def PreprocessImage(image, should_be_reversed: bool) -> np.array:
        '''
        1) Filter noise
        2) Create gradient with Sobel filter
        3) reshape image because algorithm works only with degrees of 2
        '''
        result = cv2.GaussianBlur(np.array(image), (3, 3), 0)

        if should_be_reversed:
            result = np.flip(result, axis=1)
        
        gradient_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.convertScaleAbs(cv2.Sobel(gradient_image, 3, 1, 0, ksize=3))
        gradient_y = cv2.convertScaleAbs(cv2.Sobel(gradient_image, 3, 0, 1, ksize=3))
        gradient_image = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
        
        MaxDegreeOfTwo = 2 ** int(np.log2(gradient_image.shape[1]))
        return np.array(gradient_image[:, :MaxDegreeOfTwo], dtype=np.float32)

    @staticmethod
    def __FHTRecursiveStep(image, begin_of_part: int, end_of_part: int) -> np.array:
        '''
        Recursive FHT realisation

        For each line calculate results on sublines and merge them. 
        Note that in some cases middle index for second cell should be bigger than int part of division
        '''
        current_width = end_of_part - begin_of_part
        if current_width == 1:
            return image[:, begin_of_part:end_of_part]

        result = np.zeros([image.shape[0], current_width])

        # execute subtasks
        middle = (begin_of_part + end_of_part) // 2
        result_on_left_part = ImageProcessor.__FHTRecursiveStep(image, begin_of_part, middle)
        result_on_right_part = ImageProcessor.__FHTRecursiveStep(image, middle, end_of_part)

        for j in range(current_width):
            first_middle_index = j // 2
            second_middle_index = j // 2 + j % 2

            for i in range(image.shape[0]):
                result[i, j] = result_on_left_part[i, first_middle_index] + \
                              result_on_right_part[(i + second_middle_index) % image.shape[0], first_middle_index]

        return result

    @staticmethod
    def GetFHTForImage(image, should_be_reversed: bool):
        gradient = ImageProcessor.PreprocessImage(image, should_be_reversed=should_be_reversed)
        return ImageProcessor.__FHTRecursiveStep(gradient, 0, gradient.shape[1])

    @staticmethod
    def __FindAngle(FHT_forwarded: np.array, FHT_reversed: np.array):
        '''
        Find angel from two versions of FHT
        '''
        variance_forwarded = np.var(FHT_forwarded, axis=0)
        max_position_forwarded = np.argmax(variance_forwarded)
        variance_forwarded = variance_forwarded[max_position_forwarded]

        variance_reversed = np.var(FHT_reversed, axis=0)
        max_position_reversed = np.argmax(variance_reversed)
        variance_reversed = variance_reversed[max_position_reversed]

        if variance_forwarded > variance_reversed:
            length = FHT_forwarded.shape[1]
            denum = np.sqrt(max_position_forwarded ** 2 + length ** 2)
            sin_alpha = max_position_forwarded / denum
            cos_alpha = length / denum
            return cos_alpha, -sin_alpha
        
        # Otherwise:
        length = FHT_reversed.shape[1]
        denum = np.sqrt(max_position_reversed ** 2 + length ** 2)
        sin_alpha = max_position_reversed / denum
        cos_alpha = length / denum
        return cos_alpha, sin_alpha

    @staticmethod
    def __FindСoords(x, y, height: int, width: int, cos_a: float, sin_a: float):
        '''
        We consider that the current picture was obtained after a set of transformations, 
        namely shift -> rotate -> shift back -> scale.
        Therefore we must do all the formations inverse to the data
        '''
        
        # scale
        first_edge = height * np.abs(cos_a) + width * np.abs(sin_a)
        second_edge = width * np.abs(cos_a) + height * np.abs(sin_a)
        scale_coeff = min(height / first_edge, width / second_edge)
        
        y *= scale_coeff
        x *= scale_coeff
        
        # shift
        y -= height * scale_coeff / 2
        x -= width * scale_coeff / 2

        # back rotation
        x, y = x * cos_a + y * sin_a, -x * sin_a + y * cos_a
        
        # shift back
        y += height / 2
        x += width / 2

        return x, y

    def RotateCurreтеImage(self, interpolation=BilinearInterpolation):
        '''
        Rotate image to readable form
        '''
        if self._image is None:
            print("Can't rotate image, because it wasn't read")
        
        # Find angels for rotation
        FHT_forwarded = self.GetFHTForImage(self._image, should_be_reversed=False)
        FHT_reversed = self.GetFHTForImage(self._image, should_be_reversed=True)
        angle = ImageProcessor.__FindAngle(FHT_forwarded, FHT_reversed)

        image = np.array(self._image, dtype=np.float32)
        result = np.zeros_like(self._image)
    
        # calculate rotation for each coordinates
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                x_found, y_found = ImageProcessor.__FindСoords(x, y, image.shape[0], image.shape[1], angle[0], angle[1])
                interpolation(image, result, y, x, y_found, x_found)

        # convert to 8-bytes to get image
        return cv2.convertScaleAbs(result)
    
    def RotateImage(self, image, interpolation=BilinearInterpolation):
        self._image = image

        # Find angels for rotation
        FHT_forwarded = self.GetFHTForImage(self._image, should_be_reversed=False)
        FHT_reversed = self.GetFHTForImage(self._image, should_be_reversed=True)
        angle = ImageProcessor.__FindAngle(FHT_forwarded, FHT_reversed)

        image = np.array(self._image, dtype=np.float32)
        result = np.zeros_like(self._image)
    
        # calculate rotation for each coordinates
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                x_found, y_found = ImageProcessor.__FindСoords(x, y, image.shape[0], image.shape[1], angle[0], angle[1])
                interpolation(image, result, y, x, y_found, x_found)

        # convert to 8-bytes to get image
        return cv2.convertScaleAbs(result)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image_path: str):
        self._image = Image.open(new_image_path)
