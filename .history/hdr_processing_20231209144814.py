import numpy as np
import cv2
import random
import logging
import exifread
import os
import zipfile
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput

# Enhanced logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to extract metadata
def extract_metadata(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file)
            return {tag: str(tags[tag]) for tag in tags if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']}
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}", exc_info=True)
        return {}


# HDR Processing Functions
def linearWeight(pixel_value):
    z_min, z_max = 0., 255.
    return pixel_value - z_min if pixel_value <= (z_min + z_max) / 2 else z_max - pixel_value


def sampleIntensities(images):
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)
    mid_img = images[num_images // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    z_min, z_max = 0, 255
    intensity_range = 255
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)
    mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, intensity_range + 1 + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1

    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    mat_A[k, (z_max - z_min) // 2] = 1
    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)
    g = x[0:intensity_range + 1]
    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)
    num_images = len(images)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([response_curve[images[k][i, j]] for k in range(num_images)])
            w = np.array([weighting_function(images[k][i, j]) for k in range(num_images)])
            SumW = np.sum(w)
            if SumW > 0:
                img_rad_map[i, j] = np.sum(w * (g - log_exposure_times) / SumW)
            else:
                img_rad_map[i, j] = g[num_images // 2] - log_exposure_times[num_images // 2]

    return img_rad_map


def globalToneMapping(image, gamma):
    image_corrected = cv2.pow(image / 255., 1.0 / gamma)
    return image_corrected


def intensityAdjustment(image, template):
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:,:, ch]), np.average(template[:,:, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)
    return output


def computeHDR(images, log_exposure_times, smoothing_lambda=100., gamma=0.6):
    num_channels = images[0].shape[2]
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)
    for channel in range(num_channels):
        layer_stack = [img[:,:, channel] for img in images]
        intensity_samples = sampleIntensities(layer_stack)
        response_curve = computeResponseCurve(intensity_samples, log_exposure_times, smoothing_lambda, linearWeight)
        img_rad_map = computeRadianceMap(layer_stack, log_exposure_times, response_curve, linearWeight)
        hdr_image[..., channel] = cv2.normalize(img_rad_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image_mapped = globalToneMapping(hdr_image, gamma)
    template = images[len(images) // 2]
    image_tuned = intensityAdjustment(image_mapped, template)
    output = cv2.normalize(image_tuned, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output.astype(np.uint8)


# Unified HDR Processing Invocation Class
@invocation("Unified_HDR_Processing", title="Unified HDR Processing", tags=["image", "hdr", "ai"], category="image", version="1.0.0", use_cache=False)
class UnifiedHDRProcessingInvocation(BaseInvocation):
    input_images: list[ImageField] = InputField(description="Input images for HDR processing")
    exposure_times: list[float] = InputField(description="Exposure times for each input image")
    pseudo_exposure_factors: list[float] = InputField(description="Pseudo exposure factors for single image HDR", default=[])

    def adjust_brightness(self, image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def create_pseudo_hdr_image(self, base_image, factors):
        pseudo_hdr_stack = [self.adjust_brightness(base_image, factor) for factor in factors]
        return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in pseudo_hdr_stack]

    def align_images(self, images_cv):
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images_cv, images_cv)
        return images_cv

    def tone_map_image(self, hdr_image):
        tonemap = cv2.createTonemap(2.2)
        ldr_image = tonemap.process(hdr_image)
        return np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

    def create_hdr_image(self, images_cv, exposure_values):
        hdr_image = computeHDR(images_cv, exposure_values)
        return hdr_image

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        try:
            output_images = []
            if len(self.input_images) > 1 and self.exposure_times:
                images_cv = [cv2.cvtColor(np.array(context.services.images.get_pil_image(image_field.image_name).convert('RGB')), cv2.COLOR_RGB2BGR) for image_field in self.input_images]
                images_cv = self.align_images(images_cv)
                hdr_result_cv = self.create_hdr_image(images_cv, np.log(self.exposure_times))
                hdr_result = self.tone_map_image(hdr_result_cv)
            elif len(self.input_images) == 1 and self.pseudo_exposure_factors:
                base_image_pil = context.services.images.get_pil_image(self.input_images[0].image_name)
                pseudo_hdr_images_cv = self.create_pseudo_hdr_image(base_image_pil, self.pseudo_exposure_factors)
                pseudo_hdr_images_cv = self.align_images(pseudo_hdr_images_cv)
                hdr_result_cv = self.create_hdr_image(pseudo_hdr_images_cv, np.log(self.pseudo_exposure_factors))
                hdr_result = self.tone_map_image(hdr_result_cv)
            else:
                logger.error("Invalid input for HDR processing: insufficient images or missing exposure data", exc_info=True)
                raise ValueError("Invalid input for HDR processing.")

            hdr_result_pil = Image.fromarray(cv2.cvtColor(hdr_result, cv2.COLOR_BGR2RGB))
            output_image_name = f"Unified_HDR_result_{context.graph_execution_state_id}.jpg"
            hdr_result_pil.save(output_image_name)
            output_images.append(ImageField(image_name=output_image_name))

            return ImageCollectionOutput(images=output_images, result_message="HDR processing completed successfully.")
        except Exception as e:
            logger.error("Unexpected error in Unified HDR processing node", exc_info=True)
            return ImageCollectionOutput(images=[], result_message=f"Unexpected error occurred: {e}")


# Invocation for retrieving images from file or directory
@invocation(
    "Retrieve_Images_From_File_Unique",  # Changed name to be unique
    title="Retrieve Images from File or Directory",
    tags=["image", "file"],
    category="image",
    version="0.1.0",
    use_cache=False
)
class RetrieveImagesFromFileInvocation(BaseInvocation):
    input_path: str = InputField(description="Path to the file or directory containing images")
    save_to_zip: bool = InputField(description="Save all retrieved images to a ZIP file.", default=False)
    zip_save_path: str = InputField(description="Custom path to save the ZIP file.", default="")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        try:
            selected_images = self.get_images_from_path(self.input_path)
            processed_images = []
            metadata_collection = []

            for image_path in selected_images:
                try:
                    with Image.open(image_path) as img:
                        metadata = extract_metadata(image_path)
                        metadata_collection.append(metadata)
                        invoke_image = context.services.images.create(
                            image=img,
                            image_origin=ResourceOrigin.EXTERNAL,
                            image_category="general",
                            node_id=self.id,
                            session_id=context.graph_execution_state_id,
                        )
                        processed_images.append(ImageField(image_name=invoke_image.image_name))
                except Exception as img_error:
                    logger.error(f"Error processing image {image_path}: {img_error}", exc_info=True)

            if self.save_to_zip:
                zip_file_name = os.path.join(self.zip_save_path or SCRIPT_DIR, "images.zip")
                with zipfile.ZipFile(zip_file_name, "w") as zipf:
                    for image_field in processed_images:
                        image_path = os.path.join(context.services.images.get_base_path(), image_field.image_name)
                        zipf.write(image_path, os.path.basename(image_path))
                result_message = f"Your images are saved in {zip_file_name}"
            else:
                result_message = "Images retrieved and processed successfully."

            return ImageCollectionOutput(images=processed_images, metadata=metadata_collection, result_message=result_message)

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}", exc_info=True)
            return ImageCollectionOutput(images=[], metadata=[], result_message=f"Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return ImageCollectionOutput(images=[], metadata=[], result_message=f"Unexpected error: {e}")

    def get_images_from_path(self, path):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(path):
            with open(path, 'r') as file:
                return [line.strip() for line in file.readlines() if os.path.isfile(line.strip())]
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")

# Additional custom functions or nodes can be added below


# Example of an additional custom function
def custom_image_processing_function(image_path):
    """
    Custom image processing function.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Processed image object.
    """
    try:
        # Placeholder for actual image processing logic
        processed_image = some_image_processing_function(image_path)
        return processed_image
    except Exception as e:
        logger.error(f"Error in custom image processing for {image_path}: {e}", exc_info=True)
        return None


# Example of an additional custom node
@invocation(
    "Custom_Image_Processing",
    title="Custom Image Processing",
    tags=["image", "processing"],
    category="image",
    version="1.0.0",
    use_cache=False
)
class CustomImageProcessingInvocation(BaseInvocation):
    input_image: ImageField = InputField(description="Input image for processing")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        try:
            image_path = context.services.images.get_base_path() + self.input_image.image_name
            processed_image = custom_image_processing_function(image_path)
            if processed_image:
                output_image_name = f"Processed_{self.input_image.image_name}"
                processed_image.save(output_image_name)
                output_image_field = ImageField(image_name=output_image_name)
                return ImageCollectionOutput(images=[output_image_field], result_message="Image processed successfully.")
            else:
                return ImageCollectionOutput(images=[], result_message="Failed to process the image.")
        except Exception as e:
            logger.error("Unexpected error in Custom Image Processing node", exc_info=True)
            return ImageCollectionOutput(images=[], result_message=f"Unexpected error occurred: {e}")


# Example usage of the additional custom node
if __name__ == "__main__":
    custom_invocation = CustomImageProcessingInvocation()
    custom_result = custom_invocation.invoke(context)
    logger.debug("Custom image processing completed")

# Further custom nodes or logic can be added here.
