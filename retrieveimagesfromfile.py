# File: retrievefromfile.py

import os
import zipfile
import logging
from PIL import Image
import exifread
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, OutputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput

# Configure the logger for detailed logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging, INFO for general logging
logger = logging.getLogger(__name__)

def extract_metadata(image_path):
    """
    Extracts metadata from an image file using exifread.
    """
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file)
            return {tag: str(tags[tag]) for tag in tags if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']}
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        return {}

@invocation("Retrieve_Images_From_File", title="Retrieve Images from File or Directory", tags=["image", "file"], category="image", version="1.0.0", use_cache=False)
class RetrieveImagesFromFileInvocation(BaseInvocation):
    input_path: str = InputField(description="Path to the file or directory containing images")
    save_to_zip: bool = InputField(description="Save all retrieved images to a ZIP file.", default=False)
    zip_save_path: str = InputField(description="Custom path to save the ZIP file.", default="")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        logger.info(f"Invoked RetrieveImagesFromFile with path: {self.input_path}")
        selected_images = self.get_images_from_path(self.input_path)
        processed_images = []
        metadata_collection = []

        for image_path in selected_images:
            try:
                with Image.open(image_path) as img:
                    metadata = extract_metadata(image_path)
                    metadata_collection.append(metadata)
                    processed_images.append(ImageField(image_name=image_path))  # Modify as per your requirement
                    logger.debug(f"Processed image: {image_path}")
            except Exception as img_error:
                logger.error(f"Error processing image {image_path}: {img_error}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        if self.save_to_zip:
            self.create_zip_file(processed_images)

        return ImageCollectionOutput(images=processed_images, metadata=metadata_collection, result_message="Images retrieved successfully.")

    def get_images_from_path(self, path):
        logger.debug(f"Retrieving images from path: {path}")
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(path):
            with open(path, 'r') as file:
                return [line.strip() for line in file if line.strip().lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"The specified path does not exist: {path}")

    def create_zip_file(self, images):
        zip_file_name = os.path.join(self.zip_save_path or os.getcwd(), "images.zip")
        with zipfile.ZipFile(zip_file_name, 'w') as zipf:
            for image in images:
                zipf.write(image.image_name, os.path.basename(image.image_name))
            logger.info(f"Saved images to ZIP file at: {zip_file_name}")

if __name__ == "__main__":
    # Example usage or testing of the node
    pass
