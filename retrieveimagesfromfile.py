import os
import zipfile
import logging
from PIL import Image
import exifread
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput
from invokeai.app.invocations.image import ResourceOrigin
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_metadata(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file)
            return {tag: str(tags[tag]) for tag in tags if tag not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']}
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}", exc_info=True)
        return {}


class ImageZipResult(BaseModel):
    message: str


@invocation(
    "Retrieve_Images_From_File",
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

            return ImageCollectionOutput(collection=processed_images, metadata=metadata_collection, result_message=result_message)

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}", exc_info=True)
            return ImageCollectionOutput(collection=[], metadata=[], result_message=f"Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return ImageCollectionOutput(collection=[], metadata=[], result_message=f"Unexpected error: {e}")

    def get_images_from_path(self, path):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(path):
            with open(path, 'r') as file:
                return [line.strip() for line in file.readlines() if os.path.isfile(line.strip())]
        else:
            raise FileNotFoundError(f"The specified path does not exist: {path}")


if __name__ == "__main__":
    # Example usage or testing of the node
    pass
