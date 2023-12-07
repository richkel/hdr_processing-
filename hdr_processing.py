# Import necessary libraries
import numpy as np
import cv2
import random
import logging
from PIL import Image, ImageEnhance
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, OutputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput

# HDR functions from your code (e.g., linearWeight, sampleIntensities, etc.)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@invocation("Unified_HDR_Processing", title="Unified HDR Processing", tags=["image", "hdr", "ai"], category="image", version="1.0.0", use_cache=False)
class UnifiedHDRProcessingInvocation(BaseInvocation):
    # ... [Class Fields]

    def create_pseudo_exposure_stack(self, base_image, factors):
        # ... [Function Body]

     def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        logger.info("Unified HDR Processing node invoked")
        try:
            output_images = []
            if len(self.input_images) > 1 and self.exposure_times:
                # True HDR Processing with multiple images
                images_cv = [cv2.cvtColor(np.array(context.services.images.get_pil_image(image_field.image_name).convert('RGB')), cv2.COLOR_RGB2BGR) for image_field in self.input_images]
                hdr_result_cv = computeHDR(images_cv, np.log(self.exposure_times))
            elif len(self.input_images) == 1 and self.pseudo_exposure_factors:
                # Pseudo-HDR Processing with a single image
                base_image = context.services.images.get_pil_image(self.input_images[0].image_name)
                pseudo_stack = self.create_pseudo_exposure_stack(base_image, self.pseudo_exposure_factors)
                images_cv = [cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR) for image in pseudo_stack]
                hdr_result_cv = computeHDR(images_cv, np.log(self.pseudo_exposure_factors))
            else:
                raise ValueError("Invalid input for HDR processing.")

            # Convert back to PIL image and save
            hdr_result = Image.fromarray(cv2.cvtColor(hdr_result_cv, cv2.COLOR_BGR2RGB))
            output_image_name = f"Unified_HDR_result_{context.graph_execution_state_id}.jpg"
            hdr_result.save(output_image_name)
            output_images.append(ImageField(image_name=output_image_name))
            self.result_message = "HDR processing completed successfully."

            return ImageCollectionOutput(images=output_images, result_message=self.result_message)
        except Exception as e:
            logger.error(f"Unexpected error in Unified HDR processing node: {e}")
            return ImageCollectionOutput(images=[], result_message="Unexpected error occurred")


if __name__ == "__main__":
    # Example usage
    pass
