from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

@invocation("HDR_Processing", title="HDR Image Processing", tags=["image", "hdr"], category="image", version="1.0.0", use_cache=False)
class HDRProcessingInvocation(BaseInvocation):
    input_images: list[ImageField] = InputField(description="Input images for HDR processing")
    exposure_times: list[float] = InputField(description="Exposure times for each input image")

    def create_hdr_image(self, images, exposure_times):
        images_cv = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images_cv, images_cv)
        merge_debevec = cv2.createMergeDebevec()
        hdr_image = merge_debevec.process(images_cv, times=np.array(exposure_times))
        tonemap = cv2.createTonemap(2.2)
        ldr_image = tonemap.process(hdr_image)
        ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))

    def invoke(self, context: InvocationContext) -> ImageOutput:
        try:
            input_pil_images = []
            valid_exposures = []

            for image_field, exposure_time in zip(self.input_images, self.exposure_times):
                # Replace get_pil_image with the actual method to retrieve the image
                pil_image = context.services.images.get_pil_image(image_field.image_name)
                if pil_image is not None:
                    input_pil_images.append(pil_image)
                    valid_exposures.append(exposure_time)
                else:
                    logger.error(f"Image not found or invalid: {image_field.image_name}")

            if not input_pil_images:
                logger.error("No valid input images found.")
                return ImageOutput()

            hdr_result = self.create_hdr_image(input_pil_images, valid_exposures)

            output_image = context.services.images.create(
                image=hdr_result,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )

            return ImageOutput(
                image=ImageField(image_name=output_image.image_name),
                width=output_image.width,
                height=output_image.height
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e
