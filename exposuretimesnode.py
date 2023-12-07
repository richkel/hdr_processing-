# File: exposuretimes.py

import logging
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, OutputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@invocation("Exposure_Times", title="Exposure Times for HDR", tags=["hdr", "exposure"], category="image", version="1.0.0", use_cache=False)
class ExposureTimesInvocation(BaseInvocation):
    image_metadata: list[dict] = InputField(description="Metadata for each image", default=[])
    default_exposure_time: float = InputField(description="Default exposure time if not found in metadata", default=1/60)
    input_images: list[ImageField] = InputField(description="Input images for HDR processing")
    exposure_times: list[float] = OutputField(description="Calculated exposure times for HDR image processing.")
    result_message: str = OutputField(description="Result message")

    def calculate_exposure_time(self, metadata):
        exposure_time = metadata.get('EXIF ExposureTime')
        if exposure_time:
            return float(exposure_time)
        return self.default_exposure_time

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        logger.info("Exposure Times node invoked")
        try:
            calculated_exposure_times = [self.calculate_exposure_time(metadata) for metadata in self.image_metadata]
            self.result_message = "Exposure times calculated successfully."
            return ImageCollectionOutput(images=self.input_images, metadata=self.image_metadata, exposure_times=calculated_exposure_times, result_message=self.result_message)
        except Exception as e:
            logger.error(f"Error in calculating exposure times: {e}")
            return ImageCollectionOutput(images=[], metadata=[], exposure_times=[], result_message="Error in calculating exposure times")

if __name__ == "__main__":
    # Example usage
    pass
