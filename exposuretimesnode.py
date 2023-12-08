
import logging
from enum import Enum
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InputField, OutputField, InvocationContext, invocation, invocation_output
from invokeai.app.invocations.primitives import ImageField, ImageCollectionOutput
from pydantic import validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated Enum for simulated exposure times with descriptions
class SimulatedExposureTimes(Enum):
    LONG_EXPOSURE = (30, "Specialty night and low-light photos on a tripod")
    SILKY_WATER = (0.5, "Silky look to flowing water, landscape photos on tripod")
    MOTION_BLUR = (1/15, "Motion blur in background, hand-held photos with stabilization")
    HAND_HELD = (1/60, "Typical hand-held photos without substantial zoom")
    ACTION_FREEZE = (1/250, "Freeze everyday sports/action subject movement, telephoto lens")
    FAST_ACTION = (1/2000, "Freeze extremely fast, up-close subject motion")

@invocation_output('exposure_times_output')
class ExposureTimesOutput(BaseInvocationOutput):
    exposure_times: list[float] = OutputField(description="Exposure times for HDR image processing.")

    @validator('exposure_times', each_item=True)
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError("Exposure times must be positive numbers.")
        return v

@invocation("exposure_times", title="Exposure Times for HDR", tags=["hdr", "exposure"], category="image", version="0.1.0", use_cache=False)
class ExposureTimesInvocation(BaseInvocation):
    use_simulated_exposures: bool = InputField(description="Use simulated exposure times for HDR images.", default=False)
    user_exposure_times: list[float] = InputField(description="Custom exposure times provided by the user.", default=[])
    image_metadata: list[dict] = InputField(description="Metadata for each image", default=[])
    default_exposure_time: float = InputField(description="Default exposure time if not found in metadata", default=1/60)

    def calculate_exposure_time(self, metadata):
        # List of possible keys for exposure time in metadata
        possible_keys = ['EXIF ExposureTime', 'ExposureTime', 'Exposure Time']

        for key in possible_keys:
            exposure_time = metadata.get(key)
            if exposure_time:
                # Additional parsing logic can be added here if needed
                try:
                    # Handle different formats (e.g., '1/60', '0.016666666666666666')
                    if '/' in exposure_time:
                        numerator, denominator = exposure_time.split('/')
                        return float(numerator) / float(denominator)
                    return float(exposure_time)
                except ValueError:
                    logger.warning(f"Could not parse exposure time: {exposure_time}")

        # Default exposure time if not found or parsed
        return self.default_exposure_time

    def invoke(self, context: InvocationContext) -> ExposureTimesOutput:
        try:
            exposure_times = []
            if self.use_simulated_exposures:
                # Using the first value in the tuple (actual exposure time)
                exposure_times = [e.value[0] for e in SimulatedExposureTimes]
            elif self.user_exposure_times:
                if len(set(self.user_exposure_times)) != len(self.user_exposure_times):
                    raise ValueError("Duplicate values in custom exposure times are not allowed.")
                exposure_times = self.user_exposure_times
            else:
                exposure_times = [self.calculate_exposure_time(metadata) for metadata in self.image_metadata]

            return ExposureTimesOutput(exposure_times=exposure_times)
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    pass
