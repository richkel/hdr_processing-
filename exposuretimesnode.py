from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InputField, OutputField, InvocationContext, invocation, invocation_output
from pydantic import validator
import logging

logger = logging.getLogger(__name__)

@invocation_output('exposure_times_output')
class ExposureTimesOutput(BaseInvocationOutput):
    exposure_times: list[float] = OutputField(description="Exposure times for HDR image processing.")

    # Ensure that exposure times are positive numbers
    @validator('exposure_times', each_item=True)
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError("Exposure times must be positive numbers.")
        return v

@invocation(
    "exposure_times",
    title="Exposure Times for HDR",
    tags=["hdr", "exposure"],
    category="image",
    version="0.1.0",
    use_cache=False,
)
class ExposureTimesInvocation(BaseInvocation):
    """
    A node to calculate and provide exposure times for HDR image processing.
    This node can use either simulated exposure times or user-provided exposure times.
    """
    use_simulated_exposures: bool = InputField(description="Use simulated exposure times for HDR images.", default=False)
    user_exposure_times: list[float] = InputField(description="Custom exposure times provided by the user.", default=[])

    def invoke(self, context: InvocationContext) -> ExposureTimesOutput:
        try:
            if self.use_simulated_exposures:
                exposure_times = [1/30, 1/60, 1/120]
            else:
                if not self.user_exposure_times:
                    raise ValueError("No custom exposure times provided.")
                if len(set(self.user_exposure_times)) != len(self.user_exposure_times):
                    raise ValueError("Duplicate values in custom exposure times are not allowed.")
                exposure_times = self.user_exposure_times

            # Validate using Pydantic model
            return ExposureTimesOutput(exposure_times=exposure_times)
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            # Consider raising an HTTPException for better API error handling
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Handle unexpected errors appropriately
            raise

# Additional optimizations for performance can be applied based on specific use-cases and context.
