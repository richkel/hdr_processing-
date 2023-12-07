from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
import logging

logger = logging.getLogger(__name__)

@invocation("User_Input_Node", title="User Input for HDR", tags=["hdr", "input"], category="input", version="1.0.0", use_cache=False)
class UserInputinvocation(BaseInvocation):
    """
    Custom node for user input in HDR image processing. This node allows users to input a custom exposure time.
    """
    custom_exposure_time: float = InputField(description="Custom exposure time entered by the user.", default=1/60)

    def invoke(self, context: InvocationContext) -> dict:
        """
        Invokes the User Input Node and validates the user's input.

        Args:
            context (InvocationContext): The context in which the node is invoked.

        Returns:
            dict: A dictionary containing the 'custom_exposure_time' or None if an error occurs.
        """
        logger.info("User Input Node invoked with exposure time: %s", self.custom_exposure_time)
        try:
            # Validate the user input
            if not 0 < self.custom_exposure_time <= 1/15:  # assuming 1/15 as an upper limit for exposure time
                raise ValueError("Exposure time must be positive and less than or equal to 1/15.")
            return {"custom_exposure_time": self.custom_exposure_time}
        except ValueError as e:
            logger.error("ValueError in User Input Node: %s", e)
            return {"custom_exposure_time": None}
        except Exception as e:
            logger.error("Unexpected error in User Input Node: %s", e)
            return {"custom_exposure_time": None}
