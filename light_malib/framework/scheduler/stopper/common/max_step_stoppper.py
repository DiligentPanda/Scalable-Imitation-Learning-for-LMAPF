from light_malib.registry import registry

@registry.registered(registry.STOPPER, "max_step_stopper")
class MaxStepStopper:
    def __init__(self, **kwargs):
        self.max_steps = kwargs["max_steps"]

    def stop(self, **kwargs):
        step = kwargs["step"]
        if step >= self.max_steps:
            return True
        return False
