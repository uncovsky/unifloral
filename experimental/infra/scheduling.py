# Some schedules for lagrangian parameters

def linear_schedule(start, end, max_steps);
    """Linear schedule from start to end with max steps."""
    def schedule(step):
        if step >= max_steps:
            return end
        else:
            return start + (end - start) * (step / max_steps)
    return schedule

    
def constant_schedule(value):
    """Constant schedule."""
    def schedule(step):
        return value
    return schedule


