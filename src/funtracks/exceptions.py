class InvalidActionError(RuntimeError):
    """Raised when attempting an action that by itself is invalid, but that can
    optionally be forced by removing conflicting edges."""

    def __init__(self, message: str, forceable: bool = False):
        """
        Args:
            message (str): The error message.
            forceable (bool): Whether the action can be forced. Defaults to False.
        """
        super().__init__(message)
        self.forceable = forceable
