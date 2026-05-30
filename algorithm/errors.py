class SfmError(Exception):
    """Base exception for recoverable SfM pipeline failures."""
    pass

class InsufficientMatchesError(SfmError):
    """Raised when a frame pair has too few reliable feature matches."""
    pass

class RegisterError(SfmError):
    """Raised when a frame cannot be registered into the current map."""
    pass

class TriangulateError(SfmError):
    """Raised when triangulation fails or returns invalid geometry."""
    pass

class PnPError(SfmError):
    """Raised when pose estimation from 2D-3D correspondences fails."""
    pass

class DegeneracyError(SfmError):
    """Raised for geometrically degenerate configurations."""
    pass
