class SfmError(Exception):
    """SfM 项目的基础异常类"""
    pass

class InsufficientMatchesError(SfmError):
    pass

class RegisterError(SfmError):
    pass

class TriangulateError(SfmError):
    pass

class PnPError(SfmError):
    pass

class DegeneracyError(SfmError):
    pass