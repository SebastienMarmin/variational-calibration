class VcalException(Exception):
    """Base Exception"""

class VcalNaNLossException(VcalException):
    """When the loss stop being a number"""

class VcalTimeOutException(VcalException):
    """When reaching available time"""


