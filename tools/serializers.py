import datetime

import numpy as np


def custom_serializer(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
