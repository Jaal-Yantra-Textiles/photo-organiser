import io
import numpy as np
from logger_util import logger

def serialize_array(arr):
    """
    Serialize a numpy array into bytes.
    """
    serialized = io.BytesIO()
    np.save(serialized, arr, allow_pickle=False)
    return serialized.getvalue()

def deserialize_array(serialized_arr):
    if serialized_arr is None:
        return None
    try:
        return np.load(io.BytesIO(serialized_arr), allow_pickle=False)
    except EOFError:
        logger.error(f"EOFError encountered. Serialized data: {serialized_arr}")
        return None
    
def bytes_to_float(byte_val):
    if byte_val is None:
        return None
    return np.frombuffer(byte_val, dtype=np.float32)[0]



