

from b2sdk.v2 import *
import os
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_random_exponential

import uuid

def get_uuid():
    return str(uuid.uuid4())


# init b2
info = InMemoryAccountInfo()
b2_api = B2Api(info)
application_key_id = os.environ['B2_KEY_ID']
application_key = os.environ['B2_APP_KEY']
b2_api.authorize_account("production", application_key_id, application_key)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def b2_upload_file_and_get_url(file_path=None, bytes=None, buffer=None):
    if file_path is not None:
        local_file_path = file_path
        fn = Path(file_path).stem
        ext = Path(file_path).suffix  # suffix has dot!!!
    elif buffer is not None:
        fn = get_uuid()  # Generate a unique name for the buffer
        ext = '.bin'  # Default extension for buffer data
    else:
        raise ValueError("Either file_path or buffer must be provided.")

    b2_file_name = f'{fn}{ext}'
    file_info = None

    bucket = b2_api.get_bucket_by_name('godmodeaigendanims')

    if bytes is not None:
        ret = bucket.upload_bytes(
            data_bytes=bytes,
            file_name=b2_file_name,
            file_infos=file_info,
        )
    elif buffer is not None:
        ret = bucket.upload_bytes(
            data_bytes=buffer.read(),
            file_name=b2_file_name,
            file_infos=file_info,
        )
    else:
        ret = bucket.upload_local_file(
            local_file=local_file_path,
            file_name=b2_file_name,
            file_infos=file_info,
        )
    url = b2_api.get_download_url_for_fileid(ret.id_)
    return url