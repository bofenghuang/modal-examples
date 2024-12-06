# coding=utf-8
# Copyright 2023  Bofeng Huang

import json
import io
import os
import threading

lck = threading.Lock()

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def jsonl_dump(obj, f, mode="a", default=str, ensure_ascii=False):
    f = _make_w_io_base(f, mode)
    if isinstance(obj, dict):
        f.write(json.dumps(obj, default=default, ensure_ascii=ensure_ascii) + "\n")
    elif isinstance(obj, list):
        for item in obj:
            f.write(json.dumps(item, default=default, ensure_ascii=ensure_ascii) + "\n")
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def thread_safe_jsonl_dump(obj, f, **kwargs):
    # acquire the lock
    with lck:
        jsonl_dump(obj, f, **kwargs)
