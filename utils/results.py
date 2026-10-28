import json
import os
from datetime import datetime


def result_file_path(args):
    exp_name = args.exp_name or "default"
    result_dir = os.path.join(args.result_path, exp_name)
    os.makedirs(result_dir, exist_ok=True)

    if args.turned:
        return os.path.join(result_dir, f"turnd-{args.turned}.json")
    return os.path.join(result_dir, "metres.json")


def save_json_result(args, payload):
    path = result_file_path(args)
    result = {
        "exp_name": args.exp_name,
        "turned": args.turned,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **payload,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path
