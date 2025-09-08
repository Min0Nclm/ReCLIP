import torch

__all__ = ["to_dict"]

def to_dict(x):
    if isinstance(x, dict):
        if "group_id" not in x:
            x["group_id"] = torch.zeros(
                (x["img"].size(0),), dtype=x["img"].dtype, device=x["img"].device
            )
        return x

    group = x[2] if len(x) == 3 else None
    return {"img": x[0], "lbl": x[1], "group_id": group}
