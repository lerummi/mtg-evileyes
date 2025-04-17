import uuid
import json
import torch
from typing import Generator


def get_embedding(patch_generator, device, model, feature_extractor) -> Generator:

    for uuid, info, patches in patch_generator:
        inputs = feature_extractor(images=patches, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        X = outputs.last_hidden_state.mean(axis=1).cpu().numpy()

        yield uuid, info, X


def uuid_from_dict(data: dict) -> uuid.UUID:
    # Convert the dictionary to a JSON string with sorted keys
    json_string = json.dumps(data, sort_keys=True)
    # Use a namespace UUID (can be uuid.NAMESPACE_DNS or a custom one)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, json_string))
