import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from ..extras.constants import DATA_CONFIG
from ..hparams import DataArguments


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    """ basic configs """
    load_from: Literal["script", "file"]
    dataset_name: str
    """ extra configs """
    subset: Optional[str] = None
    folder: Optional[str] = None
    ranking: bool = False

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_args: "DataArguments") -> List["DatasetAttr"]:
    if data_args.dataset_name is not None:
        dataset_names = [ds.strip() for ds in data_args.dataset_name.split(",")]
    else:
        dataset_names = []

    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(
                    data_args.dataset_dir, DATA_CONFIG), str(err))
            )
        dataset_info = None

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:
            dataset_attr = DatasetAttr("hf_hub", dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(
                "Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        if has_hf_url:
            dataset_attr = DatasetAttr(
                "hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr(
                "script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = DatasetAttr(
                "file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)

        dataset_list.append(dataset_attr)

    return dataset_list
