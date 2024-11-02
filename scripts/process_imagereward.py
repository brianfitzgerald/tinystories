from typing import Dict, cast

import webdataset as wds
from datasets import DatasetDict, load_dataset
from fire import Fire
from tqdm import tqdm

COLUMNS = [
    "image_text_alignment_rating",
    "fidelity_rating",
    "overall_rating",
    "classification",
    "rank",
    "image_amount_in_total",
]


def main():

    out_wds = wds.ShardWriter(
        "/weka/home-brianf/imagereward_cache/shard-%06d.tar", maxcount=1000
    )

    # ImageReward

    dataset = cast(
        DatasetDict,
        load_dataset("THUDM/ImageRewardDB", "4k", verification_mode="no_checks"),
    )
    ds_iter = iter(dataset["train"])
    iter_tqdm = tqdm(total=len(dataset["train"]))

    while iter_tqdm.n < iter_tqdm.total:
        try:
            sample: Dict = next(ds_iter)  # type: ignore
        except StopIteration:
            print("End of dataset")
            break
        except Exception as e:
            print(e)
            continue
        image = sample["image"]

        json_data = {}
        for column in COLUMNS:
            json_data[column] = sample[column]
        out_wds.write({"png": image, "json": json_data, "__key__": str(iter_tqdm.n)})
        iter_tqdm.update(1)


if __name__ == "__main__":
    Fire(main)
