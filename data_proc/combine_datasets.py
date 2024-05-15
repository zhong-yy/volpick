from volpick.data.utils import assemble_datasets, generate_chunk_file
import seisbench.data as sbd
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    assemble_datasets(
        datasets_dir="/mnt/DATA2/YiyuanZhong/my_datasets_seisbench",
        datasets=[
            "alaska",
            "hawaii1986to2009",
            "hawaii2012to2021",
            "japan_vol_lp",
            "japan_vt",
            "stead_noise",
        ],
        dest_dir="/mnt/DATA2/YiyuanZhong/my_datasets_seisbench/lp_dataset",
    )
    generate_chunk_file("/mnt/DATA2/YiyuanZhong/my_datasets_seisbench/lp_dataset")

    # data_path = "/home/zhongyiyuan/DATA/my_datasets_seisbench/lp_dataset"
    # dataset = sbd.WaveformDataset(
    #     Path(data_path),
    #     sampling_rate=100,
    #     component_order="ZNE",
    #     dimension_order="NCW",
    #     cache="full",
    # )
    # trace_with_spikes_index = np.flatnonzero(
    #     dataset.metadata["trace_has_spikes"].to_numpy()
    # )
    # assert len(trace_with_spikes_index) == 0

    # dataset_lp=dataset.filter((dataset.metadata["source_type"]=="lp"),inplace=False)
    # dataset_rg=dataset.filter((dataset.metadata["source_type"]!="lp")&(dataset.metadata["source_type"]!="noise"),inplace=False)
    # dataset_noise=dataset.filter(dataset.metadata["source_type"]=="noise",inplace=False)

    # print(f"{len(dataset_lp)} lp traces")
    # print(f"{len(dataset_rg)} regular earthquake traces")
    # print(f"{len(dataset_noise)} noise traces")
