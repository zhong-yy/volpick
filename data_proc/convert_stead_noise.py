from volpick.data.convert import extract_stead_noise
import time
import datetime

if __name__ == "__main__":
    t1 = time.perf_counter()
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/stead_noise"
    extract_stead_noise(dest_dir=dest_dir, n_traces=200000, chunk="_STEAD_noise")
    # extract_stead_noise(dest_dir=dest_dir, use_all_noise=True, chunk="_STEAD_noise")
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(f"Finish conversion. Running time: {running_time}")
