from datasets import load_dataset
import json
import time

""""""



def main():
    file_path = "/home/lukas/PycharmProjects/DaugBackend/test_files/xyxy_covid.json"
    with open(file_path, "r") as f:
        configs = json.load(f)
    data = load_dataset(configs)
    print(f"Data length: {len(data)}")
    print(f"Test data: {data[0]}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 10 ** 3:.03f}ms")

# without image loading (no width and height)-> 96.6 ms ~ 0.262 ms/it
# with image loading in cv2 (with width and height)-> 2963 ms ~ 8.07 ms/it
# without image loading (with width and height) -> 96.6ms ~ 0.262 ms/it

