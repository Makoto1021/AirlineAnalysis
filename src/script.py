from get_data.get_data import get_data
from prepare_data.prepare_data_dep import prepare_data
import time


def main():
    start_time = time.time()
    get_data()
    middle_time = time.time()
    print("Execution time for scraping:", middle_time - start_time)
    prepare_data()
    end_time = time.time()
    print("Total execution time:", end_time - start_time)

if __name__ == "__main__": 
    main()