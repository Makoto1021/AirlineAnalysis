from prepare_data.test import test_func
test_func()

from get_data.get_data import get_data
from prepare_data.prepare_data_dep import prepare_data

def main():
    get_data()
    prepare_data()

if __name__ == "__main__": 
    main()