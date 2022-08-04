from pyais import chunk_read

if __name__ == '__main__':
    # chunk_read.read_ais_data_region(input="../data/1.csv", output='../data/test.csv')  # 此处填写csv路径
    chunk_read.read_ais_data_mmsi(input="../data/1.csv", output='../data/test.csv', mmsi='244726000')  # 此处填写csv路径
