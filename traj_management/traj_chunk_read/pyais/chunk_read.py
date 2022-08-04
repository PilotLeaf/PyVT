import pandas as pd
import time


def suply10(x):
    return x * 10.0


def read_ais_data_region(input: str, output: str, logleftup: float = 122.0, logrightdown: float = 123.0,
                         latleftup: float = 32.0, latrightdown: float = 31.0, dtst: str = '2019-01-01 00:00:01',
                         dtend: str = '2020-01-01 00:00:01'):
    '''
    :param input: filename of the input file
    :param output: filename of the output file
    :param logleftup: longitude of upper left corner
    :param logrightdown: longitude of lower right corner
    :param latleftup: longitude of upper left corner
    :param latrightdown: latitude of lower right corner
    :param dtst: start time
    :param dtend: end time
    :return: null
    '''
    cols = read_data_nm(input, 5)
    inputfile = open(input, 'rb')
    data = pd.read_csv(inputfile, sep=',', iterator=True)
    timearrayst = time.strptime(dtst, "%Y-%m-%d %H:%M:%S")
    timearrayend = time.strptime(dtend, "%Y-%m-%d %H:%M:%S")
    timestampst = time.mktime(timearrayst)  # timestamp
    timestampend = time.mktime(timearrayend)
    loop = True
    chunkSize = 10000  # The number of reads per chunk
    chunks = []
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunk.columns = cols
            chunked = chunk[(chunk['DRGPSTIME'] > timestampst) & (chunk['DRGPSTIME'] < timestampend) & (
                    chunk['DRLONGITUDE'] > logleftup) & (chunk['DRLONGITUDE'] < logrightdown) & (
                                    chunk['DRLATITUDE'] < latleftup) & (chunk['DRLATITUDE'] > latrightdown)]
            cn = chunked.fillna(value=0)
            # cn['DRTRUEHEADING'] = cn['DRTRUEHEADING'].transform(suply10)
            chunks.append(cn)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    rdata = pd.concat(chunks, ignore_index=True)
    rdata.to_csv(output, index=False)  # 输出csv
    print("Finished.")
    return


def read_ais_data_callsign(input: str, output: str, callsign: str = ''):
    '''
    :param input: filename of the input file
    :param output: filename of the output file
    :param callsign: CALLSIGN
    :return: null
    '''
    cols = read_data_nm(input, 5)
    inputfile = open(input, 'rb')
    data = pd.read_csv(inputfile, sep=',', iterator=True)
    loop = True
    chunkSize = 10000  # The number of reads per chunk
    chunks = []
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunk.columns = cols
            chunk['CALLSIGN'] = chunk['CALLSIGN'].map(lambda x: str(x))
            chunked = chunk[(str(chunk['CALLSIGN']) == callsign)]
            cn = chunked.fillna(value=0)
            chunks.append(cn)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    rdata = pd.concat(chunks, ignore_index=True)
    rdata.to_csv(output, index=False)  # 输出csv
    print("Finished.")
    return


def read_ais_data_mmsi(input: str, output: str, mmsi: str = ''):
    '''
    :param input: filename of the input file
    :param output: filename of the output file
    :param mmsi: DRMMSI
    :return: null
    '''
    cols = read_data_nm(input, 5)
    inputfile = open(input, 'rb')
    data = pd.read_csv(inputfile, sep=',', iterator=True)
    loop = True
    chunkSize = 10000  # The number of reads per chunk
    chunks = []
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunk.columns = cols
            chunk['DRMMSI'] = chunk['DRMMSI'].map(lambda x: str(x))
            chunked = chunk[(chunk['DRMMSI'] == mmsi)]
            cn = chunked.fillna(value=0)
            chunks.append(cn)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    rdata = pd.concat(chunks, ignore_index=True)
    rdata.to_csv(output, index=False)  # 输出csv
    print("Finished.")
    return


def read_data_nm(input_file, nm):
    '''
    The function is to print the first * line of the file
    :param input_file: filename
    :param nm: row number
    :return: null
    '''
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    data = pd.read_csv(input_file, nrows=nm)
    print(data)
    return data.columns
