from scapy.all import rdpcap
from tqdm import tqdm

# DATASET ='/home/swei20/SymNormSlidingWindows/data/testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'

def get_packet_iter(DATASET):
    packets = rdpcap(DATASET)
    packetsIter = iter(packets)
    return packetsIter

def get_packet_stream(DATASET, name, m=None):
    stream = []
    print('reading data please wait')
    packets = rdpcap(DATASET)
    if m is None:
        m = len(packets)
    query = get_query_fn(name)
    for i in tqdm(range(m)):
        stream.append(query(packets[i]))
    return stream, m

def get_query_fn(name):
    if name == 'len':
        return lambda x: x.len
    if name == 'src':
        return None