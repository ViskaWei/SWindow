from scapy.all import rdpcap
from tqdm import tqdm

# DATASET ='/home/swei20/SymNormSlidingWindows/data/testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'

def get_packet_iter(DATASET):
    packets = rdpcap(DATASET)
    packetsIter = iter(packets)
    return packetsIter

def get_packet_stream(DATASET, name, num=None):
    stream = []
    print('reading data please wait')
    packets = rdpcap(DATASET)
    if num is None:
        num = len(packets)
    if name == 'len':
         query = lambda x: x.len
    for i in tqdm(range(num)):
        stream.append(query(packets[i]))
    return stream