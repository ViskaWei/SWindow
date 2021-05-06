import numpy as np
from tqdm import tqdm
from scapy.all import sniff,rdpcap
from ipaddress import IPv4Address as ipv4

# DATAPATH ='/home/swei20/SymNormSlidingWindows/data/testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'


def get_packet_stream(DATAPATH, ftr, m=-1):
    stream = []
    query = get_query_fn(ftr, stream)
    print('sniffing packet please wait')
    if m is None: m = -1
    sniff(offline= DATAPATH,count= m, store = 0,\
        prn=lambda x: stream.append(x.len))
    return stream

def get_sniffed_stream(ftr, DATASET, m = None, save=False):
    out=[]
    if ftr == 'src':
        query = lambda x: out.append(int(ipv4(x.src)))
    elif ftr == 'dst':
        query = lambda x: out.append(int(ipv4(x.dst)))
    elif ftr =='sport':
        query = lambda x: out.append(x.sport)
    elif ftr =='dport':
        query = lambda x: out.append(x.dport)
    elif ftr =='len':
        query = lambda x: out.append(x.len)      
    sniff(offline=DATASET, prn=query,count= m, store = 0)
    print(m, out[:5])
    if save:
        if (m <= 1000) and (m >10 ):
            np.savetxt(f'/home/swei20/SymNormSlidingWindows/test/data/stream/traffic_{ftr}_m{m}.txt', out)
        elif (m >1000) or (m ==-1):
            np.savetxt(f'/home/swei20/SymNormSlidingWindows/data/stream/traffic_{ftr}.txt', out)
    return out


def get_query_fn(name, out):
    if name == 'len':
        def query_len(x, out):
            try:
                out.append(x.len)
            except:
                pass
        return lambda x: query_len(x, out)
    if name == 'src':
        return lambda x: out.append(x.src)

def test_sniff(DATASET):
    sniff(offline=DATASET, prn=lambda x:x.len, count=10, store = 0)

def load_all_packets(DATAPATH, query):
    packets = rdpcap(DATAPATH)
    stream = []
    for i in tqdm(range(len(stream))):
        stream.append(query(packets[i]))
    return packets

def get_packet_iter(packets):
    packetsIter = iter(packets)
    return packetsIter