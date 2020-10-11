import numpy as np
from tqdm import tqdm
from scapy.all import sniff,rdpcap
from ipaddress import IPv4Address as ipv4

# DATAPATH ='/home/swei20/SymNormSlidingWindows/data/testdata/equinix-nyc.dirA.20190117-131558.UTC.anon.pcap'


def get_packet_stream(DATAPATH, name, m=-1):
    stream = []
    query = get_query_fn(name, stream)
    print('sniffing packet please wait')
    if m is None: m = -1
    sniff(filter="tcp",offline= DATAPATH,count= m, store = 0,\
        prn=lambda x: stream.append(x.len))
    return stream, len(stream)


def load_all_packets(DATAPATH, query):
    packets = rdpcap(DATAPATH)
    stream = []
    for i in tqdm(range(len(stream))):
        stream.append(query(packets[i]))
    return packets

def get_packet_iter(packets):
    packetsIter = iter(packets)
    return packetsIter

def get_traffic_stream(NAME, DATASET, m = None):
    out=[]
    if NAME == 'src':
        query = lambda x: out.append(int(ipv4(x.src)))
        level = 'ip'
    if NAME == 'dst':
        query = lambda x: out.append(int(ipv4(x.dst)))
        level = 'ip'
    elif NAME =='sport':
        query = lambda x: out.append(x.sport)
        level = 'ip'
    elif NAME =='dport':
        level = 'ip'
        query = lambda x: out.append(x.dport)
    elif NAME =='len':
        level = 'ip'
        query = lambda x: out.append(x.len)      
    sniff(filter=level, offline=DATASET, prn=query,count= m, store = 0)
    print(m, out)
    if m >10:
        np.savetxt(f'/home/swei20/SymNormSlidingWindows/data/stream/traffic_{NAME}.txt', out)

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