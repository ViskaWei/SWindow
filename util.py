def get_norm(streamTr, csv):
    d = csv.d
    for ii in range(streamTr.shape[0]//d+1):
        substream=streamTr[ii*d:(ii+1)*d]
        csv.accumulateVec(substream)
    norm = csv.get_norm()
    del csv
    return norm

def get_sketchs(stream, d, c, r, k, device):
    sketchs = torch.tensor([], dtype=torch.int64)
    for i in range(stream):
        streamSeg =stream[:i+1]
        if c is None: c=10*k 
        csv = CSNorm( d, c, r, k, device=device)
        norm = get_norm(streamSeg, csv)
        sketchs = update_sketchs(sketchs, norm)
    return sketchs

def update_sketchs(sketchs, norm):
    if len(sketch) < 3:
        return torch.cat((sketchs, streamTr[:2]), 0)        
    else:
        pass