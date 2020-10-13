def kept_sketchs_id_old(norms):
    l= len(norms)
    if l <3: return list(range(l))
    result = set([])
    i = 0
    while i < l:
        result.add(i)
        found = False
        j = i+1
        while j < l:
            if norms[j] > norms[i]:
                i = j -1
                found = True
                break
            if (abs(norms[j]-norms[i]) >= (norms[i] / 2.0)):
                if j != i+1:
                    result.add(j-1)                    
                i = j - 1
                found = True
                break
            j+=1
        if not found and i != l-1: 
            result.add(l-1)
            return result
        i+=1
    return result


def kept_sketchs_id(norms):
    l= len(norms)
    keep = []
    i = 0
    while i < l:
        keep.append(i)
        if i == l-2: return keep +[i+1]
        found = False
        j = i+1
        while j < l:
            if (abs(norms[j]-norms[i]) >= (norms[i] / 2.0)):
                i = j if j-1 == i else j-1                
                found = True
                break
            j+=1
        if not found: return keep + [l-1]
    return keep

# norms = [100, 95,93,18, 76, 43, 21, 18, 15, 2, 1, 1]
# # {0, 2, 3, 4, 5, 6, 8, 9, 10, 11}
# print(kept_sketchs_id2(norms))