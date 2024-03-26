

def sum_dict(los, ignore=""):
    temp = 0
    for l in los:
        if l != ignore:
            temp += los[l]
    return temp