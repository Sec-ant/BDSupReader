import sys
sys.path.append('../src')

from bdsupreader import BDSupReader

if __name__ == '__main__':  
    x = BDSupReader('../dataset/test_02.sup')
    for pic in x.subPictures:
        print(f'timestamp: {pic.startTimeStr} ~ {pic.endTimeStr} | duration: {pic.durationStr} | max alpha: {pic.maxAlpha}')
