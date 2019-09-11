import sys
import os
sys.path.append('../src')

from bdsupreader import BDSupReader

if __name__ == '__main__':
    for f in os.listdir('../dataset/'):
        x = BDSupReader(f'../dataset/{f}')
        screenPath = f'../temp/Screen Images/{f.split(".")[0]}'
        imagePath = f'../temp/Images/{f.split(".")[0]}'
        os.makedirs(screenPath)
        os.makedirs(imagePath)
        for s in x.subPictures:
            #s.screenImage.save(f'{screenPath}/{s.startTimeStr} ~ {s.endTimeStr}.png')
            for img in s.image:
                img['data'].save(f'{imagePath}/{s.startTimeStr} ~ {s.endTimeStr}_{img["id"]}.png')
