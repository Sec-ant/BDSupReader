import datetime
from bdsupreader import BDSupReader

if __name__ == '__main__':
    x = BDSupReader('./TestSupSet/test_01.sup')
    subPictures = x.subPictures
    for i, pic in enumerate(subPictures):
        print((str(datetime.timedelta(milliseconds = pic.startTimems)), str(datetime.timedelta(milliseconds = pic.endTimems))))
