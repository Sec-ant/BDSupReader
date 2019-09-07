import datetime
from bdsupreader import BDSupReader

if __name__ == '__main__':
    x = BDSupReader('./TestSupSet/test_04.sup')
    subPictures = x.subPictures
    for pic in subPictures:
        pic.screenImage.save(f'./Result/{str(datetime.timedelta(milliseconds = pic.startTimems))} - {str(datetime.timedelta(milliseconds = pic.endTimems))}.png')
