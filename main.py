from bdsupreader import BDSupReader

if __name__ == '__main__':
    x = BDSupReader('./TestSupSet/test_04.sup')
    subPictures = x.subPictures
    for pic in subPictures:
        pic.screenImage.save(f'./Result/{pic.startTimeStr} ~ {pic.endTimeStr}.png')
