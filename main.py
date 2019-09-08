from bdsupreader import BDSupReader

if __name__ == '__main__':
    x = BDSupReader('./TestSupSet/test_02.sup')
    subPictures = x.subPictures
    for pic in subPictures:
        print(f'timestamp: {pic.startTimeStr} ~ {pic.endTimeStr} | duration: {pic.durationStr} | max alpha: {pic.maxAlpha}')
        #pic.screenImage.save(f'./Result/{pic.startTimeStr} ~ {pic.endTimeStr}.png')
