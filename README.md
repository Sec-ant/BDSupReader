# BDSupReader
Original File | Screen Images (converted to APNG)
:-----------: | :-------------------------------:
[:clapper:](https://github.com/Sec-ant/BDSupReader/blob/master/TestSupSet/test_04.sup?raw=true) | <a target="_blank" rel="noopener noreferrer" href="https://github.com/Sec-ant/BDSupReader/blob/master/Screen%20Image%20Example.png?raw=true"><img src="https://github.com/Sec-ant/BDSupReader/raw/master/Screen%20Image%20Example.png?raw=true" alt="Screen Image Example.png" width="50%"></a>

## Intro
A BluRay PGS subtitle (.sup) parser, which is developed as a helper CLI tool to sync pgs subtitles timewise. More features will be considered when the basic requirements are met.

This project is still under development, comments and advices are always appreciated!

## Features
* Acquire all the information that the subtitle carries, please refer to [US Patent US 20090185789 A1](https://encrypted.google.com/patents/US20090185789?cl=da) for details;
* Acquire the start and end timestamp for each caption;
* Acquire and save images of each caption;
* Acquire and save the screen image, which means not only the images themselves, but also their placements on the screen, of each caption;
* Assign weights for each screen image according to their transparency value. (Fade in or out captions have small weights relativley)

## Future Work
* Re-assign timestamps by provding a mapping function, taking index and original timestamp as arguments;
* Apply filters on RLE data (i.e. the subtitle image);
* Convert text format strings to RGBA image;
* Encode RGBA image to RLE data;
* Convert subrip or ASS/SSA subtitles to pgs subtitles;
* OCR. (Maybe not, for this feature is too heavy)

## Known Issues
Please refer to the [Open Issues Page](https://github.com/Sec-ant/BDSupReader/issues?q=is%3Aopen).

## Reference
* [BDSup2Sub](https://github.com/mjuhasz/BDSup2Sub) by @EzraBC
* [pgsreader](https://github.com/EzraBC/pgsreader) by @mjuhasz
* [Presentation Graphic Stream (SUP files) BluRay Subtitle Format](http://blog.thescorpius.com/index.php/2017/07/15/presentation-graphic-stream-sup-files-bluray-subtitle-format/)
* [US Patent US 20090185789 A1](https://encrypted.google.com/patents/US20090185789?cl=da)
* [US 7912305 B1 patent](https://www.google.com/patents/US7912305)
* [YCbCr - Wikipedia](https://en.wikipedia.org/wiki/YCbCr)
