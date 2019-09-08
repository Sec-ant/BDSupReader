# BDSupReader
### Intro
A Blu-Ray PGS subtitle (.sup) parser. Still under development.

### Known Issues
:x: The minimum unit for the resource (i.e. image and palette) should be epoch but not display set.

### Future Work
- [ ] Re-assign timestamps by provding a mapping function.
- [ ] Apply filters on RLE data, which are the subtitle pictures.
- [ ] Convert text format strings to RGBA image.
- [ ] Encode RGBA image to RLE data.
- [ ] Convert subrip or ass/ssa subtitles to pgs subtitles.

### Reference
* [BDSup2Sub](https://github.com/mjuhasz/BDSup2Sub) by @EzraBC
* [pgsreader](https://github.com/EzraBC/pgsreader) by @mjuhasz
* [Presentation Graphic Stream (SUP files) BluRay Subtitle Format](http://blog.thescorpius.com/index.php/2017/07/15/presentation-graphic-stream-sup-files-bluray-subtitle-format/)
* [US Patent US 20090185789 A1](https://encrypted.google.com/patents/US20090185789?cl=da)
* [US 7912305 B1 patent](https://www.google.com/patents/US7912305)
* [YCbCr - Wikipedia](https://en.wikipedia.org/wiki/YCbCr)
