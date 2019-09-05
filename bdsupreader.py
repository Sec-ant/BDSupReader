import cProfile, pstats, io
from enum  import Enum
from bufferedrandomplus import BufferedRandomPlus
from pprint import pprint
import time
import io
import itertools
import numpy as np
from PIL import Image

class InvalidSegmentError(Exception):
    pass

class SEGMENT_TYPE(Enum):

    PDS = b'\x14'
    ODS = b'\x15'
    PCS = b'\x16'
    WDS = b'\x17'
    END = b'\x80'

class COMPOSITION_STATE(Enum):

    NORMAL = b'\x00'
    ACQUISITION_POINT = b'\x40'
    EPOCH_START = b'\x80'
    EPOCH_CONTINUE = b'\xC0'
    
class SEQUENCE(Enum):

    FIRST = b'\x40'
    LAST = b'\x80'
    FISRT_LAST = b'\xC0'

class SupReader:
    
    def __init__(self, filePath, bufferSize=1024*1024, verbose=False):
        self.filePath = filePath
        self.bufferSize = bufferSize
        self.verbose = verbose

    def iterSegments(self):
        ds = []
        ep = []
        self._displaySets = []
        self._epochs = []
        with open(self.filePath, 'r+b', buffering = self.bufferSize) as _:
            self._stream = BufferedRandomPlus(_)
            stream = self._stream
            while stream.peekByte():
                segment = Segment(stream)
                yield segment
                ds.append(segment)
                if segment.type == SEGMENT_TYPE.END:
                    dsObj = DisplaySet(ds)
                    ep.append(dsObj)
                    self._displaySets.append(dsObj)
                    ds = []
                elif ep \
                and segment.type == SEGMENT_TYPE.PCS \
                and segment.data.compositionState == COMPOSITION_STATE.EPOCH_START:
                    self._epochs.append(Epoch(ep))
                    ep = []
            if ds:
                print("Warning: [Read Stream] The last epoch lacks END segment")
            self._epochs.append(Epoch(ep))
                
    
    def iterDisplaySets(self):
        ds = []
        ep = []
        self._segments = []
        self._epochs = []
        with open(self.filePath, 'r+b', buffering = self.bufferSize) as _:
            self._stream = BufferedRandomPlus(_)
            stream = self._stream
            while stream.peekByte():
                segment = Segment(stream)
                self._segments.append(segment)
                ds.append(segment)
                if segment.type == SEGMENT_TYPE.END:
                    dsObj = DisplaySet(ds)
                    yield dsObj
                    ep.append(dsObj)
                    ds = []
                elif ep \
                and segment.type == SEGMENT_TYPE.PCS \
                and segment.data.compositionState == COMPOSITION_STATE.EPOCH_START:
                    self._epochs.append(Epoch(ep))
                    ep = []
            if ds:
                print("Warning: [Read Stream] The last epoch lacks END segment")
            self._epochs.append(Epoch(ep))

    def iterEpochs(self):
        ds = []
        ep = []
        self._segments = []
        self._displaySets = []
        with open(self.filePath, 'r+b', buffering = self.bufferSize) as _:
            self._stream = BufferedRandomPlus(_)
            stream = self._stream
            while stream.peekByte():
                segment = Segment(stream)
                self._segments.append(segment)
                ds.append(segment)
                if segment.type == SEGMENT_TYPE.END:
                    dsObj = DisplaySet(ds)
                    ep.append(dsObj)
                    self._displaySets.append(dsObj)
                    ds = []
                elif ep \
                and segment.type == SEGMENT_TYPE.PCS \
                and segment.data.compositionState == COMPOSITION_STATE.EPOCH_START:
                    yield Epoch(ep)
                    ep = []
                    flag = True
            if ds:
                print("Warning: [Read Stream] The last epoch lacks END segment")
            yield Epoch(ep)

    @property
    def segments(self):
        if not hasattr(self, '_segments'):
            self._segments = list(self.iterSegments())
        return self._segments

    @property
    def displaySets(self):
        if not hasattr(self, '_displaySets'):
            self._displaySets = list(self.iterDisplaySets())
        return self._displaySets

    @property
    def epochs(self):
        if not hasattr(self, '_epochs'):
            self._epochs = list(self.iterEpochs())
        return self._epochs

class PresentationCompositionSegment:
    
    class CompositionObject:

        def __init__(self, stream, parent):
            self._parent = parent
            self.objectID = stream.readUShort()
            self.windowID = stream.readUChar()
            self.cropped = bytes([stream.readByte()[0] & b'\x40'[0]]) == b'\x40'
            self.xPos = stream.readUShort()
            self.yPos = stream.readUShort()
            if self.cropped:
                self.cropXPos = stream.readUShort()
                self.cropYPos = stream.readUShort()
                self.cropWidth = stream.readUShort()
                self.cropHeight = stream.readUShort()

        @property
        def parent(self):
            return self._parent

    def __init__(self, stream, parent):
        self._parent = parent
        self.width = stream.readUShort()
        self.height = stream.readUShort()
        self.frameRate = stream.readByte()
        self.compositionNumber = stream.readUShort()
        self.compositionState = COMPOSITION_STATE(stream.readByte())
        self.paletteUpdate = bytes([stream.readByte()[0] & b'\x80'[0]]) == b'\x80'
        self.paletteID = stream.readUChar()
        self.numberOfCompositionObjects = stream.readUChar()
        self.compositionObjects = self.getCompositionObjects(stream)
        
    def getCompositionObjects(self, stream):
        comps = []
        stop = stream.offset + len(self.parent) - 11
        while stream.offset < stop:
            comps.append(self.CompositionObject(stream, self))
        numberOfCompositionObjects = len(comps)
        if numberOfCompositionObjects != self.numberOfCompositionObjects:
            print('Warning: [PCS] Number of composition objects asserted ({:d}) '
                    'does not match the amount found ({:d}). '
                    'The attribute will be reassigned'
                    .format(self.numberOfCompositionObjects, numberOfCompositionObjects))
            self.numberOfCompositionObjects = numberOfCompositionObjects
        return comps

    @property
    def parent(self):
        return self._parent

class WindowDefinitionSegment:
    
    class WindowObject:
        
        def __init__(self, stream, parent):
            self._parent = parent
            self.windowID = stream.readUChar()
            self.xPos = stream.readUShort()
            self.yPos = stream.readUShort()
            self.width = stream.readUShort()
            self.height = stream.readUShort()
        
        @property
        def parent(self):
            return self._parent

    def __init__(self, stream, parent):
        self._parent = parent
        self.numberOfWindows = stream.readUChar()
        self.windowObjects = self.getWindowObjects(stream)

    def getWindowObjects(self, stream):
        windows = []
        stop = stream.offset + len(self.parent) - 1
        while stream.offset < stop:
            windows.append(self.WindowObject(stream, self))
        numberOfWindows = len(windows)
        if numberOfWindows != self.numberOfWindows:
            print('Warning: [WDS] Number of windows asserted ({:d}) '
                    'does not match the amount found ({:d}). '
                    'The attribute will be reassigned'
                    .format(self.numberOfWindows, numberOfWindows))
            self.numberOfWindows = numberOfWindows
        return windows

    @property
    def parent(self):
        return self._parent

class PaletteDefinitionSegment:
    
    def __init__(self, stream, parent):
        self._parent = parent
        self.paletteID = stream.readUChar()
        self.version = stream.readUChar()
        self.palette = self.getPalette(stream)

    def getPalette(self, stream):
        palette = [[16, 128, 128, 0]] * 256
        stop = stream.offset + len(self.parent) - 2
        while stream.offset < stop:
            entry = stream.readUChar()
            y = stream.readUChar()
            cr = stream.readUChar()
            cb = stream.readUChar()
            alpha = stream.readUChar()
            palette[entry] = [y, cr, cb, alpha]
        return palette

    def yCrCb2RGB(self, yCrCb):
        xForm = np.array([[255/219, 255/224*1.402, 0],
            [255/219, -255/224*1/402*0.299/0.587, -255/224*1.772*0.114/0.587], 
            [255/219, 0, 255/224*1.772]])
        rgb = np.array(yCrCb, dtype = np.float)
        rgb -= [16, 128, 128]
        rgb = rgb.dot(xForm.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return np.uint8(rgb)

    @property
    def yCrCb(self):
        if not hasattr(self, '_yCrCb'):
            self._yCrCb, self._alpha = [list(x) for x in zip(*[(p[:-1], p[-1]) for p in self.palette])]
        return self._yCrCb

    @property
    def alpha(self):
        if not hasattr(self, '_alpha'):
            self._yCrCb, self._alpha = [list(x) for x in zip(*[(p[:-1], p[-1]) for p in self.palette])]
        return self._alpha

    @property
    def rgb(self):
        if not hasattr(self, '_rgb'):
            self._rgb = self.yCrCb2RGB(self.yCrCb)
        return self._rgb

    @property
    def parent(self):
        return self._parent

class ObjectDefinitionSegment:

    def __init__(self, stream, parent):
        self._parent = parent
        self.objectID = stream.readUShort()
        self.version = stream.readUChar()
        self.sequence = stream.readByte()
        
        # Fisrt Fragment: has data Length, width and height fields
        if self.isFirst:
            self.dataLength = int.from_bytes(stream.readBytes(3), byteorder = 'big')
            self.width = stream.readUShort()
            self.height = stream.readUShort()
            # Data length includes width and height (2 bytes each, 4 bytes total)
            dataLength = len(self.parent) - 7
            self.imgData = stream.readBytes(dataLength - 4)
        # Consequent Fragments: no data Length, width or height field
        else:
            # Data Length uses 3 bytes, so we need to minus 4 not 7
            dataLength = len(self.parent) - 4
            self.dataLength = dataLength
            # No width or height, so we don't need to minus 4
            self.imgData = stream.readBytes(dataLength)
        
        # Single Fragment Correction
        if self.isFirst and self.isLast and dataLength != self.dataLength:
            print('Warning: [ODS] Length of image data asserted ({:d}) '
                    'does not match the amount found ({:d}). '
                    'The attribute will be reassigned'
                    .format(self.dataLength, dataLength))
            self.dataLength = dataLength

    @property
    def parent(self):
        return self._parent
    
    @property
    def isFirst(self):
        return bytes([self.sequence[0] & b'\x80'[0]]) == b'\x80'
    
    @property
    def isLast(self):
        return bytes([self.sequence[0] & b'\x40'[0]]) == b'\x40'

    @property
    def rle(self):
        if not hasattr(self, '_rle'):
            self._rle = self.RLEChunk(self.imgData, self)
        return self._rle

class EndSegment:
    
    def __init__(self, stream, parent):
        self._parent = parent

    @property
    def parent(self):
        return self._parent

class Segment:

    OPTION = {
        SEGMENT_TYPE.PCS: PresentationCompositionSegment,
        SEGMENT_TYPE.WDS: WindowDefinitionSegment,
        SEGMENT_TYPE.PDS: PaletteDefinitionSegment,
        SEGMENT_TYPE.ODS: ObjectDefinitionSegment,
        SEGMENT_TYPE.END: EndSegment
    }

    def __init__(self, stream):
        if stream.readWord() != b'PG':
            raise InvalidSegmentError
        self.pts = stream.readUInt()
        self.dts = stream.readUInt()
        self.type = SEGMENT_TYPE(stream.readByte())
        self.size = stream.readUShort()
        self.data = self.OPTION[self.type](stream, self)

    def __len__(self):
        return self.size

    @property
    def ptsms(self):
        return self.pts/90

    @property
    def dtsms(self):
        return self.dts/90

class DisplaySet:

    def __init__(self, segments):
        self.segments = segments

    @property
    def pcsSegment(self):
        if not hasattr(self, '_pcsSegment'):
            pcs = next((s for s in self.getType(SEGMENT_TYPE.PCS)), None)
            self._pcsSegment = pcs
            if pcs is not self.segments[0]:
                print('Warning: [Display Set] PCS is not the first segment')
        return self._pcsSegment

    @property
    def rle(self):
        if not hasattr(self, '_rle'):
            rle = []
            seed = b''
            prevID = -1
            for ods in self.getType(SEGMENT_TYPE.ODS):
                data = ods.data
                currID = data.objectID
                # Different object ID, so there're two different objects
                if currID != prevID and prevID != -1:
                    rle.append({'id': prevID, 'data': seed})
                    seed = b''
                # Same object ID, so we have to combine the image data together
                prevID = currID
                seed += data.imgData
            # One display set may have more than one objects, they have different object IDs
            # The number of objects is also indicated at PCS segment's number of composition objects field
            if prevID == -1:
                self._rle = []
            else:
                rle.append({'id': prevID, 'data': seed})
                self._rle = rle
        return self._rle

    @property
    def pix(self):
        if not hasattr(self, '_pix'):
            self._pix = [{'id': rle['id'], 'data': rleDecode(rle['data'])} for rle in self.rle]
        return self._pix
    
    @property
    def pds(self):
        if not hasattr(self, '_pds'):
            self._pds = next((p.data for p in self.getType(SEGMENT_TYPE.PDS)
                if p.data.paletteID == self.getType(SEGMENT_TYPE.PCS)[0].data.paletteID), None)
        return self._pds
    
    @property
    def image(self):
        if not hasattr(self, '_image'):
            self._image = [{'id': pix['id'], 'data': self.makeImage(pix['data'])} for pix in self.pix]
        return self._image

    @property
    def rgb(self):
        if not hasattr(self, '_rgb'):
            self._rgb = self.pds.rgb
        return self._rgb
    
    @property
    def alpha(self):
        if not hasattr(self, '_alpha'):
            self._alpha = self.pds.alpha
        return self._alpha
    
    @property
    def screenImage(self):
        if not hasattr(self, '_screenImage'):
            if self.pcsSegment.data.numberOfCompositionObjects > 0:
                transparentEntryPoint = next(i for i, a in enumerate(self.alpha) if a == 0)
                background = np.full((self.pcsSegment.data.height, self.pcsSegment.data.width), transparentEntryPoint, dtype = np.uint8)
                for obj in self.pcsSegment.data.compositionObjects:
                    pix = next(p['data'] for p in self.pix if p['id'] == obj.objectID)
                    height, width = pix.shape
                    if obj.cropped:
                        cropXPos = obj.cropXPos or 0
                        cropYPos = obj.cropYPos or 0
                        xStart = max(cropXPos, 0)
                        yStart = max(cropYPos, 0)
                        xEnd = min(cropXPos + obj.cropWidth, width)
                        yEnd = min(cropYPos + obj.cropHeight, height)
                        height = yEnd - yStart
                        width = xEnd - xStart
                        croppedPix = pix[yStart:yEnd, xStart:xEnd]
                    else:
                        croppedPix = pix
                    background[obj.yPos:(obj.yPos + height), obj.xPos:(obj.xPos + width)] = croppedPix
                self._screenImage = self.makeImage(background)
            else:
                self._screenImage = None
        return self._screenImage

    def makeImage(self, pixelLayer):
        alphaLayer = np.array([[self.alpha[x] for x in l] for l in pixelLayer], dtype = np.uint8)
        rgbPalette = self.rgb
        alphaImage = Image.fromarray(alphaLayer, mode='L')
        pixelImage = Image.fromarray(pixelLayer, mode='P')
        pixelImage.putpalette(rgbPalette)
        rgbaImage = pixelImage.convert('RGB')
        rgbaImage.putalpha(alphaImage)
        return rgbaImage

    def hasType(self, sType):
        return sType in self.segmentTypes
    
    def getType(self, sType):
        return [s for s in self.segments if s.type == sType]

class Epoch:

    def __init__(self, displaySets):
        self.displaySets = displaySets
        self.segments = list(itertools.chain(*[ds.segments for ds in displaySets]))

def rleDecode(rawData):
    lineBuilder = []
    pixels = []

    with io.BytesIO(rawData) as _:
        stream = BufferedRandomPlus(io.BufferedRandom(_))

        while stream.peekByte():
            first = stream.readUChar()
            if first:
                color = first
                length = 1
            else:
                second = stream.readUChar()
                if second == 0:
                    color = 0
                    length = 0
                    pixels.append(lineBuilder)
                    lineBuilder = []
                elif second < 64:
                    color = 0
                    length = second
                elif second < 128:
                    color = 0
                    length = ((second - 64) << 8) + stream.readUChar()
                elif second < 192:
                    color = stream.readUChar()
                    length = second - 128
                else:
                    length = ((second - 192) << 8) + stream.readUChar()
                    color = stream.readUChar()
            lineBuilder.extend([color] * length)
        
    if lineBuilder:
        print(f'Warning: [RLE] Hanging pixels without line ending: {lineBuilder}')
     
    return np.asarray(pixels, dtype=np.uint8)

if __name__ == '__main__':
    #pr = cProfile.Profile()
    #pr.enable()

    x = SupReader('Thor2.sup')
    #x = SupReader('[Nekomoe kissaten&VCB-Studio] Owarimonogatari S2 [01][Ma10p_1080p][x265_2flac].sc.sup')
    ss = x.segments
    dss = x.displaySets
    eps = x.epochs
    #pprint([[s.data.compositionState for s in ep.segments if s.type == SEGMENT_TYPE.PCS] for ep in eps])
    #pprint([[s.ptsms for s in ds.segments] for ds in dss])
    #pprint([[[(w.xPos, w.yPos) for w in s.data.windowObjects] for s in ep.segments if s.type == SEGMENT_TYPE.WDS] for ep in eps])
    #pprint([[s.data.paletteID for s in ep.segments if s.type == SEGMENT_TYPE.PDS] for ep in eps])
    #pprint([[(w.xPos, w.yPos) for w in s.data.windowObjects] for s in ss if s.type == SEGMENT_TYPE.WDS])
    #pprint([[(o.xPos, o.yPos) for o in s.data.compositionObjects] for s in ss if s.type == SEGMENT_TYPE.PCS and s.data.compositionState == COMPOSITION_STATE.EPOCH_START])
    for i, ds in enumerate(dss):
        if ds.screenImage:
            ds.screenImage.save(f'result/{i}.png')
    #pr.disable()
    #s = io.StringIO()
    #ps = pstats.Stats(pr, stream=s)
    #ps.print_stats()
    #print(s.getvalue())
