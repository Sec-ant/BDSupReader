#include <stdio.h>
#include <stdlib.h>
#include "RunLength.h"

unsigned int RLEncode(unsigned char * pixels, unsigned short width, unsigned short height, unsigned char * RLData){
    unsigned char curr;
    unsigned short i, j;
    unsigned int repeat = 0;
    unsigned int offset = 4;
    RLData[0] = (unsigned char) ((width & 0xFF00) >> 8);
    RLData[1] = (unsigned char) (width & 0x00FF);
    RLData[2] = (unsigned char) ((height & 0xFF00) >> 8);
    RLData[3] = (unsigned char) (height & 0x00FF);
    for (j = 0; j < height; ++j) {
        for (i = 0; i < width; ++i) {
            curr = pixels[j * width + i];
            ++repeat;
            if ((i < width - 1 && pixels[j * width + i + 1] != curr) || i == width - 1 || repeat == 16383) {
                if (repeat == 1 && curr != 0) {
                    RLData[offset] = curr;
                    ++offset;
                }
                else if (repeat == 2 && curr != 0){
                    RLData[offset] = curr;
                    RLData[offset + 1] = curr;
                    offset += 2;
                }
                else if (repeat < 64) {
                    switch (curr) {
                        case 0:
                            RLData[offset] = 0;
                            RLData[offset + 1] = repeat;
                            offset += 2;
                            break;
                        default:
                            RLData[offset] = 0;
                            RLData[offset + 1] = repeat + 128;
                            RLData[offset + 2] = curr;
                            offset += 3;
                            break;
                    }
                }
                else {
                    switch (curr) {
                        case 0:
                            RLData[offset] = 0;
                            RLData[offset + 1] = (repeat >> 8) + 64;
                            RLData[offset + 2] = repeat & 0xFF;
                            offset += 3;
                            break;
                        default:
                            RLData[offset] = 0;
                            RLData[offset + 1] = (repeat >> 8) + 192;
                            RLData[offset + 2] = repeat & 0xFF;
                            RLData[offset + 3] = curr;
                            offset += 4;
                            break;
                    }
                }
                repeat = 0;
            }
        }
        RLData[offset] = 0;
        RLData[offset + 1] = 0;
        offset += 2;
    }
    return offset;
}

void RLDecode(unsigned char * RLData, unsigned int length, unsigned char * pixels){
    unsigned int offset = 4;
    unsigned char entry, first, second;
    unsigned int repeat;
    unsigned int i = 0, j = 0, k;
    unsigned short width = (((unsigned short)RLData[0]) << 8) + RLData[1];
    unsigned short height = (((unsigned short)RLData[2]) << 8) + RLData[3];
    while (offset < length) {
        first = RLData[offset];
        if (first > 0) {
            entry = first;
            repeat = 1;
            ++offset;
        }
        else{
            second = RLData[offset + 1];
            if (second == 0) {
                entry = 0;
                repeat = 0;
                ++i;
                if (i == height) {
                    break;
                }
                offset += 2;
            }
            else if (second < 64) {
                entry = 0;
                repeat = second;
                offset += 2;
            }
            else if (second < 128) {
                entry = 0;
                repeat = ((second - 64) << 8) + RLData[offset + 2];
                offset += 3;
            }
            else if (second < 192) {
                entry = RLData[offset + 2];
                repeat = second - 128;
                offset += 3;
            }
            else{
                entry = RLData[offset + 3];
                repeat = ((second - 192) << 8) + RLData[offset + 2];
                offset += 4;
            }
        }
        for (k = j; k < j + repeat; ++k){
            pixels[k] = entry;
        }
        j = k;
    }
    if (i < height) {
        fprintf(stderr, "Warning: [RLDecode] Hanging pixels without line ending (expected: %d, actual: %d)\n", height, i);
    }
}
