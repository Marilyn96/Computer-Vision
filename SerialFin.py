# linksprite_grab.py
# grabs photo from linksprite LS-Y201 camera
#
# Jon Klein, 8/26/11
# kleinjt@ieee.org
# MIT License
#
# usage: python linksprite_grab picture.jpg

import serial, time, sys
import binascii


LK_RESET = "56002600"
LK_RESET_RE = "7600260000"
LK_PICTURE = "5600360100"
LK_PICTURE_RE = "7600360000"
LK_JPEGSIZE = "5600340100"
LK_JPEGSIZE_RE = "760034000400000000"
LK_STOP = "5600360103"
LK_STOP_RE = "7600360000"

LK_READPICTURE = "5600320C000A000000000000"
LK_PICTURE_TIME = "000A"  # .1 ms
LK_READPICTURE_RE = "7600320000"
JPEG_START = "FFD8"
JPEG_END = "FFD9"


def init_serial():
    return serial.Serial('COM8', '38400', timeout=1)


def main():
    s = init_serial()

    link_reset(s)
    take_picture(s)
    size = check_picturesize(s)
    picture = grab_picture(s, size)

    filename = 'UdderImage.jpg'
    if (len(sys.argv) > 1):
        filename = sys.argv[1]
    file = open(filename, 'wb')
    file.write(binascii.unhexlify(picture))
    file.close()
    s.close()


def grab_picture(s, size):
    s.flushInput()
    s.write(binascii.unhexlify(LK_READPICTURE + size + LK_PICTURE_TIME))
    re = binascii.hexlify(s.read(5)).decode('utf-8')

    if (re != LK_READPICTURE_RE):
        print
        'read picture response failed'
    else:
        print('read picture response successful')

    picture = binascii.hexlify(s.read(2)).decode('utf-8').upper()
    #print(picture)#.upper())
    #print(JPEG_START)
    if picture != JPEG_START:
        print('picture start incorrect')
    else:
        print("picture start correct")

    while (picture[-4:] != JPEG_END):
        re = binascii.hexlify(s.read(2)).decode('utf-8').upper()
        picture = picture + re
        #print(re)
    #print(picture)
    return picture

def link_reset(s):
    s.flushInput()
    s.write(binascii.unhexlify(LK_RESET))
    re = binascii.hexlify(s.read(5)).decode('utf-8')
    if (re != LK_RESET_RE):
        print("reset response failed")
    else:
        print("successfully reset")
    time.sleep(.5)


def take_picture(s):
    s.flushInput()
    s.write(binascii.unhexlify(LK_PICTURE))
    re = binascii.hexlify(s.read(5)).decode('utf-8')
    if (re != LK_PICTURE_RE):
        print
        "picture response failed"
    else:
        print("picture captured successfully")
    time.sleep(.1)


def check_picturesize(s):
    s.flushInput()
    s.write(binascii.unhexlify(LK_JPEGSIZE))
    re = binascii.hexlify(s.read(9)).decode('utf-8')
    return re[-4:]

if __name__ == "__main__":
    main()