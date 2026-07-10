#!/usr/bin/env python3
# Recover the 256x256 interlacing filter design grids from BBC WHP 315
# ("Optimal Interlacing using Human Sensitivity Measurements", 2016).
# Figures 5 and 7 are embedded as lossless bitmaps at design resolution:
#   fig 5: binary pass/stop template (region-grown from WHP 230 data)
#   fig 7: specification bands, grey levels 255/190/125/60/0 =
#          pass / transition / -12 dB / -40 dB / -50 dB
# Requires poppler's pdfimages. The PDF is not distributable; supply your own.
#
# usage: whp315_extract.py WHP315.pdf outdir/

import glob
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__ or 'usage: whp315_extract.py WHP315.pdf outdir/')
    pdf, outdir = sys.argv[1], sys.argv[2]

    grids = []
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(['pdfimages', '-png', pdf, td + '/im'], check=True)
        for f in sorted(glob.glob(td + '/im-*.png')):
            im = Image.open(f).convert('L')
            if im.size == (256, 256):
                grids.append(np.asarray(im))

    template = next(g for g in grids if np.array_equal(np.unique(g), [0, 255]))
    bands = next(g for g in grids if len(np.unique(g)) == 5)

    assert int((template == 255).sum()) == 256 * 256 // 2
    assert np.all((template == 255) == (template[::-1, ::-1] == 0))
    assert np.array_equal(np.unique(bands), [0, 60, 125, 190, 255])
    assert np.all((bands == 255) <= (template == 255))

    Image.fromarray(template).save(outdir + '/template.png')
    Image.fromarray(bands).save(outdir + '/bands.png')
    print('wrote', outdir + '/template.png', outdir + '/bands.png')


if __name__ == '__main__':
    main()
