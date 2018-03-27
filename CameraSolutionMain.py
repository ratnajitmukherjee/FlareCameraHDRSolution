"""
Copyright © 2018, Ratnajit Mukherjee.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from joblib import Parallel, delayed
import multiprocessing
import argparse
import numpy as np
import cv2
import os
import time
import OpenEXR
import Imath


class FlareDebayerMerge:
    """
        --------------------------------------------------------------------
        ***** Flare camera output extraction, debayer and merge to HDR *****
        --------------------------------------------------------------------
    """
    def writeEXR(self, img, filename):
        """
        ****************************************************************************************
        Function to write EXR files.
        NOTE: We do not use OpenCV EXR format since that outputs uncompressed EXR images.
        Instead, we use the OpenEXR package provided EXRs
        :param img: input image ndarray (R,G,B) dtype = float32
        :param filename: absolute path of the output filename
        :return: <none>
        *****************************************************************************************
        """
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(filename, header)
        # this order is reversed because of the OpenCV's BGR format
        R = (img[:, :, 0]).astype(np.float16).tostring()
        G = (img[:, :, 1]).astype(np.float16).tostring()
        B = (img[:, :, 2]).astype(np.float16).tostring()
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()
        return

    def merge2HDR(self, input_path, output_path, exp_times, verbosity):
        """
        ***************************************************************
        Function to merge Debayered images into HDR images
        :param input_path: the debayered images path
        :param output_path: the output HDR path
        :param exp_times: the input exposure times in seconds
        :param verbosity: show individual file progress
        :return: <none>
        NOTE: Function uses Debevec's merging algorithm to merge
        exposures to HDR file.
        ***************************************************************
        """
        # global starting number of the HDR frames
        hdrnum = 0

        # list all files in the debayer folder
        filelist = [filename for dirpath, dirnames, filenames in os.walk(input_path) for filename in filenames
                    if filename.endswith('.jpg')]

        # check whether directory exists else create a directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # process debayer images to HDR images
        for i in range(0, len(filelist), len(exp_times)):
            exposures = []
            # read the exposures and append a list of exposures to be merged to HDR
            for j in range(i, i + 4):
                filename = os.path.join(input_path, filelist[j])
                ldr_image = cv2.imread(filename, cv2.IMREAD_COLOR)
                ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)
                exposures.append(ldr_image)

            # align the exposure list
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(exposures, exposures)

            # obtain camera response function
            calibrateDebevec = cv2.createCalibrateDebevec()
            responseDebevec = calibrateDebevec.process(exposures, exp_times)

            # create HDR from camera response
            mergeDebevec = cv2.createMergeDebevec()
            hdr = mergeDebevec.process(exposures, exp_times, responseDebevec)

            # set output file name and write exr (we use a separate exr because OpenCV EXR is not compressed)
            outfilename = os.path.join(output_path, '{0:05d}.exr'.format(hdrnum))
            self.writeEXR(hdr, outfilename)
            if verbosity == 1:
                print('HDR file: {0} merged..'.format(outfilename))
            hdrnum += 1
        return

    def debayerImages(self, input_path, filename, index, output_path, verbosity):
        """
        *******************************************************
        Function to demosaic flare camera images (pattern GBRG)
        :param input_path: bayer image directory
        :param filename: filename from the filelist in the main function
        :param index: number of file fetched (will be used for writing the images
        :param output_path: debayer image directory
        :param verbosity: show individual file progress
        :return:
        """
        bayer = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_UNCHANGED)
        debayer = cv2.cvtColor(bayer, cv2.COLOR_BAYER_GR2BGR)
        debayer = cv2.normalize(debayer, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        debayer = np.power(debayer, (1 / 2.2))
        final_img = cv2.normalize(debayer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        outfilename = '{0:05d}'.format(index) + '.jpg'
        cv2.imwrite(filename=os.path.join(output_path, outfilename), img=final_img)
        if verbosity == 1:
            print("Filename: {0} written...".format(outfilename))
        return


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Program to debayer and merge multiple exposures from the Flare camera")
    parser.add_argument("-b", "--bayer_dir", help="Directory where bayer (RAW) images are located", type=str, required=True)
    parser.add_argument("-d", "--debayer_dir", help="Directory where debayer images are located", type=str, required=True)
    parser.add_argument("-hdr", "--hdr_dir", help=" HDR directory", type=str, required=True)
    parser.add_argument("-exp", "--exp_times", nargs='+', help="Exposure times in milli-seconds (Space delimited)", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    # Get all arguments
    bayer_path = args.bayer_dir
    debayer_path = args.debayer_dir
    hdr_path = args.hdr_dir
    exp_times = np.round(np.asarray(args.exp_times, dtype=np.float), decimals=2)

    # get optional arguments (verbosity)
    if args.verbose:
        verbosity = 1
    elif args.quiet:
        verbosity = 0
    else:
        verbosity = 1  # default behaviour

    filelist = [filename for dirpath, dirnames, filenames in os.walk(bayer_path)
                for filename in filenames if filename.endswith('.bmp')]

    # calling the debayer merge class
    flr = FlareDebayerMerge()

    # calculating the number of CPU cores which can be used
    nCPU = multiprocessing.cpu_count()

    # iterating through the filelist (timed execution)
    start_time = time.time()
    print("Debayer Operation started")
    Parallel(n_jobs=nCPU)(delayed(flr.debayerImages)(bayer_path, filelist[index], index, debayer_path, verbosity)
                          for index in range(0, len(filelist)))

    print("Debayer operation completed. \n "
          "Total Execution time: {0}".format(time.time() - start_time))

    mtime = time.time()
    flr.merge2HDR(debayer_path, hdr_path, exp_times, verbosity)
    print("Merge operation completed. \n "
          "Total execution time: {0}".format(time.time()-mtime))