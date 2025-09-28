import argparse
from os.path import join, dirname, basename, splitext, sep


from dsprocess_300wlp import ReadSample, HdfDatasetWriter


def is_sample_file(fn):
    return splitext(fn)[1] == '.mat' and not fn.endswith(sep) and dirname(fn) == 'AFLW2000'


def discover_samples(zf):
    filenames = [f.filename for f in zf.filelist if is_sample_file(f.filename)]
    return filenames


class HdfWriter300WLPWithoutRotations(HdfDatasetWriter):
    def get_file_groups(self, zf):
        return sorted(discover_samples(zf))

    def make_sample_reader(self) -> ReadSample:
        return ReadSample(full_face_bounding_box=True, load_pt3d_68=True, load_pt2d_68=False, load_roi=False, load_face_params=True)


def generate_hdf5_dataset(source_file, outfilename, count=None):
    HdfWriter300WLPWithoutRotations().generate_hdf5_dataset(source_file, outfilename, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest='count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else splitext(args.source)[0] + '.h5'
    generate_hdf5_dataset(args.source, dst, args.count)
