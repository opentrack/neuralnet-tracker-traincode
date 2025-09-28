from typing import Sequence
import numpy as np
import h5py
import argparse
from contextlib import ExitStack


def copy_attributes(src: h5py.HLObject, dst: h5py.HLObject):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _batched_copy(dst, src, dst_offset=0):
    bs = 1024
    n = min(dst.shape[0] + dst_offset, src.shape[0])
    for a in range(0, n, bs):
        b = min(n, a + bs)
        dst[a + dst_offset : b + dst_offset, ...] = src[
            a:b, ...
        ]  # Buffer in memory, then write. This works ...


def concatenating_join(name1: str, items: Sequence[h5py.Dataset], fout: h5py.File):
    first = next(iter(items))
    sizes = [ds.shape[0] for ds in items]
    N = sum(sizes)
    print(f"Copying {name1}: {sizes} items of type {first.dtype}")

    dst = fout.create_dataset_like(
        name1, first, shape=(N, *first.shape[1:]), maxshape=(N,) + first.shape[1:]
    )
    assert all(list(first.attrs.items()) == list(ds.attrs.items()) for ds in items)
    copy_attributes(first, dst)
    try:
        offset = 0
        for src, count in zip(items, sizes):
            dst[offset : offset + count, ...] = src
            offset += count
    except TypeError:  # Variable length elements
        # Buffer in memory needs lots of memory. So copy the data in bachtes.
        offset = 0
        for src, count in zip(items, sizes):
            _batched_copy(dst, src, dst_offset=offset)
            offset += count


def join_sequence_starts(name1: str, items: Sequence[h5py.Dataset], fout: h5py.File):
    first = next(iter(items))
    starts = [first[:1]]
    for ds in items:
        current = starts[-1][-1]
        starts.append(ds[...][1:] + current)
    starts = np.concatenate(starts)
    print(
        f"Joining sequence_starts `{name1}`: {[ds.shape[0] for ds in items]} sequences. New number of samples: {starts[-1]}"
    )
    fout.create_dataset(name1, data=starts)


def dsjoin(grps: list[h5py.Group], fout_parent: h5py.Group):
    first = next(iter(grps))
    assert all(g.keys() == first.keys() for g in grps)
    for name in first.keys():
        items = [g[name] for g in grps]
        if isinstance(next(iter(items)), h5py.Dataset):
            assert all(isinstance(i, h5py.Dataset) for i in items)
            if name != "sequence_starts":
                concatenating_join(name, items, fout_parent)
            else:
                join_sequence_starts(name, items, fout_parent)
        else:
            assert all(isinstance(i, h5py.Group) for i in items)
            dsjoin(items, fout_parent.create_group(name))


def test_dsjoin():
    seqs1 = [[1, 2, 3], [4, 5]]
    seqs2 = [[6, 7], [8]]
    with h5py.File("/tmp/a.h5", "w") as f1, h5py.File("/tmp/b.h5", "w") as f2:
        f1.create_dataset("x", data=np.concatenate(seqs1))
        f2.create_dataset("x", data=np.concatenate(seqs2))
        f1.create_dataset("sequence_starts", data=[0, 3, 5])
        f2.create_dataset("sequence_starts", data=[0, 2, 3])
    with h5py.File("/tmp/a.h5", "r") as f1, h5py.File("/tmp/b.h5", "r") as f2:
        with h5py.File("/tmp/c.h5", "w") as fout:
            dsjoin([f1, f2], fout)
    with h5py.File("/tmp/c.h5", "r") as fout:
        assert np.all(fout["x"][...] == np.array([1, 2, 3, 4, 5, 6, 7, 8]))
        assert np.all(fout["sequence_starts"][...] == np.array([0, 3, 5, 7, 8]))


if __name__ == "__main__":
    # TODO: testfile
    # test_dsjoin()

    parser = argparse.ArgumentParser(description="Join datasets")
    parser.add_argument("destination", help="destination file")
    parser.add_argument("sources", help="source files", type=str, nargs="*")
    args = parser.parse_args()
    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn)) for fn in args.sources]
        with h5py.File(args.destination, "w") as fout:
            dsjoin(files, fout)
