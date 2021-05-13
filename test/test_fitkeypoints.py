import scripts.fitkeypoints
from os.path import join, dirname

def test_fitkeypoints():
    class Args(object):
        filename = join(dirname(__file__),'..','aflw2kmini.h5')
        force = True
    scripts.fitkeypoints.fitall(Args)