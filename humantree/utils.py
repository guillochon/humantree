import os
import tempfile
from contextlib import contextmanager

# Below from
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python


@contextmanager
def temp_atomic(suffix='', dir=None):
    """Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    dir : string
        optional directory to save temporary file in

    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            os.remove(tf.name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """Open temporary file object that atomically moves upon exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    *args : mixed
        Any valid arguments for :code:pen    **kwargs : mixed
        Any valid keyword arguments for :code:pen
    """
    fsync = kwargs.get('fsync', False)

    with temp_atomic(
            dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, *args, **kwargs) as file:
            try:
                yield file
            finally:
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())
        if os.path.isfile(filepath):
            os.remove(filepath)
        os.rename(tmppath, filepath)
