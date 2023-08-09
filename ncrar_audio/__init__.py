from .version import __version__

import os


def disable_quick_edit():
    # From https://stackoverflow.com/questions/73486528/python-script-pausing-in-cmd
    import win32console as con
    import signal

    # Missing constants in pywin
    ENABLE_EXTENDED_FLAGS = 0x0080
    ENABLE_QUICK_EDIT_MODE = 0x0040

    # Modify console mode to disable quick edit mode
    h = con.GetStdHandle(con.STD_INPUT_HANDLE)
    oldMode = h.GetConsoleMode()
    h.SetConsoleMode((oldMode | ENABLE_EXTENDED_FLAGS) &
            ~ENABLE_QUICK_EDIT_MODE)


# This ensures that running scripts from the command line does not accidentally
# pause the script (such as for tqdm callbacks).
if os.name == 'nt':
    try:
        disable_quick_edit()
    except:
        pass
