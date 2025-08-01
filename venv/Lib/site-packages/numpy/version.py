
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.2.4"
__version__ = version
full_version = version

git_revision = "3b377854e8b1a55f15bda6f1166fe9954828231b"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
