from videocore import __version__
from distutils.core import setup

setup(name='py-videocore',
      version=__version__,
      description='Python library for GPGPU programming on Raspberry Pi',
      author='Koichi Nakamura',
      author_email='koichi@idein.jp',
      url='https://github.com/nineties/py-videocore',
      packages=['videocore']
      )
