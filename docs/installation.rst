============
Installation
============

.. highlight:: bash

At the command line::

    $ pip install --user kyos

Update your .bashrc file with the path to user-installed python packages::

    export PATH=~/.local/bin:$PATH

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv kyos
    $ pip install kyos


Upgrading Kyos
-----------------------------------------

If you previously installed with pip, you can upgrade to the newest version from the command line::

    $ pip install --user --upgrade kyos


Uninstalling Kyos
--------------------------------------------

If you installed with pip, you can uninstall from the command line::

    $ pip uninstall kyos
