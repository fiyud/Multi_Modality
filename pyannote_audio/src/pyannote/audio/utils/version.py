# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Text

from packaging.version import Version


def check_version(library: Text, theirs: Text, mine: Text, what: Text = "Pipeline"):

    their_version = Version(theirs)
    my_version = Version(mine)

    if their_version.major > my_version.major:
        print(
            f"{what} was trained with {library} {theirs}, yours is {mine}. "
            f"Bad things will probably happen unless you upgrade {library} to {their_version.major}.x."
        )

    elif their_version.major < my_version.major:
        print(
            f"{what} was trained with {library} {theirs}, yours is {mine}. "
            f"Bad things might happen unless you revert {library} to {their_version.major}.x."
        )

    elif their_version.minor > my_version.minor:
        print(
            f"{what} was trained with {library} {theirs}, yours is {mine}. "
            f"This should be OK but you might want to upgrade {library}."
        )
