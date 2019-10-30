#/usr/bin/env bash
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

GITDIR=$(git rev-parse --show-toplevel)
if [ -e ${GITDIR}/.git/hooks ]
then
  rm -rf ${GITDIR}/.git/hooks
fi

ln -s ${GITDIR}/.githooks ${GITDIR}/.git/hooks
