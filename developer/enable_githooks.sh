#/bin/bash

GITDIR=$(git rev-parse --show-toplevel)
if [ -e ${GITDIR}/.git/hooks ]
then
  rm -rf ${GITDIR}/.git/hooks
fi

ln -s ${GITDIR}/.githooks ${GITDIR}/.git/hooks
