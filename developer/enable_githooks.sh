#/bin/bash

if [ -e .git/hooks ]
then
  rm -rf .git/hooks
fi

ln -s ../.githooks .git/hooks
