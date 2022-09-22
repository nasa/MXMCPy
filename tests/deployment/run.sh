#!/bin/sh

echo
echo THIS TEST SHOULD ONLY BE RUN AFTER A NEW DEPLOY TO PYPI.
echo

docker-compose build
docker-compose up

exit $?

