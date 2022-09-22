#!/bin/sh

if [ "$1" = "release" ] ; then
    echo "DEPLOYING TO RELEASE."
else
    echo "Deploying to test."
fi

# Clean up an previous build remnants.
rm -rf dist build

# Deploy to PyPi.
python3 setup.py sdist bdist_wheel

# shellcheck disable=SC2181
if [ $? != 0 ] ; then
    echo
    echo "Build failed. Aborting"
    exit 1
else
    echo
    echo "Build successful. Uploading to PyPi..."
    echo
fi

echo
echo
if [ "$1" = "release" ] ; then
    echo "Uploading to release..."
    python3 -m twine upload dist/*
else
    echo
    echo "Uploading to test. Use __token__ for username."
    echo
    #python3 -m twine upload --repository testpypi dist/*
fi

# shellcheck disable=SC2181
if [ $? != 0 ] ; then
    echo
    echo "PyPi deployment failed. Aborting."
    exit 1
else
    echo
    echo "PyPi deploy successful. Have a nice day."
    echo
fi

# Clean up.
rm -rf dist build
