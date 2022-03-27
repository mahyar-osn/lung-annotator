SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
echo $SCRIPTPATH

g++ -std=c++11 $SCRIPTPATH/../utils/render.cpp -o $SCRIPTPATH/../utils/render.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
