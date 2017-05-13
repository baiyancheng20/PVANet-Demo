#This part is used for compile faster_rcnn_demo.cpp
#这里是版本要求，根据自己项目而定，我用了默认的
cmake_minimum_required (VERSION 2.8)
#我们的工程的名字
project (pva_demo)

#添加我们要生成的可执行文件的名字，以及相应的源码文件
add_executable(pva_demo pva_demo.cpp)
#这里添加这个faster_rcnn_demo.cpp所依赖的头文件路径
#首先是Caffe目录的include
#其次是用了gpu_nms.cu，所以也要添加相应的头文件gpu_nms.hpp，在py-faster-rcnn根目录下的lib/nms中
#下面就是几个Caffe的依赖项，包括Python
#值得注意的是boost/python。hpp的头文件路径也要加入
#还有opencv的路径，cuda路径，线性代数库路径相应的都要添加
include_directories ( "/home/mordekaiser/pva-faster-rcnn/caffe-fast-rcnn/include"
    "/home/mordekaiser/pva-faster-rcnn/lib/nms" 
    "/home/mordekaiser/pva-faster-rcnn/lib"
    /usr/include
    /usr/local/include 
    /usr/include/python2.7
    /usr/local/cuda/include )
#在这里值得说一下，target_link_libraries 语句中 生成的目标是可执行文件 后面紧跟的得是动态库的完整路径，否则会出错
#我一开始用的是Link_directoreis然后在后面直接加入了动态库的路径，结果他一直报错，提示找不到库，ORZ真是跪了，
#所以要么在这里直接加入完整路径或者同通过另一条语句find_library()，这种方式也比较好，直接去指定路径查找，返回相应的绝对路径也可避免直接添加地址的问题
#gpu_nms.so 在py-faster-rcnn根目录下的lib\nms中，直接make就会生成这个so文件
target_link_libraries(pva_demo 
    /home/mordekaiser/pva-faster-rcnn/caffe-fast-rcnn/build/lib/libcaffe.so
    /home/mordekaiser/pva-faster-rcnn/lib/nms/libgpu_nms.so 
    /usr/local/lib/libopencv_highgui.so 
    /usr/local/lib/libopencv_core.so 
    /usr/local/lib/libopencv_imgproc.so 
    /usr/local/lib/libopencv_imgcodecs.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    /usr/lib/x86_64-linux-gnu/libboost_python.so
    /usr/lib/x86_64-linux-gnu/libpython2.7.so
    )
