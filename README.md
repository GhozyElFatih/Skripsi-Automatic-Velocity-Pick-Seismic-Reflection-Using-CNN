# Skripsi - Automatic Velocity Pick Seismic Reflection Using CNN
This repository serves as an attachment to my bachelor thesis.
Main idea is to detect peak of semblance from image and extract the center of coordinates.

![gambar001](https://user-images.githubusercontent.com/85453675/124712513-987e4f80-df29-11eb-95ef-a26ba66b9bcf.png)

Next, the center of coordinates from semblance target will be normalized to x scale as velocity (range 1500 to 4000 m/s) and y scale as two-way-traveltime (range 0 to 3 s).
Then, another semblance from each CDP are treated same. Velocity we got from each semblance will be stored and used for NMO correction, stacking, and migration to seismic data.

![semb](https://user-images.githubusercontent.com/85453675/126952002-0ed8262c-d48b-44cc-8fef-fe43e52ad996.png)

The training is conducted by using the TensorFlow Object Detection API. The repository for the module can be downloaded via the following page.

https://github.com/tensorflow/models

Pre-trained model or CNN architecture for this thesis is using RCNN Inception ResNet V2 that can be downloaded via link below.

http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz

For labelling purpose, I use LabelImg from Tzutalin.

https://github.com/tzutalin/labelImg

Last but not least, my highest appreciation goes to Evan Juras for his tutorial and permission to use his repositor for the purposes of this thesis. Thank you very much!

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

The work on this thesis is carried out on a computer with the following specifications:
- Processor : Intel® Core™ i7-6700 3.40 GHz
- GPU : NVIDIA GeForce GT 730 VRAM 2 GB
- RAM : 16 GB DDR 4

For software and modules version, I used:
- Python 3.6
- CUDA 10
- cuDNN 7.4
- TensorFlow-GPU 1.15.0
- LVIS 0.5.3
- Pillow 8.2.0
- Matplotlib 3.2
- Pycocotools 2.02
- OpenCV Python 4.5.2.52
- LXml 4.6.3
- ContextLib2 0.6.0
- Pywin32 301
- SciPy 1.6.3
- LabelImg 1.8.5

Any question or discussion for this research, you can hit me up through my email fatih.el.ghozy@gmail.com. For academic usage of this repository, kindly please cite my thesis.
- Fatih, Ghozy El. (2021). _Pemilihan Kecepatan Otomatis Seismik Refleksi Menggunakan Convolutional Neural Network_. Skripsi. Depok: Universitas Indonesia. \
(Bahasa Indonesia)

In english or other language and other format, just simply replace 'Skripsi' to language used, and make sure every citation information are there.


>_Best regards,_ \
>_Ghozy El Fatih_ \
>_Geophysics, Faculty of Mathematics and Natural Sciences, University of Indonesia_
