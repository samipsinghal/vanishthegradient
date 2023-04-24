# vanishthegradient
Tasked to coming up with a modified residual network (ResNet) architecture with the highest test accuracy on the CIFAR- 10 image classification dataset, under the constraint that your model has no more than 5 million parameters.



Code Structure : 


models.py	 = Resnet definitions and model configuartion 

resnet.py	= main python file

tools.py	= Debug functions

transformers.py	= Data Transform definitions



**Help Options -**

python resnet.py -h



**How to run**

**To start training with best model configuration for 50 epochs execute the following command  :-**
python resnet.py -e 50 -o adadelta -an -sc -mx -v -m project1_model



**To resume training for 50 epochs from best state checkpoint :-**
python resnet.py -e 50 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test


**To save logs to a file in logs directory  :-**
python resnet.py -e 50 -m project1_model -r AA4Test >> logs/<filename>.log

Replace <filename> with your choice of filename 



<img width="993" alt="image" src="https://user-images.githubusercontent.com/27934754/234092643-9b67f2de-501a-4bb0-9749-9e0a6dfaea03.png">

