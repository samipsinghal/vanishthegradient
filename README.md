# vanishthegradient
Tasked to coming up with a modified residual network (ResNet) architecture with the highest test accuracy on the CIFAR- 10 image classification dataset, under the constraint that your model has no more than 5 million parameters.



Code Structure : 


models.py	 = Resnet definitions and model configuartion 

resnet.py	= main python file

tools.py	= Debug functions

transformers.py	= Data Transform definitions


**How to run**

**To start training with best model configuration for 100 epochs execute the following command  :-**
python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model




**To resume training for 100 epochs from best state checkpoint :-**
python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test


**To save logs to a file in logs directory  :-**
python resnet.py -e 100 -m project1_model -r AA4Test >> logs/<filename>.log

Replace <filename> with your choice of filename
