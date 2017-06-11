#-------------------------------------------------------------------------------
# Kaggle Intel Cervix Cancer Challenge
#
#
# Image loading and basic pre-processing with EBImage
# Color images
# Submission to Kaggle is generated
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#Load librarys
#-------------------------------------------------------------------------------


library(dplyr)
library(EBImage)
library(mxnet)
library(nnet)


#-------------------------------------------------------------------------------
# Load images using EBImage
#
# This loop resize all images, we dont have to do data-augmentation 
# because, it is done with the script data-augmentation.R. 
#-------------------------------------------------------------------------------


paths <- c("/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_1",
           "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_2",
           "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_3")


# Uncomment this if you want to load all training + extra images


#-------------------------------------------------------------------------------
#paths <- c("/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_1",
#           "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_2",
#           "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/all_data_resized/Type_3",
#            "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/train-extra-unidas/Type_1",
#            "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/train-extra-unidas/Type_2",
#            "/Users/joseadiazg/Desktop/Temporales UGR/SIGE/train-extra-unidas/Type_3",
#            )
#-------------------------------------------------------------------------------

in_type_counter <- 1 

for (t in  paths){
  
  patients <- dir(t)
  
  #For this simple example We rescale photos to 128*128 (= 16384)

  
  n_columnas <- 1 + 1 + 49152
  
  
  ordered_images <- data.frame(matrix(nrow = length(patients), ncol = n_columnas))
  colnames(ordered_images) <- c("paciente", "Type", paste("R", (1:16384), sep = ""), paste("G", (1:16384), sep = ""), paste("B", (1:16384), sep = ""))
  
  contador <- 1
  
  if(in_type_counter == 1){
    ordered_images$Type <- 1 #(*) ojo , asignación manual segun el folder que esté procesando
  }
  
  if(in_type_counter == 2){
    ordered_images$Type <- 2 #(*) ojo , asignación manual segun el folder que esté procesando
  }
  
  if(in_type_counter == 3){
    ordered_images$Type <- 3 #(*) ojo , asignación manual segun el folder que esté procesando
  }
  
  #+++
  mom_inicio <- Sys.time()
  print("beginning calculation: ")
  print(mom_inicio)
  #+++
  
  for (p in patients){
    
    cat("contador:", contador, " paciente:", p, "\n")
    
    ordered_images$paciente[contador] <- p
    
    imagen_paciente <- readImage(paste(t, p, sep = "/"))   #abre imagen de cada paciente...
    
    #TODO: We have to change this resize to a new version in witch we use the same
    # proportion of tam but smaller. 
    
    imagen_paciente <- resize(imagen_paciente, w = 128, h = 128)
    
    
    ordered_images[contador,  c(3:16386)] <- imagen_paciente[, , 1]
    ordered_images[contador,  c(16387:32770)] <- imagen_paciente[, , 2]
    ordered_images[contador,  c(32771:49154)] <- imagen_paciente[, , 3]
    
    contador <- contador + 1 
    
  }
  
  
  #+++
  print("calculation time: ")
  print(Sys.time() - mom_inicio)
  #+++
  
  
  if(in_type_counter == 1){
    ordered_images_Type_1 <- ordered_images
  }
  
  if(in_type_counter == 2){
    ordered_images_Type_2 <- ordered_images
  }
  
  if(in_type_counter == 3){
    ordered_images_Type_3 <- ordered_images
  }
  
  
  in_type_counter <-  in_type_counter + 1 
}


ordered_images_all <- bind_rows(ordered_images_Type_1, ordered_images_Type_2, ordered_images_Type_3)
  

#-------------------------------------------------------------------------------
# Subset for validation
#-------------------------------------------------------------------------------

table(ordered_images_all$Type)


ordered_images_all$Type <- ordered_images_all$Type - 1  


set.seed(9)
set_validacion <- ordered_images_all %>% 
  group_by(Type) %>% 
  sample_n(25) %>% 
  ungroup()


#-------------------------------------------------------------------------------
# Undersampling
#-------------------------------------------------------------------------------


set_train_unbalanced <- ordered_images_all[ordered_images_all$paciente %in% setdiff(ordered_images_all$paciente, set_validacion$paciente), ]

#primera muestra train undersampled:
set.seed(9)
undersample_1_set_train <- set_train_unbalanced %>% 
  group_by(Type) %>% 
  #sample_n(225) %>% 
  ungroup() %>% 
  sample_frac(1) %>%  
  sample_frac(1) #WE SHUFFLE TWICE

# dim(undersample_1_set_train)
# #[1]   675 12290
# dim(set_validacion)
# #[1]    75 12288


#_____train and  validación labels___________________________
target_undersample_1 <- undersample_1_set_train$Type
label_set_validacion <- set_validacion$Type
#____________________________________________________________


#_____train set1(undersampled) y and validation set, both balanced_______
undersample_1_set_train <-undersample_1_set_train[, c(3:49154)]
set_validacion <-set_validacion[, c(3:49154)]
#_____target y label de validación_________________________________________

undersample_1_set_train <- t(undersample_1_set_train)
dim(undersample_1_set_train) <- c(128, 128, 3, 675)


set_validacion <- t(set_validacion)
dim(set_validacion) <- c(128, 128, 3, 75)




#-------------------------------------------------------------------------------
# Tunning parametters for the nn
#-------------------------------------------------------------------------------


n_output <- 3
num_filters_conv2 = 14
num.round = 10
learning.rate = 0.1	
momentum = 0.0056
weight_decay = 0.0046	
initializer = 0.0667	


# AND SOME HELPER FUNCTIONS:
mLogLoss.normalize = function(p, min_eta=1e-15, max_eta = 1.0){
  #min_eta
  for(ix in 1:dim(p)[2]) {
    p[,ix] = ifelse(p[,ix]<=min_eta,min_eta,p[,ix]);
    p[,ix] = ifelse(p[,ix]>=max_eta,max_eta,p[,ix]);
  }
  #normalize
  for(ix in 1:dim(p)[1]) {
    p[ix,] = p[ix,] / sum(p[ix,]);
  }
  return(p);
}

# helper function
#calculates logloss
mlogloss = function(y, p, min_eta=1e-15,max_eta = 1.0){
  class_loss = c(dim(p)[2]);
  loss = 0;
  p = mLogLoss.normalize(p,min_eta, max_eta);
  for(ix in 1:dim(y)[2]) {
    p[,ix] = ifelse(p[,ix]>1,1,p[,ix]);
    class_loss[ix] = sum(y[,ix]*log(p[,ix]));
    loss = loss + class_loss[ix];
  }
  #return loss
  return (list("loss"=-1*loss/dim(p)[1],"class_loss"=class_loss));
}

# mxnet specific logloss metric
mx.metric.mlogloss <- mx.metric.custom("mlogloss", function(label, pred){
  p = t(pred);
  m = mlogloss(class.ind(label),p);
  gc();
  return(m$loss);
})
  


#-------------------------------------------------------------------------------
# Train the nn
#-------------------------------------------------------------------------------



train.x <- undersample_1_set_train
train.y <-target_undersample_1


#____________
data = mx.symbol.Variable('data')
#FIRST CONVOLUITIONAL LAYER + POOLING
conv1 = mx.symbol.Convolution(data=data, kernel=c(3, 3), num_filter = 3) 
relu1 = mx.symbol.Activation(data=conv1, act_type="relu") 
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=c(2,2), stride=c(2,2))


#SECOND CONVOLUTIONAL LAYER + POOLING
conv2 = mx.symbol.Convolution(data=pool1, kernel=c(3,3), num_filter = num_filters_conv2) 
relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",kernel=c(2,2), stride=c(2,2)) 

#FLATTEN THE OUTPUT
flatten = mx.symbol.Flatten(data=pool2) 

#FEED FULLY CONNECTED LAYER, NUMBER OF HIDDEN NODES JUST GEOMETRIC MEAN OF INPUT(14.5 * 14.5  * 13 =  2733.25) AND OUTPUT (3), sqrt(2733.25*3) =  91
input_previo_a_filtroconv2 <- 14.5*14.5
n_input <- input_previo_a_filtroconv2 * num_filters_conv2
num_hidden_fc1 <- round(sqrt(n_input*n_output)) 

fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=84) 
relu4 = mx.symbol.Activation(data=fc1, act_type="relu") 

#____________


fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=3) #ESTA PARA CLASIFICACION

mi_softmax  = mx.symbol.SoftmaxOutput(data=fc2)


devices <- mx.cpu()

mx.set.seed(0)

#___
tic <- proc.time()
#___
model <- mx.model.FeedForward.create( mi_softmax #for clasification
                                      , X=train.x
                                      , y=train.y
                                      , eval.data  = list("data" = set_validacion,"label" = label_set_validacion) 
                                      , ctx=devices
                                      , num.round=num.round
                                      , array.batch.size = 75 
                                      , learning.rate = learning.rate
                                      , momentum = momentum
                                      , wd=weight_decay
                                      , eval.metric = mx.metric.mlogloss 
                                      , initializer=mx.init.uniform(initializer) 
                                      #, epoch.end.callback = mx.callback.save.checkpoint("modelo_guardado_ccs") #(TO SAVE MODEL AT EVERY ITERATION)
                                      , batch.end.callback = mx.callback.log.train.metric(10)#, log)  
                                      , array.layout="columnmajor"
) 

#___
print(proc.time() - tic)
#___


#-------------------------------------------------------------------------------
# Validation
#-------------------------------------------------------------------------------

  
  preds <- predict(model, set_validacion,
                   ctx = NULL,
                   array.layout = "auto")
  
  #WE CAN INSPECT OUR LIMITED VALIDATION SET, PROBABILITIES:
  predicciones <- t(preds)
  predicciones <- cbind(label_set_validacion, predicciones)
  head(predicciones)
  