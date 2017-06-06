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
  
  #For this simple example We rescale photos to 256*256 (= 65536)
  
  n_columnas <- 1 + 1 + 196608
  
  
  ordered_images <- data.frame(matrix(nrow = length(patients), ncol = n_columnas))
  colnames(ordered_images) <- c("paciente", "Type", paste("R", (1:65536), sep = ""), paste("G", (1:65536), sep = ""), paste("B", (1:65536), sep = ""))
  
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
    
    imagen_paciente <- resize(imagen_paciente, w = 256, h = 256)
    
    
    ordered_images[contador,  c(3:65538)] <- imagen_paciente[, , 1]
    ordered_images[contador,  c(65539:131074)] <- imagen_paciente[, , 2]
    ordered_images[contador,  c(131075:196610)] <- imagen_paciente[, , 3]
    
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
  
  


  