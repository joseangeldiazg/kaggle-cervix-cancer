#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Set run parameters parameters
train_img_1_path <- "./Type_1"
train_img_2_path <- "./Type_2"
train_img_3_path <- "./Type_3"


#-------------------------------------------------------------------------------
# Load and pre-process train images
#-------------------------------------------------------------------------------

# Load EBImage library
library(EBImage)

# Load images into a dataframe
library(gsubfn)
img_file_1_list <- list.files(path = train_img_1_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
img_file_2_list <- list.files(path = train_img_2_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
img_file_3_list <- list.files(path = train_img_3_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)


#train_df <- data.frame()

#Hacemos 3 bucles porque tienen tama?os distintos cada clase.




Imagenes<-function(image,type) {
  img_file_name <- image
  
  img <- readImage(img_file_name)
  
  if(type=="flip"){
    img <- flip(img)
  }
  
  if(type=="flop"){
    img <- flop(img)
  }
  
  if(type=="rot"){
    img = rotate(img, 90, bg.col = "white")
  }
  if(type=="trans"){
    img_t = transpose(img)
  }
  
  #print("IMG_FILE_NAME")
  #print(img_file_name)
  
  salida <- substr(img_file_name, 0, 8)
  name <- paste(salida,type, sep="/")
  salida <- substr(img_file_name, 10,nchar(img_file_name) )
  #print("salida")
  #print(salida)
  salida <- paste(type,salida, sep="-")
  
  name2 <- paste(name,salida, sep="/")
  #print("NAME2")
  #print(name2)
  
  dir.create(name, showWarnings = FALSE)
  writeImage(img, name2)
}



tipos<-function(imagen){
  Imagenes(imagen,"flip")
  Imagenes(imagen,"flop")
  Imagenes(imagen,"rot")
  Imagenes(imagen,"trans")
}

for(i in 1:length(img_file_1_list)) {
  tipos(img_file_1_list[i])
}


for(i in 1:length(img_file_2_list)) {
  tipos(img_file_2_list[i])
}

for(i in 1:length(img_file_3_list)) {
  
  tipos(img_file_3_list[i])
  
}

print("FINALIZADO")