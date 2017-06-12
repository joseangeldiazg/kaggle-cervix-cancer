#-------------------------------------------------------------------------------
# This script cut the images to prevent original data deformation 
#-------------------------------------------------------------------------------


#setwd("D:/Facultad/Master/Segundo cuatrimestre/SIGE/PracticaFinal")
# Clear workspace
rm(list=ls())

# Set run parameters parameters
test_img_path <- "."

#-------------------------------------------------------------------------------
# Load train images
#-------------------------------------------------------------------------------

# Load EBImage library
library(EBImage)
# Load images into a dataframe
library(gsubfn)
img_file_list <- list.files(path = test_img_path, pattern = "*.jpg", full.names = TRUE, recursive = FALSE)


img_file_list
dir.create("crop", showWarnings = FALSE)
for(i in 1:length(img_file_list)) {
  
  img_file_name <- img_file_list[i]
  name <- paste("crop",img_file_name, sep="/")
  print(name)
  if(!file.exists(name)){
    img <- readImage(img_file_name)
    width <- dim(img)[1]
    heigth <- dim(img)[2]
    if (width < heigth){ 
      cut <- (heigth-width)/2
      p <- heigth-cut
      img2 <- img[0:width,cut:p,]
    }else{
      cut <- (width-heigth)/2
      p <- width-cut
      img2 <- img[cut:p,0:heigth,]
    }
    
    
    
    writeImage(img2, name)
  }
  
}
print("Finalizado")