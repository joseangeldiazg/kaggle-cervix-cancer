#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Set run parameters parameters
test_img_path <- "."
width  <- 256
height <- 256


#-------------------------------------------------------------------------------
# Load and pre-process train images
#-------------------------------------------------------------------------------

# Load EBImage library
library(EBImage)

# Load images into a dataframe
library(gsubfn)
img_file_list <- list.files(path = test_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)



train_df <- data.frame()

for(i in 1:length(img_file_list)) {
  img_file_name <- img_file_list[i]
  #img_class <- strapplyc(img_file_list[i], ".*/Type_(.*)/")[[1]]
  img <- readImage(img_file_name)
  img_resized <- resize(img, w=width, h=height)
  
  name <- paste("reducidas",img_file_name, sep="/")
  dir.create("reducidas", showWarnings = FALSE)
  writeImage(img_resized, name)
  
}
