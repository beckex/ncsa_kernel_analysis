#include <iostream>
#include <cstring>
#include <cmath>

#include <stdlib.h>
#include "gdal_priv.h"
#include <ogr_spatialref.h>
#include "cpl_conv.h"
#include "sptw/sptw.h"

int main(int argc, char *argv[])
{
  
  // check arguments
  if (argc < 6)
    {
      std ::cout << "usage: ./" << argv[0] << ", input file name, output file name, kernel x size, kernel y size, kernel file name" << std::endl;
      return -1;
    }
  
  // primary arguments
  int raster_x_size, raster_y_size;
  double p_x_size, p_y_size;
  string input_file_name = argv[1];
  string output_file_name = argv[2];
  string kernel_file_name = argv[5];
  int kernel_x_size = atoi(argv[3]);
  int kernel_y_size = atoi(argv[4]);


  // get rank and size
  int rank, size;  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // read input file with gdal to get metadata 
  
  GDALDataset * poDataset;
  GDALAllRegister();  
  poDataset = (GDALDataset *) GDALOpen(input_file, GA_ReadOnly);
  if (poDataset == NULL)
    {
      perror("can't open input file");
      return -1;
    }

  geoTransform[6];
  x_size = poDataset->GetRasterXSize();
  y_size = poDataset->GetRasterYSize();
  if(poDataset->GetGeoTransform(geoTransform) == CE_None)
    {
      p_x_size = geoTransform[1];
      p_y_size = geoTransform[5];
    }  
  
}
