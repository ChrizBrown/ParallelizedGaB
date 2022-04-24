//#####################################################################################################
__device__ void DataPassGB_(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int *numB)
{
	int t,n,buf,Global;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
	
  if(n < N){ //make sure thread isnt out of bounds
    Global=(1-2*Receivedword[n]); 
		for (t=0;t<ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[numB[n]+t]]+1;

		for (t=0;t<ColumnDegree[n];t++)
		{
		  buf=Global-((-2)*CtoV[Interleaver[numB[n]+t]]+1);
		  if (buf<0)  VtoC[Interleaver[numB[n]+t]]= 1; //else VtoC[Interleaver[numB+t]]= 1;
		  else if (buf>0) VtoC[Interleaver[numB[n]+t]]= 0; //else VtoC[Interleaver[numB+t]]= 1;
		  else  VtoC[Interleaver[numB[n]+t]]=Receivedword[n];
		}
  }

  // if(n<N){
  //   printf("Finished DataPassGB_\n");
  // }
}
//#####################################################################################################
__device__ void DataPassGBIter0_(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int *numB)
{
  int n,t;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N){ //make sure thread isnt out of bounds
    for (t=0;t<ColumnDegree[n];t++)
    {
      VtoC[Interleaver[numB[n]+t]]=Receivedword[n];
    }  
  }

  // if(n<N){
  //   printf("Finished DataPassGBIter0_\n");
  // }	     
}
//##################################################################################################
__device__ void CheckPassGB_(int *CtoV,int *VtoC,int M,int *RowDegree, int *numB)
{
   int t,m,signe;
   // m = curr thread index aka current row 
   m = threadIdx.x + blockIdx.x * blockDim.x;
   if(m < M){
	  signe=0;
      for (t=0;t<RowDegree[m];t++){
        signe^=VtoC[numB[m]+t];
      } 
	  for (t=0;t<RowDegree[m];t++){
        CtoV[numB[m]+t]=signe^VtoC[numB[m]+t];
      } 	
   }

  // if(m<M){
  //   printf("Finished CheckPassGB_\n");
  // }
}
//#####################################################################################################
__device__ void APP_GB_(int *Decide,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int M, int *numB)
{
  int t,n,Global;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
  if(n<N){
	Global=(1-2*Receivedword[n]);
	for (t=0;t<ColumnDegree[n];t++){
      Global+=(-2)*CtoV[Interleaver[numB[n]+t]]+1;
    }
    if(Global>0) Decide[n]= 0;
    else if (Global<0) Decide[n]= 1;
    else  Decide[n]=Receivedword[n];
  }

  // if(n<N){
  //   printf("Finished APP_GB_\n");
  // }
}
// //#####################################################################################################
__device__ int ComputeSyndrome_(int *Decide,int **Mat,int *RowDegree,int M)
{
  int Synd,k,l;
  // k = curr thread index aka current row 
  k = threadIdx.x + blockIdx.x * blockDim.x;
  
	if(k<M)
	{
		Synd=0;
		for (l=0;l<RowDegree[k];l++){
      Synd=Synd^Decide[Mat[k][l]];
    }

    if(Synd == 1){
      return (0);
    }
	}
  
  return (-1);

  // if(k<M){
  //   printf("Finished ComputeSyndrome_\n");
  // }
}

// version 0
// global memory only version
__global__ void global_decode(int *VtoC,int *CtoV,int** Mat,int* RowDegree,int* ColumnDegree,
                              int* Decide,int *Receivedword,int *Interleaver,int M,int N,
                              int *numBrow,int *numBcol, int iter, int* isCodeWord){

  // int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(iter==0){
    DataPassGBIter0_(VtoC,CtoV,Receivedword,Interleaver,ColumnDegree,N,numBcol);
  }
  else{
    DataPassGB_(VtoC,CtoV,Receivedword,Interleaver,ColumnDegree,N,numBcol);
  }
  __syncthreads();
  
  CheckPassGB_(CtoV,VtoC,M,RowDegree,numBrow);
  __syncthreads();

  APP_GB_(Decide,CtoV,Receivedword,Interleaver,ColumnDegree,N,M,numBcol);
  __syncthreads();

  if(ComputeSyndrome_(Decide,Mat,RowDegree,M) == 0){
    *isCodeWord = 0;
  }
}

__global__ void global_decode_stream(int *VtoC,int *CtoV,int** Mat,int* RowDegree,int* ColumnDegree,
                              int* Decide,int *Receivedword,int *Interleaver,int M,int N,
                              int *numBrow,int *numBcol, int iter, int* isCodeWord, int num_codewords, int* stop){

  for (int i = 0; i < num_codewords; i++)
  {
    if (stop[i])
      continue;

    if(iter==0){
      DataPassGBIter0_(&VtoC[i*N],&CtoV[i*N],&Receivedword[i*N],Interleaver,ColumnDegree,N,numBcol);
    }
    else{
      DataPassGB_(&VtoC[i*N],&CtoV[i*N],&Receivedword[i*N],Interleaver,ColumnDegree,N,numBcol);
    }
    __syncthreads();
    
    CheckPassGB_(&CtoV[i*N],&VtoC[i*N],M,RowDegree,numBrow);
    __syncthreads();

    APP_GB_(&Decide[i*N],&CtoV[i*N],&Receivedword[i*N],Interleaver,ColumnDegree,N,M,numBcol);
    __syncthreads();

    if(ComputeSyndrome_(&Decide[i*N],Mat,RowDegree,M) == 0){
      isCodeWord[i] = 0;
    }
    __syncthreads();
  }
}
