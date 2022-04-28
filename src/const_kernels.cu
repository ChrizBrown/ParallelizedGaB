__constant__ int device_numBrow[648];
__constant__ int device_numBcol[1296];
__constant__ int device_ColumnDegree[1296];
__constant__ int device_RowDegree[648];

//#####################################################################################################
__device__ void DataPassGB_(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int N)
{
	int t,n,buf,Global;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
	
  if(n < N){ //make sure thread isnt out of bounds
    Global=(1-2*Receivedword[n]); 
		for (t=0;t<device_ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[device_numBcol[n]+t]]+1;

		for (t=0;t<device_ColumnDegree[n];t++)
		{
		  buf=Global-((-2)*CtoV[Interleaver[device_numBcol[n]+t]]+1);
		  if (buf<0)  VtoC[Interleaver[device_numBcol[n]+t]]= 1; //else VtoC[Interleaver[device_numBcol+t]]= 1;
		  else if (buf>0) VtoC[Interleaver[device_numBcol[n]+t]]= 0; //else VtoC[Interleaver[device_numBcol+t]]= 1;
		  else  VtoC[Interleaver[device_numBcol[n]+t]]=Receivedword[n];
		}
  }

  // if(n<N){
  //   printf("Finished DataPassGB_\n");
  // }
}
//#####################################################################################################
__device__ void DataPassGBIter0_(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int N)
{
  int n,t;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N){ //make sure thread isnt out of bounds
    for (t=0;t<device_ColumnDegree[n];t++)
    {
      VtoC[Interleaver[device_numBcol[n]+t]]=Receivedword[n];
    }  
  }

  // if(n<N){
  //   printf("Finished DataPassGBIter0_\n");
  // }	     
}
//##################################################################################################
__device__ void CheckPassGB_(int *CtoV,int *VtoC,int M)
{
   int t,m,signe;
   // m = curr thread index aka current row 
   m = threadIdx.x + blockIdx.x * blockDim.x;
   if(m < M){
	  signe=0;
      for (t=0;t<device_RowDegree[m];t++){
        signe^=VtoC[device_numBrow[m]+t];
      } 
	  for (t=0;t<device_RowDegree[m];t++){
        CtoV[device_numBrow[m]+t]=signe^VtoC[device_numBrow[m]+t];
      } 	
   }

  // if(m<M){
  //   printf("Finished CheckPassGB_\n");
  // }
}
//#####################################################################################################
__device__ void APP_GB_(int *Decide,int *CtoV,int *Receivedword,int *Interleaver,int N,int M)
{
  int t,n,Global;
  // n = curr thread index aka current column 
  n = threadIdx.x + blockIdx.x * blockDim.x;
  if(n<N){
	Global=(1-2*Receivedword[n]);
	for (t=0;t<device_ColumnDegree[n];t++){
      Global+=(-2)*CtoV[Interleaver[device_numBcol[n]+t]]+1;
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
__device__ int ComputeSyndrome_(int *Decide,int** Mat, int M)
{
  int Synd,k,l;
  // k = curr thread index aka current row 
  k = threadIdx.x + blockIdx.x * blockDim.x;
  
	if(k<M)
	{
		Synd=0;
		for (l=0;l<device_RowDegree[k];l++){
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


__global__ void global_decode_batched_stream(int *VtoC,int *CtoV,int** Mat,
                              int* Decide,int *Receivedword,int *Interleaver,int M,int N,
                              int iter, int* isCodeWord, int batchSize, int NbBranch)
{                          

  for(int batchIdx=0;batchIdx<batchSize;batchIdx++)
  {
    int batchDataOffset     = batchIdx*N;
    int batchRelationOffset = batchIdx*NbBranch;

    if(iter==0){
      DataPassGBIter0_(VtoC+batchRelationOffset,CtoV+batchRelationOffset,Receivedword+batchDataOffset,Interleaver,N);
    }
    else{
      DataPassGB_(VtoC+batchRelationOffset,CtoV+batchRelationOffset,Receivedword+batchDataOffset,Interleaver,N);
    }
    __syncthreads();

    CheckPassGB_(CtoV+batchRelationOffset,VtoC+batchRelationOffset,M);
    __syncthreads();

    APP_GB_(Decide+batchDataOffset,CtoV+batchRelationOffset,Receivedword+batchDataOffset,Interleaver,N,M);
    __syncthreads();  

    if(ComputeSyndrome_(Decide+batchDataOffset,Mat,M) == 0){
      isCodeWord[batchIdx] = 0;
    }
  }
}


__global__ void global_decode_batched_stream_shared(int *VtoC,int *CtoV,int** Mat,
                              int* Decide,int *Receivedword,int *Interleaver,int M,int N,
                              int iter, int* isCodeWord, int batchSize, int NbBranch){                          

  extern __shared__ int Decide_shared[];
  extern __shared__ int Receivedword_shared[];

  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (idx < N*batchSize)
  {
    for (int i = 0; i < batchSize; i++)
    {
      Decide_shared[idx+i*N] = Decide[idx+i*N];
      Receivedword_shared[idx+i*N] = Receivedword[idx+i*N];
    }
  } 
  __syncthreads();
  for(int batchIdx=0;batchIdx<batchSize;batchIdx++)
  {
    int batchDataOffset     = batchIdx*N;
    int batchRelationOffset = batchIdx*NbBranch;

    if(iter==0){
      DataPassGBIter0_(VtoC+batchRelationOffset,CtoV+batchRelationOffset,Receivedword_shared+batchDataOffset,Interleaver,N);
    }
    else{
      DataPassGB_(VtoC+batchRelationOffset,CtoV+batchRelationOffset,Receivedword_shared+batchDataOffset,Interleaver,N);
    }
    __syncthreads();

    CheckPassGB_(CtoV+batchRelationOffset,VtoC+batchRelationOffset,M);
    __syncthreads();

    APP_GB_(Decide_shared+batchDataOffset,CtoV+batchRelationOffset,Receivedword_shared+batchDataOffset,Interleaver,N,M);
    __syncthreads();  

    if(ComputeSyndrome_(Decide_shared+batchDataOffset,Mat,M) == 0){
      isCodeWord[batchIdx] = 0;
    }
  }
  if (idx < N*batchSize)
  {
    for (int i = 0; i < batchSize; i++)
    {
      Decide[idx+i*N] = Decide_shared[idx+i*N];
    }
  } 
}
