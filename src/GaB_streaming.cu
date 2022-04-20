/* ###########################################################################################################################
## Organization         : The University of Arizona
##                      :
## File name            : GaB.cu
## Language             : CUDA C (ANSI)
## Short description    : Parallel Based Gallager-B Hard decision Bit-Flipping algorithm
##                      :
##                      :
##                      :
## History              : Modified 19/01/2016, Created by Burak UNAL
##                      : Modified Spring 2022, Parallellized by Christopher Brown and Jared Causey
##                      :
## COPYRIGHT            : burak@email.arizona.edu
## ######################################################################################################################## */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include "kernels.cu"

#define arrondi(x) ((ceil(x)-x)<(x-floor(x))?(int)ceil(x):(int)floor(x))
#define min(x,y) ((x)<(y)?(x):(y))
#define signf(x) ((x)>=0?0:1)
#define	max(x,y) ((x)<(y)?(y):(x))
#define SQR(A) ((A)*(A))
#define BPSK(x) (1-2*(x))
#define PI 3.1415926536


//#####################################################################################################
void DataPassGB(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int* numBcol)
{
	int t,numB,n,buf;
	int Global;
	numB=0;
	for (n=0;n<N;n++)
	{
		//Global=(Amplitude)*(1-2*ReceivedSymbol[n]);
		Global=(1-2*Receivedword[n]); 
		//Global=(1-2*(Decide[n] + Receivedword[n])); //Decide[n]^Receivedword[n];
		for (t=0;t<ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[numB+t]]+1;

		for (t=0;t<ColumnDegree[n];t++)
		{
		  buf=Global-((-2)*CtoV[Interleaver[numB+t]]+1);
		  if (buf<0)  VtoC[Interleaver[numB+t]]= 1; //else VtoC[Interleaver[numB+t]]= 1;
		  else if (buf>0) VtoC[Interleaver[numB+t]]= 0; //else VtoC[Interleaver[numB+t]]= 1;
		  else  VtoC[Interleaver[numB+t]]=Receivedword[n];
		}
		numB=numB+ColumnDegree[n];
	}
}
//#####################################################################################################
void DataPassGBIter0(int *VtoC,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int* numBcol)
{
	int t,numB,n;
	numB=0;
	for (n=0;n<N;n++)
	{
		for (t=0;t<ColumnDegree[n];t++)     VtoC[Interleaver[numB+t]]=Receivedword[n];
		numB=numB+ColumnDegree[n];
	}
}
//##################################################################################################
void CheckPassGB(int *CtoV,int *VtoC,int M,int* numBrow,int *RowDegree)
{
   int t,numB=0,m,signe;
   for (m=0;m<M;m++)
   {
		signe=0;for (t=0;t<RowDegree[m];t++) signe^=VtoC[numB+t];
	    for (t=0;t<RowDegree[m];t++) 	CtoV[numB+t]=signe^VtoC[numB+t];
		numB=numB+RowDegree[m];
   }
}
//#####################################################################################################
void APP_GB(int *Decide,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int M,int* numBcol)
{
   	int t,numB,n;
	int Global;
	numB=0;
	for (n=0;n<N;n++)
	{
		Global=(1-2*Receivedword[n]);
		for (t=0;t<ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[numB+t]]+1;
        if(Global>0) Decide[n]= 0;
        else if (Global<0) Decide[n]= 1;
        else  Decide[n]=Receivedword[n];
		numB=numB+ColumnDegree[n];
	}
}
//#####################################################################################################
int ComputeSyndrome(int *Decide,int **Mat,int *RowDegree,int M)
{
	int Synd,k,l;

	for (k=0;k<M;k++)
	{
		Synd=0;
		for (l=0;l<RowDegree[k];l++) Synd=Synd^Decide[Mat[k][l]];
		if (Synd==1) break;
	}
	return(1-Synd);
}
//#####################################################################################################
int GaussianElimination_MRB(int *Perm,int **MatOut,int **Mat,int M,int N)
{
	int k,n,m,m1,buf,ind,indColumn,nb,*Index,dep,Rank;

	Index=(int *)calloc(N,sizeof(int));

	// Triangularization
	indColumn=0;nb=0;dep=0;
	for (m=0;m<M;m++)
	{
		if (indColumn==N) { dep=M-m; break; }

		for (ind=m;ind<M;ind++) { if (Mat[ind][indColumn]!=0) break; }
		// If a "1" is found on the column, permutation of rows
		if (ind<M)
		{
			for (n=indColumn;n<N;n++) { buf=Mat[m][n]; Mat[m][n]=Mat[ind][n]; Mat[ind][n]=buf; }
		// bottom of the column ==> 0
			for (m1=m+1;m1<M;m1++)
			{
				if (Mat[m1][indColumn]==1) { for (n=indColumn;n<N;n++) Mat[m1][n]=Mat[m1][n]^Mat[m][n]; }
			}
			Perm[m]=indColumn;
		}
		// else we "mark" the column.
		else { Index[nb++]=indColumn; m--; }

		indColumn++;
	}

	Rank=M-dep;

	for (n=0;n<nb;n++) Perm[Rank+n]=Index[n];

	// Permutation of the matrix
	for (m=0;m<M;m++) { for (n=0;n<N;n++) MatOut[m][n]=Mat[m][Perm[n]]; }

	// Diagonalization
	for (m=0;m<(Rank-1);m++)
	{
		for (n=m+1;n<Rank;n++)
		{
			if (MatOut[m][n]==1) { for (k=n;k<N;k++) MatOut[m][k]=MatOut[n][k]^MatOut[m][k]; }
		}
	}
	free(Index);
	return(Rank);
}

//#####################################################################################################
int main(int argc, char * argv[])
{
  // Variables Declaration
  FILE *f;
  int Graine,NbIter,nbtestedframes,NBframes;
  float alpha_max, alpha_min,alpha_step,alpha,NbMonteCarlo;
  // ----------------------------------------------------
  // lecture des param de la ligne de commande
  // ----------------------------------------------------
  char *FileName,*FileMatrix,*FileResult;
  FileName=(char *)malloc(200);
  FileMatrix=(char *)malloc(200);
  FileResult=(char *)malloc(200);

  strcpy(FileMatrix,argv[1]); 	// Matrix file
  strcpy(FileResult,argv[2]); 	// Results file

  //--------------Simulation input for GaB BF-------------------------
  NbMonteCarlo=100000000000;	    // Maximum nb of codewords sent
  NbIter=100; 	            // Maximum nb of iterations
  alpha= 0.01;              // Channel probability of error
  NBframes=100;	            // Simulation stops when NBframes in error
  Graine=1;		            // Seed Initialization for Multiple Simulations

    // brkunl
  alpha_max= 0.0600;		    //Channel Crossover Probability Max and Min
  alpha_min= 0.0200;
  alpha_step=0.0100;


  // ----------------------------------------------------
  // Load Matrix
  // ----------------------------------------------------
  int *ColumnDegree,*RowDegree,**Mat;
  int M,N,m,n,k;
  strcpy(FileName,FileMatrix);strcat(FileName,"_size");
  f=fopen(FileName,"r");fscanf(f,"%d",&M);fscanf(f,"%d",&N);
  ColumnDegree=(int *)calloc(N,sizeof(int));
  RowDegree=(int *)calloc(M,sizeof(int));fclose(f);
  strcpy(FileName,FileMatrix);strcat(FileName,"_RowDegree");
  f=fopen(FileName,"r");for (m=0;m<M;m++) fscanf(f,"%d",&RowDegree[m]);fclose(f);
  Mat=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) Mat[m]=(int *)calloc(RowDegree[m],sizeof(int));
  strcpy(FileName,FileMatrix);
  f=fopen(FileName,"r");for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) fscanf(f,"%d",&Mat[m][k]); }fclose(f);
  for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) ColumnDegree[Mat[m][k]]++; }

  printf("Matrix Loaded \n");

  // ----------------------------------------------------
  // Build Graph
  // ----------------------------------------------------
  int NbBranch,**NtoB,*Interleaver,*ind,numColumn,numBranch;
  NbBranch=0; for (m=0;m<M;m++) NbBranch=NbBranch+RowDegree[m];
  NtoB=(int **)calloc(N,sizeof(int *)); for (n=0;n<N;n++) NtoB[n]=(int *)calloc(ColumnDegree[n],sizeof(int));
  Interleaver=(int *)calloc(NbBranch,sizeof(int));
  ind=(int *)calloc(N,sizeof(int));
  numBranch=0;for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { numColumn=Mat[m][k]; NtoB[numColumn][ind[numColumn]++]=numBranch++; } }
  free(ind);
  numBranch=0;for (n=0;n<N;n++) { for (k=0;k<ColumnDegree[n];k++) Interleaver[numBranch++]=NtoB[n][k]; }

  printf("Graph Build \n");

  // ----------------------------------------------------
  // Decoder
  // ----------------------------------------------------
  int *CtoV,*VtoC,*Codeword,*Receivedword,*Decide,*U,l,*numBrow,*numBcol;
  int iter;
  CtoV=(int *)calloc(NbBranch,sizeof(int));
  VtoC=(int *)calloc(NbBranch,sizeof(int));
  Codeword=(int *)calloc(N,sizeof(int));
  Receivedword=(int *)calloc(N,sizeof(int));
  Decide=(int *)calloc(N,sizeof(int));
  U=(int *)calloc(N,sizeof(int));
  srand48(time(0)+Graine*31+113);

  //precompute numB values
  numBrow=(int *)calloc(M,sizeof(int));
 	int numB=0;
  for (m=0;m<M;m++)
  {
    // if(m == M-1){
    //   printf("numBrow[%d]= %d\n",m,numB);
    // }
		numBrow[m] = numB;
    // printf("%d\n", numB);
    numB=numB+RowDegree[m];
  }
    // exit(0);


  numBcol=(int *)calloc(N,sizeof(int));
	numB=0;
	for (n=0;n<N;n++)
	{
    // if(n == N-1){
    //   printf("numBcol[%d]= %d\n",n,numB);
    // }
		numBcol[n] = numB;
    // printf("%d\n", numB);
		numB=numB+ColumnDegree[n];
	}

  // ----------------------------------------------------
  // Allocate and fill GPU Data for Matrix and Decoder
  // ----------------------------------------------------
  int *device_ColumnDegree,*device_RowDegree,**device_Mat,*device_Interleaver,*device_numBrow,*device_numBcol;
  
  // Initialize and Fill Matrix and Degree Arrays on Device (Should never be modified)
  cudaMalloc((void **)&device_Mat, M * sizeof(int*));
  int** temp_i_ptrs = (int**) malloc(M * sizeof(int*));
  for (m=0;m<M;m++){
    cudaMalloc((void**)&temp_i_ptrs[m], RowDegree[m] * sizeof(int));
    cudaMemcpy(temp_i_ptrs[m], Mat[m], RowDegree[m] * sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(device_Mat, temp_i_ptrs, sizeof(int*) * M, cudaMemcpyHostToDevice);

  // for(int a=0;a<M;a++){for(int b=0;b<RowDegree[a];b++){printf("%d\n",Mat[a][b]);}}

  cudaMalloc((void **)&device_RowDegree, M * sizeof(int));
  cudaMemcpy(device_RowDegree, RowDegree, M * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_ColumnDegree, N * sizeof(int));
  cudaMemcpy(device_ColumnDegree, ColumnDegree, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_Interleaver, NbBranch * sizeof(int));
  cudaMemcpy(device_Interleaver, Interleaver, NbBranch * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_numBrow, M * sizeof(int));
  cudaMemcpy(device_numBrow, numBrow, M * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_numBcol, N * sizeof(int));
  cudaMemcpy(device_numBcol, numBcol, N * sizeof(int), cudaMemcpyHostToDevice);

  int *device_CtoV,*device_VtoC,*device_Codeword,*device_Receivedword,*device_Decide,*device_IsCodeword;

  // Initialize GaB node connections and Codeword Arrays on Device
  cudaMalloc((void **)&device_CtoV, NbBranch * sizeof(int));
  cudaMemset((void **)&device_CtoV, 0, NbBranch * sizeof(int));
  
  cudaMalloc((void **)&device_VtoC, NbBranch * sizeof(int));
  cudaMemset((void **)&device_VtoC, 0, NbBranch * sizeof(int));
  
  const int code_word_size = N;//8 * sizeof(int);
  cudaMalloc((void **)&device_Codeword, N * sizeof(int));
  cudaMemset((void **)&device_Codeword, 0, N * sizeof(int));  
  
  cudaMalloc((void **)&device_Receivedword, code_word_size * sizeof(int));
  cudaMemset((void **)&device_Receivedword, 0, code_word_size * sizeof(int));   
  
  cudaMalloc((void **)&device_Decide, code_word_size * sizeof(int));
  cudaMemset((void **)&device_Decide, 0, code_word_size * sizeof(int));   

  cudaMalloc((void **)&device_IsCodeword, sizeof(int));
  
  // Set Up GPU Kernel Dimensions
  dim3 blockDim(M),gridDim(32);

  // dim3 blockDim,gridDim(32);
  // if(M > N){
  //   blockDim = (M/2);
  // }
  // else{
  //   blockDim = (N/2);
  // }

  // ----------------------------------------------------
  // Initialize Timing Structures
  // ----------------------------------------------------
  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);

  // ----------------------------------------------------
  // Gaussian Elimination for the Encoding Matrix (Full Representation)
  // ----------------------------------------------------
  int **MatFull,**MatG,*PermG;
  int rank;
  MatG=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatG[m]=(int *)calloc(N,sizeof(int));
  MatFull=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatFull[m]=(int *)calloc(N,sizeof(int));
  PermG=(int *)calloc(N,sizeof(int)); for (n=0;n<N;n++) PermG[n]=n;
  for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { MatFull[m][Mat[m][k]]=1; } }
  rank=GaussianElimination_MRB(PermG,MatG,MatFull,M,N);
  //for (m=0;m<N;m++) printf("%d\t",PermG[m]);printf("\n");

  // Variables for Statistics
  int IsCodeword,nb;
  int NiterMoy,NiterMax;
  int Dmin;
  int NbTotalErrors,NbBitError;
  int NbUnDetectedErrors,NbError;
  float timeAverage;


  strcpy(FileName,FileResult);
  f=fopen(FileName,"w");
  fprintf(f,"-------------------------Gallager B--------------------------------------------------\n");
  fprintf(f,"alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\t\tTimePerFrame\n");

  printf("-------------------------Gallager B--------------------------------------------------\n");
  printf("alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)\t\tTimePerFrame\n");

  // Set up CUDA stream objects
  char* temp;
  const int num_streams = strtol(argv[3],&temp,10);
  printf("Creating %d streams\n",num_streams);
  cudaStream_t* pStreams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
  for (int i = 0; i < num_streams; i++)
    cudaStreamCreate(&(pStreams[i]));

  for(alpha=alpha_max;alpha>=alpha_min;alpha-=alpha_step) {

  NiterMoy=0;NiterMax=0;
  Dmin=1e5;
  NbTotalErrors=0;NbBitError=0;
  NbUnDetectedErrors=0;NbError=0;
  timeAverage=0.0;
  //--------------------------------------------------------------
  for (nb=0,nbtestedframes=0;nb<NbMonteCarlo;nb++)
  {
  //encoding
  for (k=0;k<rank;k++) U[k]=0;
	for (k=rank;k<N;k++) U[k]=floor(drand48()*2);
	for (k=rank-1;k>=0;k--) { for (l=k+1;l<N;l++) U[k]=U[k]^(MatG[k][l]*U[l]); }
	for (k=0;k<N;k++) Codeword[PermG[k]]=U[k];

	// All zero codeword
	//for (n=0;n<N;n++) { Codeword[n]=0; }

  // Add Noise
  for (n=0;n<N;n++)  if (drand48()<alpha) Receivedword[n]=1-Codeword[n]; else Receivedword[n]=Codeword[n];
  //============================================================================
 	// Decoder
	//============================================================================
  cudaEventRecord(astartEvent, 0);
  if(argc == 4){ //parallel
    //printf("decoding\n");
    for (int stream_cnt = 0; stream_cnt < num_streams; stream_cnt++)
    {
      // Copy Received Word to the GPU
      cudaMemcpyAsync(device_Decide, Receivedword/*+(code_word_size*stream_cnt)*/, code_word_size * sizeof(int), cudaMemcpyHostToDevice, pStreams[stream_cnt]);
      cudaMemcpyAsync(device_Receivedword, Receivedword/*+(code_word_size*stream_cnt)*/, code_word_size * sizeof(int), cudaMemcpyHostToDevice, pStreams[stream_cnt]);
    }
    for (int stream_cnt = 0; stream_cnt < num_streams; stream_cnt++)
    {
      for (iter=0;iter<NbIter;iter++)
      {
        // Reset IsCodeword
        cudaMemsetAsync(device_IsCodeword, 1, sizeof(int), pStreams[stream_cnt]);
        // Call Decode
        global_decode<<<gridDim,blockDim,0,pStreams[stream_cnt]>>>(device_VtoC,device_CtoV,device_Mat,device_RowDegree,device_ColumnDegree,
                                            device_Decide,device_Receivedword,device_Interleaver,M,N,
                                            device_numBrow,device_numBcol,iter,device_IsCodeword);
        //Retreive IsCodeWord
        cudaMemcpyAsync(&IsCodeword,device_IsCodeword, sizeof(int), cudaMemcpyDeviceToHost,pStreams[stream_cnt]);
        if (IsCodeword)
          break;
      }
      // Stream 1
      cudaMemcpyAsync(Decide, device_Decide/*+(code_word_size*stream_cnt)*/, code_word_size * sizeof(int), cudaMemcpyDeviceToHost,pStreams[stream_cnt]);
    }
  }
  else{ //serial
    // REPLACE THE CODE BELOW WITH CUDA KERNEL CALLS -------------------------------------------------
    for (k=0;k<NbBranch;k++) {CtoV[k]=0;}

    for (k=0;k<N;k++) Decide[k]=Receivedword[k];

    for (iter=0;iter<NbIter;iter++){
        if(iter==0){
          DataPassGBIter0(VtoC,CtoV,Receivedword,Interleaver,ColumnDegree,N,numBcol);
        }
        else{
          DataPassGB(VtoC,CtoV,Receivedword,Interleaver,ColumnDegree,N,numBcol);
        }
        
        CheckPassGB(CtoV,VtoC,M,numBrow,RowDegree);
        APP_GB(Decide,CtoV,Receivedword,Interleaver,ColumnDegree,N,M,numBcol);
        
        IsCodeword=ComputeSyndrome(Decide,Mat,RowDegree,M);
        if (IsCodeword){
          break;
        } 
    }
    // -----------------------------------------------------------------------------------------------
  }
  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  timeAverage += aelapsedTime;
	//============================================================================
  	// Compute Statistics
	//============================================================================
  nbtestedframes++;
	NbError=0;for (k=0;k<N;k++)  if (Decide[k]!=Codeword[k]) NbError++;
	NbBitError=NbBitError+NbError;
	
  // Case Divergence
	if (!IsCodeword)
	{
	  NiterMoy=NiterMoy+NbIter;
		NbTotalErrors++;
	}
	
  // Case Convergence to Right Codeword
	if ((IsCodeword)&&(NbError==0)) { NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1); }
	
  // Case Convergence to Wrong Codeword
	if ((IsCodeword)&&(NbError!=0))
	{
	  NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1);
	  NbTotalErrors++; NbUnDetectedErrors++;
	  Dmin=min(Dmin,NbError);
	}

	// Stopping Criterion
	if (NbTotalErrors==NBframes) break;
  }

  float timeAveragePerNb = timeAverage/nbtestedframes;
  
  printf("%1.5f\t\t",alpha);
  printf("%10d (%1.16f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
  printf("%4d (%1.16f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
  printf("%10d\t\t",nbtestedframes);
  printf("%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
  printf("%d(%d)\t\t",NbUnDetectedErrors,Dmin);
  printf("%f\n",timeAveragePerNb);


  fprintf(f,"%1.5f\t\t",alpha);
  fprintf(f,"%10d (%1.8f)\t\t",NbBitError,(float)NbBitError/N/nbtestedframes);
  fprintf(f,"%4d (%1.8f)\t\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
  fprintf(f,"%10d\t\t",nbtestedframes);
  fprintf(f,"%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
  fprintf(f,"%d(%d)\t\t",NbUnDetectedErrors,Dmin);
  fprintf(f,"%f\n",timeAveragePerNb);

}

for (int i = 0; i < num_streams; i++)
  cudaStreamDestroy(pStreams[i]);
free(pStreams);

// Free up GPU memory
cudaFree(device_Mat);
cudaFree(device_RowDegree);
cudaFree(device_ColumnDegree);
cudaFree(device_Interleaver);
cudaFree(device_numBrow);
cudaFree(device_numBcol);
cudaFree(device_CtoV);
cudaFree(device_VtoC);
cudaFree(device_Codeword);
cudaFree(device_Receivedword);
cudaFree(device_Decide);
cudaFree(device_IsCodeword);

fclose(f);
return(0);
}
