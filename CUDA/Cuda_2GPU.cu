/************************************************************
 * Program to solve a finite difference
 * discretization of the screened Poisson equation:
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * with zero Dirichlet boundary condition using the iterative
 * Jacobi method with overrelaxation.
 *
 * RHS (source) function
 *   f(x,y) = -alpha*(1-x^2)(1-y^2)-2*[(1-x^2)+(1-y^2)]
 *
 * Analytical solution to the PDE
 *   u(x,y) = (1-x^2)(1-y^2)
 *
 * Current Version: Christian Iwainsky, RWTH Aachen University
 * MPI C Version: Christian Terboven, RWTH Aachen University, 2006
 * MPI Fortran Version: Dieter an Mey, RWTH Aachen University, 1999 - 2005
 * Modified: Sanjiv Shah,        Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux,  Kuck and Associates, Inc. (KAI), 1998
 *
 * Unless READ_INPUT is defined, a meaningful input dataset is used (CT).
 *
 * Input : n     - grid dimension in x direction
 *         m     - grid dimension in y direction
 *         alpha - constant (always greater than 0.0)
 *         tol   - error tolerance for the iterative solver
 *         relax - Successice Overrelaxation parameter
 *         mits  - maximum iterations for the iterative solver
 *
 * On output
 *       : u(n,m)       - Dependent variable (solution)
 *       : f(n,m,alpha) - Right hand side function
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/*************************************************************
 * Performs one iteration of the Jacobi method and computes
 * the residual value.
 *
 * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
 * are BOUNDARIES and therefore not part of the solution.
 *************************************************************/

__global__ void halo_points(double *src_temp,double *dest_temp,int *maxXCount){
#define SRC_TEMP(XX,YY) src_temp[(YY)*(*maxXCount)+(XX)]
#define DEST_TEMP(XX,YY) dest_temp[(YY)*(*maxXCount)+(XX)]
    for(int i = 0 ; i < (*maxXCount); i ++){
        SRC_TEMP(i,(*maxXCount)/2 + 1) = DEST_TEMP(i,1);
        DEST_TEMP(i,0) = SRC_TEMP(i,(*maxXCount)/2);
    }
}

__global__ void one_jacobi_iteration(double *xStart, double *yStart,
                            int *maxXCount, int *maxYCount,
                            double *src, double *dst,
                            double *deltaX, double *deltaY,
                            double *alpha, double *omega,double *f_error)
{
    if (blockDim.x*blockIdx.x+threadIdx.x > (*maxXCount)){
        return;
    }

    double error = 0.0;
    __syncthreads();

#define SRC(XX,YY) src[(YY)*(*maxXCount)+(XX)]
#define DST(XX,YY) dst[(YY)*(*maxXCount)+(XX)]
    int x, y;
    double fX, fY;    
    double updateVal;
    double f;

    // Coefficients
    double cx = 1.0/((*deltaX)*(*deltaX));
    double cy = 1.0/((*deltaY)*(*deltaY));
    double cc = -2.0*cx-2.0*cy-(*alpha);
    y = blockDim.x*blockIdx.x+threadIdx.x;
   
        fY = (*yStart) + (y-1)*(*deltaY);
        for (x = 1; x < ((*maxXCount)-1); x++)
        {
            fX = (*xStart) + (x-1)*(*deltaX);
            f = -(*alpha)*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY);
            updateVal = ((SRC(x-1,y) + SRC(x+1,y))*cx +
                			(SRC(x,y-1) + SRC(x,y+1))*cy +
                			SRC(x,y)*cc - f
						)/cc;
            DST(x,y) = SRC(x,y) - (*omega)*updateVal;
            error += updateVal*updateVal;
        }
    f_error[blockDim.x*blockIdx.x+threadIdx.x] = error;
}

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double error;
    int iterationCount, maxIterationCount;
//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    int size_n = n+2;
    int size_m = m+2;

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    double *tmpc;

    double *Gpu_u_1;
    double *Gpu_u_2;
    double *Gpu_u_old1;
    double *Gpu_u_old2;
    // double *uc;
    // double *uc_old;
    
    cudaMalloc((void**)&Gpu_u_1, (n+2)*(m/2+2)*sizeof(double));
    cudaMalloc((void**)&Gpu_u_2, (n+2)*(m/2+2)*sizeof(double));

    cudaMalloc((void**)&Gpu_u_old1, (n+2)*(m/2+2)*sizeof(double));
    cudaMalloc((void**)&Gpu_u_old2, (n+2)*(m/2+2)*sizeof(double));
    
    double *Gpu_xLeft;
    double *Gpu_yBottom;
    int *Gpu_size_n;
    int *Gpu_size_m;
    double *Gpu_deltaX;
    double *Gpu_deltaY;
    double *Gpu_alpha;
    double *Gpu_relax;

    cudaMalloc((void**)&Gpu_xLeft,sizeof(double));
    cudaMalloc((void**)&Gpu_yBottom,sizeof(double));
    cudaMalloc((void**)&Gpu_size_n,sizeof(int));
    cudaMalloc((void**)&Gpu_size_m,sizeof(int));
    cudaMalloc((void**)&Gpu_deltaX,sizeof(double));
    cudaMalloc((void**)&Gpu_deltaY,sizeof(double));
    cudaMalloc((void**)&Gpu_alpha,sizeof(double));
    cudaMalloc((void**)&Gpu_relax,sizeof(double));

    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;
    
    cudaMemcpy(Gpu_xLeft,&xLeft,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_yBottom,&yBottom,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_size_n,&size_n,sizeof(int),cudaMemcpyHostToDevice);
   
    size_m = m/2+2;
    cudaMemcpy(Gpu_size_m,&size_m,sizeof(int),cudaMemcpyHostToDevice);   
    
    cudaMemcpy(Gpu_deltaX,&deltaX,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_deltaY,&deltaY,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_alpha,&alpha,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_relax,&relax,sizeof(double),cudaMemcpyHostToDevice);

    /* Iterate as long as it takes to meet the convergence criterion */
    double error2[n+2];

    double *return_Error1;
    double *return_Error2;
    
    cudaMalloc((void**)&return_Error1,(m/2+2)*sizeof(double));
    cudaMalloc((void**)&return_Error2,(m/2+2)*sizeof(double));

    int blocks = m/1024;
    blocks++;

    double tempError;
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    { 
        tempError = 0.0;
        
        halo_points<<<1,1>>>(Gpu_u_old1,Gpu_u_old2,Gpu_size_m);
        cudaSetDevice(0);
        one_jacobi_iteration<<<blocks,1024>>>(Gpu_xLeft, Gpu_yBottom,
                                    Gpu_size_n, Gpu_size_m,
                                    Gpu_u_old1, Gpu_u_1,
                                    Gpu_deltaX, Gpu_deltaY,
                                    Gpu_alpha, Gpu_relax,return_Error1);
        cudaMemcpy(error2,return_Error1,(m/2)*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i = 0 ;i < (m/2);i++){
            tempError+=error2[i];
        }

        cudaSetDevice(1);
        one_jacobi_iteration<<<blocks,1024>>>(Gpu_xLeft, Gpu_yBottom,
                                    Gpu_size_n, Gpu_size_m,
                                    Gpu_u_old1, Gpu_u_1,
                                    Gpu_deltaX, Gpu_deltaY,
                                    Gpu_alpha, Gpu_relax,return_Error2);                      
        cudaDeviceSynchronize();

        cudaMemcpy(error2,return_Error2,(m/2)*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i = 0 ;i < (m/2);i++){
            tempError+=error2[i];
        }
        iterationCount++;

        tmpc = Gpu_u_old1;
        Gpu_u_old1 = Gpu_u_1;
        Gpu_u_1 = tmpc;

        tmpc = Gpu_u_old2;
        Gpu_u_old2 = Gpu_u_2;
        Gpu_u_2 = tmpc;
        
        error = sqrt(tempError)/((n)*(m));
    }
    
    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

    // u_old holds the solution after the most recent buffers swap
    // cudaMemcpy(u_old,uc_old,allocCount*sizeof(double),cudaMemcpyDeviceToHost);
    // double absoluteError = checkSolution(xLeft, yBottom,
    //                                      n+2, m+2,
    //                                      u_old,
    //                                      deltaX, deltaY,
    //                                      alpha);
    // printf("The error of the iterative solution is %g\n", absoluteError);

    // cudaFree(uc_old);
    return 0;
}
