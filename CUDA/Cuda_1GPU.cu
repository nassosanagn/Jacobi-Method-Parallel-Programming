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

    x = blockDim.x*blockIdx.x+threadIdx.x+1;
    fX = (*xStart) + (x-1)*(*deltaX);

    for (y = 1; y < ((*maxYCount)-1); y++)
    {
        fY = (*yStart) + (y-1)*(*deltaY);
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
  Checks the error between numerical and exact solutions
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
    int allocCount;
    int iterationCount, maxIterationCount;

    /* Read data from the input */
    scanf("%d,%d", &n, &m);
    scanf("%lf", &alpha);
    scanf("%lf", &relax);
    scanf("%lf", &tol);
    scanf("%d", &mits);

    int size_n = n+2;
    int size_m = m+2;

    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);

    allocCount = (n+2)*(m+2);
    maxIterationCount = mits;
    maxAcceptableError = tol;
    
    double *Gpu_u, *Gpu_u_old;
    cudaMalloc((void**)&Gpu_u, allocCount*sizeof(double));
    cudaMalloc((void**)&Gpu_u_old, allocCount*sizeof(double));
    
    int *Gpu_size_n, *Gpu_size_m;
    double *tmpU, *u_old;
    double *Gpu_xLeft, *Gpu_yBottom;
    double *Gpu_deltaX, *Gpu_deltaY;
    double *Gpu_alpha, *Gpu_relax;

    /* Cuda Malloc all the variables needed for the jacobi_iteration */
    cudaMalloc((void**)&Gpu_xLeft,sizeof(double));
    cudaMalloc((void**)&Gpu_yBottom,sizeof(double));
    cudaMalloc((void**)&Gpu_size_n,sizeof(int));
    cudaMalloc((void**)&Gpu_size_m,sizeof(int));
    cudaMalloc((void**)&Gpu_deltaX,sizeof(double));
    cudaMalloc((void**)&Gpu_deltaY,sizeof(double));
    cudaMalloc((void**)&Gpu_alpha,sizeof(double));
    cudaMalloc((void**)&Gpu_relax,sizeof(double));

    /* Solve in [-1, 1] x [-1, 1] */
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;
    
    /* Initialize GPU variables */ 
    cudaMemcpy(Gpu_xLeft,&xLeft,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_yBottom,&yBottom,sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(Gpu_size_n,&size_n,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_size_m,&size_m,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(Gpu_deltaX,&deltaX,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_deltaY,&deltaY,sizeof(double),cudaMemcpyHostToDevice);
    
    cudaMemcpy(Gpu_alpha,&alpha,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Gpu_relax,&relax,sizeof(double),cudaMemcpyHostToDevice);

    double errorArray[m];

    
    double *return_Error;
    // cudaMalloc((void**)&return_Error,(size_n)*sizeof(double));

    int blocks = m/1024;
    blocks++;

    cudaMalloc((void**)&return_Error,m * sizeof(double));
    
    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    { 
        double tempError = 0.0;
     
        one_jacobi_iteration<<<blocks,1024>>>(Gpu_xLeft, Gpu_yBottom,
                                    Gpu_size_n, Gpu_size_m,
                                    Gpu_u_old, Gpu_u,
                                    Gpu_deltaX, Gpu_deltaY,
                                    Gpu_alpha, Gpu_relax, return_Error);

        cudaMemcpy(errorArray,return_Error,(m)*sizeof(double),cudaMemcpyDeviceToHost);
        for(int i = 0 ;i < m;i++){
            tempError+=errorArray[i];
        }

        iterationCount++;

        // Swap the buffers
        tmpU = Gpu_u_old;
        Gpu_u_old = Gpu_u;
        Gpu_u = tmpU;
        
        error = sqrt(tempError)/(n * m);
    }
    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

    u_old = (double*)calloc(allocCount, sizeof(double));

    // u_old holds the solution after the most recent buffers swap
    cudaMemcpy(u_old,Gpu_u_old,allocCount*sizeof(double),cudaMemcpyDeviceToHost);
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    cudaFree(Gpu_u);
    cudaFree(Gpu_u_old);
    return 0;
}
