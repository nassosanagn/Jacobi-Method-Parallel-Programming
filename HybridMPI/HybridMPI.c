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
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*************************************************************
 * Performs one iteration of the Jacobi method and computes
 * the residual value.
 *
 * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
 * are BOUNDARIES and therefore not part of the solution.
 *************************************************************/

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha) {
#define U(XX, YY) u[(YY)*maxXCount + (XX)]
  int x, y;
  double fX, fY;
  double localError, error = 0.0;

  for (y = 1; y < (maxYCount - 1); y++) {
    fY = yStart + (y - 1) * deltaY;
    for (x = 1; x < (maxXCount - 1); x++) {
      fX = xStart + (x - 1) * deltaX;
      localError = U(x, y) - (1.0 - fX * fX) * (1.0 - fY * fY);
      error += localError * localError;
    }
  }
  return error;
}

int main(int argc, char **argv) {
  int n, m, mits, comm_sz, my_rank;
  double alpha, tol, relax;
  double maxAcceptableError;
  double error;
  double global_sum;
  double global_error;
  double *u, *u_old, *tmp;
  int allocCount;
  int iterationCount, maxIterationCount;
  double t1, t2;

  MPI_Comm initialComm;
  initialComm = MPI_COMM_WORLD;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(initialComm, &comm_sz);

  //Create mpi Cartesian Topology
  int dims[2] = {0, 0};
  MPI_Dims_create(comm_sz, 2, dims);
  int periods[2] = {false, false};
  MPI_Comm comm;
  MPI_Cart_create(initialComm, 2, dims, periods, true, &comm);

  MPI_Comm_rank(comm, &my_rank);

  int my_coords[2];
  MPI_Cart_coords(comm, my_rank, 2, my_coords);
  if (my_rank == 0) {
    scanf("%d,%d", &n, &m);
    scanf("%lf", &alpha);
    scanf("%lf", &relax);
    scanf("%lf", &tol);
    scanf("%d", &mits);
    printf("-> rank %d :  %d, %d, %g, %g, %g, %d\n", my_rank, n, m, alpha, relax, tol, mits);
    allocCount = (n + 2) * (m + 2);
  }

  //Broadcast matrix data to other processes
  MPI_Bcast(&n, 1, MPI_INT, 0, comm);
  MPI_Bcast(&m, 1, MPI_INT, 0, comm);
  MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&mits, 1, MPI_INT, 0, comm);

  // Those two calls also zero the boundary elements

  // printf("-> rank %d, %d \n", my_rank, n);

  maxIterationCount = mits;
  maxAcceptableError = tol;

  // Solve in [-1, 1] x [-1, 1]
  double xLeft = -1.0, xRight = 1.0;
  double yBottom = -1.0, yUp = 1.0;

  iterationCount = 0;
  error = HUGE_VAL;
  clock_t start = clock(), diff;

  int squareComm = (int)sqrt(comm_sz);
  int size_n = n / squareComm;
  int size_m = m / squareComm;

  double deltaX = (xRight - xLeft) / (n - 1);
  double deltaY = (yUp - yBottom) / (m - 1);

  MPI_Barrier(comm);
  t1 = MPI_Wtime();

  //Calculate initial positions in local matrix
  xLeft = xLeft + deltaX * size_n * (my_rank % squareComm);
  yBottom = yBottom + deltaY * size_m * (((comm_sz - squareComm) / squareComm) - (my_rank / squareComm));

  enum directions { DOWN,
                    UP,
                    LEFT,
                    RIGHT };
  int neighbour_ranks[4];

  MPI_Cart_shift(comm, 0, 1, &neighbour_ranks[LEFT], &neighbour_ranks[RIGHT]);
  MPI_Cart_shift(comm, 1, 1, &neighbour_ranks[DOWN], &neighbour_ranks[UP]);
  MPI_Comm_rank(comm, &my_rank);

  u = (double *)calloc(((size_n + 2) * (size_m + 2)), sizeof(double)); //reverse order
  u_old = (double *)calloc(((size_n + 2) * (size_m + 2)), sizeof(double));

  if (u == NULL || u_old == NULL) {
    printf("Not enough memory for two %ix%i matrices\n", n + 2, m + 2);
    exit(1);
  }

  while (iterationCount < maxIterationCount && error > maxAcceptableError) {

#define SRC(XX, YY) u_old[(YY) * (size_n + 2) + (XX)]
#define DST(XX, YY) u[(YY) * (size_n + 2) + (XX)]

    int x, y;

    double fX, fY;
    double updateVal;
    double f;

    // Coefficients
    double cx = 1.0 / (deltaX * deltaX);
    double cy = 1.0 / (deltaY * deltaY);
    double cc = -2.0 * cx - 2.0 * cy - alpha;
    error = 0.0;

    //Define Column and Row Types
    MPI_Datatype column_type;
    MPI_Type_vector(size_n, 1, size_n + 2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    MPI_Datatype row_type;
    MPI_Type_contiguous(size_n, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Request sendRequests[4];
    MPI_Request receiveRequests[4];

    double receivedRight[size_n];      //create temp right matrix
    double receivedLeft[size_n];       //create temp left matrix
    double receivedUp[size_n];         //create temp left matrix
    double receivedDown[size_n];       //create temp left matrix
    for (int i = 0; i < size_n; i++) { //initialize temp matrices
      receivedLeft[i] = 0.0;
      receivedRight[i] = 0.0;
      receivedUp[i] = 0.0;
      receivedDown[i] = 0.0;
    }

    //Receive green rows and columns from neighbours
    MPI_Irecv(&receivedLeft, size_n, MPI_DOUBLE, neighbour_ranks[0], 0, comm, &receiveRequests[0]);
    MPI_Irecv(&receivedRight, size_n, MPI_DOUBLE, neighbour_ranks[1], 0, comm, &receiveRequests[1]);
    MPI_Irecv(&receivedUp, size_n, MPI_DOUBLE, neighbour_ranks[2], 0, comm, &receiveRequests[2]);
    MPI_Irecv(&receivedDown, size_n, MPI_DOUBLE, neighbour_ranks[3], 0, comm, &receiveRequests[3]);

    //Send green rows and columns to neighbours
    MPI_Isend(&(SRC(1, 1)), 1, column_type, neighbour_ranks[0], 0, comm, &sendRequests[0]);
    MPI_Isend(&(SRC(size_n, 1)), 1, column_type, neighbour_ranks[1], 0, comm, &sendRequests[1]);
    MPI_Isend(&(SRC(1, 1)), 1, row_type, neighbour_ranks[2], 0, comm, &sendRequests[2]);
    MPI_Isend(&(SRC(1, size_n)), 1, row_type, neighbour_ranks[3], 0, comm, &sendRequests[3]);

#pragma omp parallel for num_threads(2 * comm_sz) collapse(2) private(fX, f, fY, updateVal) \
    reduction(+                                                                             \
              : error) schedule(static, 1)
    for (y = 2; y < (size_m); y++) { // calculate internal white boxes
      for (x = 2; x < (size_n); x++) {
        fY = yBottom + (y - 1) * deltaY;
        fX = xLeft + (x - 1) * deltaX;
        f = -alpha * (1.0 - fX * fX) * (1.0 - fY * fY) - 2.0 * (1.0 - fX * fX) - 2.0 * (1.0 - fY * fY);
        updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                     (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                     SRC(x, y) * cc - f) /
                    cc;
        DST(x, y) = SRC(x, y) - relax * updateVal;
        error += updateVal * updateVal;
      }
    }

    MPI_Waitall(4, receiveRequests, MPI_STATUSES_IGNORE); //wait for receive requests to be completed

    for (int i = 1; i < size_n + 1; i++) { //copy columns from temp arrays to SRC
      SRC(size_n + 1, i) = receivedRight[i - 1];
      SRC(0, i) = receivedLeft[i - 1];

      SRC(i, 0) = receivedUp[i - 1];
      SRC(i, size_n + 1) = receivedDown[i - 1];
    }

    y = 1; //calculate first column
    fY = yBottom + (y - 1) * deltaY;
#pragma omp parallel for num_threads(2 * comm_sz) private(fX, f, updateVal) \
    reduction(+                                                             \
              : error)
    for (x = 1; x < size_n + 1; x++) {
      fX = xLeft + (x - 1) * deltaX;
      f = -alpha * (1.0 - fX * fX) * (1.0 - fY * fY) - 2.0 * (1.0 - fX * fX) - 2.0 * (1.0 - fY * fY);
      updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                   (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                   SRC(x, y) * cc - f) /
                  cc;
      DST(x, y) = SRC(x, y) - relax * updateVal;
      error += updateVal * updateVal;
    }

    y = size_m; // calcualate last column
    fY = yBottom + (y - 1) * deltaY;
#pragma omp parallel for num_threads(2 * comm_sz) private(fX, f, updateVal) \
    reduction(+                                                             \
              : error)
    for (x = 1; x < size_n + 1; x++) {
      fX = xLeft + (x - 1) * deltaX;
      f = -alpha * (1.0 - fX * fX) * (1.0 - fY * fY) - 2.0 * (1.0 - fX * fX) - 2.0 * (1.0 - fY * fY);
      updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                   (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                   SRC(x, y) * cc - f) /
                  cc;
      DST(x, y) = SRC(x, y) - relax * updateVal;
      error += updateVal * updateVal;
    }

    x = 1; // calculate first row
    fX = xLeft + (x - 1) * deltaX;
#pragma omp parallel for num_threads(2 * comm_sz) private(fY, f, updateVal) \
    reduction(+                                                             \
              : error)
    for (y = 2; y < size_m; y++) {
      fY = yBottom + (y - 1) * deltaY;
      f = -alpha * (1.0 - fX * fX) * (1.0 - fY * fY) - 2.0 * (1.0 - fX * fX) - 2.0 * (1.0 - fY * fY);
      updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                   (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                   SRC(x, y) * cc - f) /
                  cc;
      DST(x, y) = SRC(x, y) - relax * updateVal;
      error += updateVal * updateVal;
    }

    x = size_n; //calculate last row
    fX = xLeft + (x - 1) * deltaX;
#pragma omp parallel for num_threads(2 * comm_sz) private(fY, f, updateVal) \
    reduction(+                                                             \
              : error)
    for (y = 2; y < size_m; y++) {
      fY = yBottom + (y - 1) * deltaY;
      f = -alpha * (1.0 - fX * fX) * (1.0 - fY * fY) - 2.0 * (1.0 - fX * fX) - 2.0 * (1.0 - fY * fY);
      updateVal = ((SRC(x - 1, y) + SRC(x + 1, y)) * cx +
                   (SRC(x, y - 1) + SRC(x, y + 1)) * cy +
                   SRC(x, y) * cc - f) /
                  cc;
      DST(x, y) = SRC(x, y) - relax * updateVal;
      error += updateVal * updateVal;
    }

    MPI_Allreduce(&error, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //reduce local error to global sum
    error = sqrt(global_sum) / ((n - 4) * (m - 4));

    MPI_Waitall(4, sendRequests, MPI_STATUSES_IGNORE); //wait for send requests to be completed

    iterationCount++;
    tmp = u_old;
    u_old = u;
    u = tmp;
  }

  t2 = MPI_Wtime();
  if (my_rank == 0) {
    printf("Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1);
  }

  diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;

  if (my_rank == 0) {
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
    printf("Residual %g\n", error);
  }

  // u_old holds the solution after the most recent buffers swap
  double absoluteError = checkSolution(xLeft, yBottom,
                                       size_n + 2, size_m + 2,
                                       u_old,
                                       deltaX, deltaY,
                                       alpha);

  MPI_Allreduce(&absoluteError, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  global_error = sqrt(global_error) / (n * m);
  printf("The error of the iterative solution is %g\n", global_error);

  MPI_Finalize();

  return 0;
}
