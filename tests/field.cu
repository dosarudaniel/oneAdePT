__host__ __device__ double stepInField(double kinE, double mass, int charge)
{
   return(kinE*mass*charge);
}
