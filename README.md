#raytraceroptimized
Taking an existing ray tracer program and using NVIDIA's CUDA platform to optimize it

Optimizations Used  

__device__ __host__ : this was used so that parts of classes such as Vec3 and Sphere needed to be accessed by both the host and device.  
class Vec3  
{  
public:  
float x, y, z;  
     __device__ __host__ Vec3() : x(float(0)), y(float(0)), z(float(0)) {}  
     __device__ __host__ Vec3(float xx) : x(xx), y(xx), z(xx) {}  
     __device__ __host__ Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}  
     __device__ __host__ Vec3& normalize()  
  {  
  ...  
    __device__ __forceinline__ : because the program uses various loops and recursion we force the compiler to use inline functions to speed up the trace and mix functions as well as some methods in the Vec3 and Sphere class.  
    __device__ __forceinline__ Vec3 trace(const Vec3 &rayorig, const Vec3 &raydir, const Sphere* spheres, const int depth, int nsphere)
  {  
...  
   sqrtf, tanf, fmaxf : where std:: was being used we replaced it with CUDA's math library equivalents although gains were marginal  
}  
   	surfaceColor += sphere->surfaceColor * transmission * fmaxf(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;  
}  
