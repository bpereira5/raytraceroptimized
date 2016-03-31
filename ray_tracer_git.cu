
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

#define MAX_RAY_DEPTH 40
#define M_SPHERES 10000
#define rnd(Max,Min) (((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min)


#ifdef __linux__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#endif

/*ray tracer that matched the criteria I was looking for at http://scratchapixel.com/assets/Uploads/Lesson001/Source%20Code/raytracer.cpp */

void check_cuda(cudaError_t a){
    if (a!=cudaSuccess){
        std::cerr<<cudaGetErrorString(a)<<std::endl;
        exit(1);
    }
}

class Vec3
{
public:
	float x, y, z;
	__device__ __host__ Vec3() : x(float(0)), y(float(0)), z(float(0)) {}
	__device__ __host__ Vec3(float xx) : x(xx), y(xx), z(xx) {}
	__device__ __host__ Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
	__device__ __host__ Vec3& normalize()
	{
		float nor2 = length2();
		if (nor2 > 0) {
			float invNor = 1 / sqrtf(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}
	__device__ __forceinline__ Vec3 operator * (const float &f) const { return Vec3(x * f, y * f, z * f); }
	__device__ __forceinline__ Vec3 operator * (const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
	__device__ __forceinline__ float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
	__device__ __forceinline__ Vec3 operator - (const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
	__device__ __forceinline__ Vec3 operator + (const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	__device__ __forceinline__ Vec3 & operator += (const Vec3 &v) { x += v.x, y += v.y, z += v.z; return *this; }
	__device__ __host__ Vec3 & operator *= (const Vec3 &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
	__device__ __forceinline__ Vec3 operator - () const { return Vec3(-x, -y, -z); }
	__device__ __host__ float length2() const { return x * x + y * y + z * z; }
	__device__ __forceinline__ float length() const { return sqrtf(length2()); }
	__device__ __forceinline__ void cp(const Vec3& a){ x=a.x; y=a.y; z=a.z;}
	__device__ __host__ void operator = (const Vec3& a){ x=a.x; y=a.y; z=a.z;}
};

class Sphere {
public:
	Vec3 center;                         /// position of the sphere
	float radius, radius2;                      /// sphere radius and radius^2
	Vec3 surfaceColor, emissionColor;    /// surface color and emission (light)
	float transparency, reflection;             /// surface transparency and reflectivity
	
    __device__ __host__ Sphere(){}
	__device__ __host__ Sphere(const Vec3 &c, const float &r, const Vec3 &sc,
		const float &refl = 0, const float &transp = 0, const Vec3 &ec = 0) :
		center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
		transparency(transp), reflection(refl)
	{}

    __device__ __host__ void cp(Sphere a){
        center = a.center;
        radius = a.radius;
        radius2 = a.radius2;
        surfaceColor = a.surfaceColor;
        emissionColor = a.emissionColor;
        transparency = a.transparency;
        reflection = a.reflection;
    }
	__device__ __forceinline__ bool intersect(const Vec3 &rayorig, const Vec3 &raydir, float *t0 = NULL, float *t1 = NULL) const
	{
		Vec3 l = center - rayorig;
		float tca = l.dot(raydir);
		if (tca < 0) return false;
		float d2 = l.dot(l) - tca * tca;
		if (d2 > radius2) return false;
		float thc = sqrtf(radius2 - d2);
		if (t0 != NULL && t1 != NULL) {
			*t0 = tca - thc;
			*t1 = tca + thc;
		}
		return true;
	}
};


__device__ __forceinline__ float mix(const float &a, const float &b, const float &mix)
{
	return b * mix + a * (float(1) - mix);
}


__device__ __forceinline__ Vec3 trace(const Vec3 &rayorig, const Vec3 &raydir,
	const Sphere* spheres, const int depth, int nsphere)
{
	float tnear = INFINITY;
	const Sphere *sphere = NULL;

	for (unsigned i = 0; i < nsphere; ++i) {
		float t0 = INFINITY, t1 = INFINITY;
		if (spheres[i].intersect(rayorig, raydir, &t0, &t1)) {
			if (t0 < 0) t0 = t1;
			if (t0 < tnear) {
				tnear = t0;
				sphere = &spheres[i];
			}
		}
	}

	if (!sphere) { return Vec3(2);}
	Vec3 surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
	Vec3 phit = rayorig + raydir * tnear; // point of intersection
	Vec3 nhit = phit - sphere->center; // normal at the intersection point
	nhit.normalize(); // normalize normal direction


	float bias = 1e-4; // add some bias to the point from which we will be tracing
	bool inside = false;
	
	if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
	
	if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
		float facingratio = -raydir.dot(nhit);
		float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
		Vec3 refldir = raydir - nhit * 2 * raydir.dot(nhit);
		refldir.normalize();
		Vec3 reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1, nsphere);
		Vec3 refraction = 0;
		if (sphere->transparency) {
			float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
			float cosi = -nhit.dot(raydir);
			float k = 1 - eta * eta * (1 - cosi * cosi);
			Vec3 refrdir = raydir * eta + nhit * (eta *  cosi - sqrtf(k));
			refrdir.normalize();
			refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1, nsphere);
		}
		surfaceColor = (reflection * fresneleffect +
			refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
	}
	else {

		for (unsigned i = 0; i < nsphere; ++i) {
			if (spheres[i].emissionColor.x > 0) {
				// this is a light
				Vec3 transmission = 1;
				Vec3 lightDirection = spheres[i].center - phit;
				lightDirection.normalize();
				for (unsigned j = 0; j < nsphere; ++j) {
					if (i != j) {
						float t0, t1;
						if (spheres[j].intersect(phit + nhit * bias, lightDirection, &t0, &t1)) {
							transmission = 0;
							break;
						}
					}
				}
				surfaceColor += sphere->surfaceColor * transmission * fmaxf(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
			}
		}
	}

	return surfaceColor + sphere->emissionColor;
}

__global__ void trace2(Vec3* des, Sphere* spheres, int nsphere, unsigned width, unsigned height)
{
	
	float invWidth = 1 / float(width), invHeight = 1 / float(height);
	float fov = 70, aspectratio = width / float(height);
	float angle = tanf(M_PI * 0.5 * fov / float(180));
	
    int depth = 0;
    Vec3 rayorig(0);
	
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
	
    float xx = (2 * ((i + 0.5) * invWidth) - 1) * angle * aspectratio;
    float yy = (1 - 2 * ((j + 0.5) * invHeight)) * angle;
	
    Vec3 raydir(xx, yy, -1);
    raydir.normalize();
	
    if (i < height && j < height && i * height + j < height * height)
        des[i * height + j]=trace(rayorig, raydir, spheres, depth, nsphere);
}

void render(const Sphere* spheres, int nsphere, unsigned width, unsigned height)
{
	Vec3 *image = new Vec3[width * height];
    
    Vec3* d_image;
    Sphere* d_spheres;
    
    check_cuda(cudaMalloc((void**)&d_image, width*height*sizeof(Vec3)));
    check_cuda(cudaMalloc((void**)&d_spheres, nsphere*sizeof(Sphere)));
    
    check_cuda(cudaMemcpy(d_image, image, width*height*sizeof(Vec3), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_spheres, spheres, nsphere*sizeof(Sphere), cudaMemcpyHostToDevice));
    int ntpb = 20;
    int nb = (height + ntpb - 1) / ntpb;
    
    dim3 dGrid(nb,nb);
    dim3 dBlock(ntpb,ntpb);
	
	cudaThreadSetLimit(cudaLimitStackSize, 40*1500);
	
    trace2<<<dGrid, dBlock>>>(d_image, d_spheres, nsphere, width, height);
    cudaDeviceSynchronize();

    cudaError_t e = cudaGetLastError();
    if (e!=cudaSuccess){
        std::cout<<cudaGetErrorString(e)<<std::endl;
    }
    
    check_cuda(cudaMemcpy(image, d_image, width*height*sizeof(Vec3), cudaMemcpyDeviceToHost));

    cudaFree(d_image);
    cudaFree(d_spheres);

    cudaDeviceReset();

	// Save result to a PPM image (keep these flags if you compile under Windows)
	std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i) {
		ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
			(unsigned char)(std::min(float(1), image[i].y) * 255) <<
			(unsigned char)(std::min(float(1), image[i].z) * 255);
	}

	ofs.close();
	delete[] image;
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		std::cerr << "***Incorrect number of arguments***\n";
		std::cerr << "***Params: #sphere, side length***\n";
		return 1;
	}

	int n = std::atoi(argv[1]);
	int length = std::atoi(argv[2]);
	if (n > M_SPHERES) n = M_SPHERES;
    srand(5);

    Sphere* spheres = new Sphere[n+1];
	float r, x, y, z;

	//random sphere generation
	for (int i = 0; i < n; i++){

		x = rnd(10.0, -10.0);
		y = rnd(8.0, -8.0);
		z = rnd(-10, -30);
		r = rnd(4.0, 0.5);
	
		spheres[i].cp(Sphere(Vec3(x, y, z), r, Vec3(rnd(1, 0), rnd(1, 0), rnd(1, 0)), 1, 0.0));
		
	}
	//light
	spheres[n].cp(Sphere(Vec3(2, 20, 5), 3, Vec3(0), 0, 0, Vec3(4)));
    n++;



	render(spheres, n, length, length);
	delete [] spheres;

	return 0;
	
}