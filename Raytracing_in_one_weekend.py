# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:19:32 2021

@author: Mahshid
"""

### introducing Hit_Record as a Dictionary (Structure)
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as mpimg

hitrecord = {'t': 0.0,
             'p': np.array([0.0,0.0,0.0]),
             'normal': np.array([0.0,0.0,0.0]),
             'material': None}
## Ray class 
class ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + self.direction*t
    
## Generating a random point in a unit disk: (Ch. 11)        
def random_in_unit_disk():
    random_num = np.inf
    while (LA.norm(random_num))**2>=1:
        random_num = 2*np.random.rand(1, 2)-1
    return np.squeeze(random_num)

## Camera
    
class Camera():
    def __init__(self,FOV,Aspect,lookfrom,lookat,vup,aperture,focus_dist):
        self.lens_radius = aperture/2
        self.FOV = FOV
        self.Aspect = Aspect
        self.theta = FOV*np.pi/180
        self.half_height = np.tan(self.theta/2)
        self.half_width = Aspect*self.half_height
        self.origin = lookfrom
        self.w = (lookfrom-lookat)/LA.norm(lookfrom-lookat)
        self.u = np.cross(vup,self.w)
        self.v = np.cross(self.w,self.u)
        self.lower_left_corner = self.origin-self.half_width*focus_dist*self.u-self.half_height*focus_dist*self.v-focus_dist*self.w
        self.horizontal = 2*self.half_width*self.u*focus_dist
        self.vertical = 2*self.half_height*self.v*focus_dist
        
        
    def get_ray(self,a1,a2):
        rd = self.lens_radius*random_in_unit_disk()
        offset = self.u*rd[0]+self.v*rd[1]
        direction = self.lower_left_corner+ a1*self.horizontal + a2*self.vertical - self.origin - offset
        return ray(self.origin+offset,direction)
        
        
### assuming all objects are Spheres, described by the "Sphere" class
## this class takes the Center of sphere and its radius, along with the type
## of the material as a class (class of Lambertian, Metal, and Dielectric)
## Then calculates the hit points and returns the required parameters  
## Using the hitrecord dictionary

class sphere:
    def __init__(self,hitrecord,Cen,radius,material):
        self.hitrecord = hitrecord
        self.Cen = Cen     #Center of the sphere
        self.radius = radius
        self.material = material
        
    def hit(self, ray, t_min, t_max, hitrecord):
        oc = ray.origin - self.Cen
        a = np.dot(ray.direction,ray.direction)
        b = 2*np.dot(oc,ray.direction)
        c = np.dot(oc,oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        
        if discriminant>0:
            temp = (-b - np.sqrt(discriminant))/(2*a)
            if temp<t_max and temp>t_min:
                self.hitrecord["t"] = temp
                self.hitrecord["p"] = ray.point_at_parameter(hitrecord["t"])
                self.hitrecord["normal"] = (hitrecord["p"] - self.Cen)/self.radius
                self.hitrecord["material"]=self.material
                return True
            
            temp = (-b + np.sqrt(discriminant))/(2*a)
            if temp<t_max and temp>t_min:
                self.hitrecord["t"] = temp
                self.hitrecord["p"] = ray.point_at_parameter(hitrecord["t"])
                self.hitrecord["normal"]=(hitrecord["p"] - self.Cen)/self.radius
                self.hitrecord["material"]=self.material
                return True
            
        else:
            return False

### Hittable_list class in Chapter 5
      
class Hittable_list(sphere):
    def __init__(self,ray,t_min,t_max,hitrecord,list_of_obj):
        self.list_size = len(list_of_obj)
        self.ray = ray
        self.t_min = t_min
        self.t_max = t_max
        self.hitrecord = hitrecord
        self.list_of_obj = list_of_obj
        
    def hit(self, ray, t_min, t_max, hitrecord):
        temp_rec = hitrecord
        hit_anything = False
        closest_so_far = t_max
        
        for item in self.list_of_obj:
            if item.hit(ray, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec["t"]
                hitrecord = temp_rec
                
        return hit_anything

### Random number generation inside a unit sphere. Chapter 7   

def random_in_unit_sphere():
    random_num = np.inf
    while (LA.norm(random_num))**2>=1:
        random_num = 2*np.random.rand(1, 3)-1
    return np.squeeze(random_num)

######## Material Classes #########
    
## Lambertian class. Chapter 8

class Lambertian():
    def __init__(self,attenuation):
        self.attenuation = attenuation
        
    def scatter(self,Ray,hitrecord,scattered):
        target = random_in_unit_sphere() + hitrecord["p"] + hitrecord["normal"]
        scattered = ray(hitrecord["p"], target - hitrecord["p"] )
        return True, scattered

## Metal class with reflection function. Chapter 8
       
def reflect(v,n):
    reflected = v-2*np.dot(v,n)*n
    return reflected

class Metal():
    def __init__(self,attenuation,fuzz):
        self.attenuation = attenuation
        self.fuzz =1.0
        if fuzz <1:
            self.fuzz = fuzz
        
    def scatter(self,Ray,hitrecord,scattered):
        unit_dir = Ray.direction/LA.norm(Ray.direction)
        reflected = reflect(unit_dir,hitrecord["normal"])       
        scattered = ray(hitrecord["p"], reflected+self.fuzz*random_in_unit_sphere())
        outt = [np.dot(scattered.direction,hitrecord["normal"])>0 , scattered]
        return outt

### Refraction function in Chapter 9

def refract(v,n,ni_nt):
    unit_v = v/LA.norm(v)
    dt = np.dot(unit_v,n)
    discriminant = 1.0-(ni_nt**2)*(1-dt**2)
    if discriminant>0:
        refracted = ni_nt*(unit_v- n*dt) - np.sqrt(discriminant)*n
        return True , refracted
    else:
        return False, 0
 
### Simple dielectric class in Chapter 9 without Schlick:
        
class Dielectric():
    def __init__(self,ref_idx):
        self.ref_idx = ref_idx
        self.attenuation = [1. , 1.,1.]
    
    def scatter(self,Ray,hitrecord,scattered):
        reflected = reflect(Ray.direction,hitrecord["normal"])  
        if np.dot(Ray.direction,hitrecord["normal"])>0:
            outward_n = -hitrecord["normal"]
            ni_nt = self.ref_idx
        else:
            outward_n = hitrecord["normal"]
            ni_nt = 1/self.ref_idx
                
        refracted = refract(Ray.direction,outward_n,ni_nt)
        if refracted[0]:
            scattered = ray(hitrecord["p"],refracted[1])
        else:
            scattered = ray(hitrecord["p"],reflected)
            return False
            
        return True, scattered

### Dielectric class with Schlick function:

def Schlick(Cosine,ref_idx):
    r0 = (1-ref_idx)/(1+ref_idx)
    r0 = r0**2
    return r0 + (1-r0)*((1-Cosine)**5)

class Dielectric_Schlick():
    def __init__(self,ref_idx):
        self.ref_idx = ref_idx
        self.attenuation = [1.,1.,1.]
        
    def scatter(self,Ray,hitrecord,scattered):
        reflected = reflect(Ray.direction,hitrecord["normal"])  
        if np.dot(Ray.direction,hitrecord["normal"])>0:
            outward_n = -hitrecord["normal"]
            ni_nt = self.ref_idx
            Cosine = np.dot(Ray.direction,hitrecord["normal"])*self.ref_idx/LA.norm(Ray.direction)
        else:
            outward_n = hitrecord["normal"]
            ni_nt = 1/self.ref_idx
            Cosine = -np.dot(Ray.direction,hitrecord["normal"])/LA.norm(Ray.direction)
                
        refracted = refract(Ray.direction,outward_n,ni_nt)
        if refracted[0]:
            reflect_prob = Schlick(Cosine,self.ref_idx)
        else:
            #scattered = ray(hitrecord["p"],reflected)
            reflect_prob = 1.0
           # return False
            
        if np.random.rand(1)<reflect_prob:
            scattered = ray(hitrecord["p"],reflected)
        else:
            scattered = ray(hitrecord["p"],refracted[1])
                
            
        return True, scattered
    

#### Color function with considering Material
    
def color(Ray,world,hitrecord,depth):

    if world.hit(Ray, 0.0001, np.inf, hitrecord):
        scattered = ray(np.array([0.,0.,0.]),np.array([0.,0.,0.]))
        s1 = hitrecord["material"].scatter(Ray,hitrecord,scattered)
        if depth<50 and s1[0]:
            return hitrecord["material"].attenuation*color(s1[1], world,hitrecord,depth+1)
        else:
            return np.array([0.,0.,0.])
    else:
        unit_dir = Ray.direction/LA.norm(Ray.direction)
        t = 0.5*(unit_dir[1]+1)
        return (1-t)*np.array([1, 1,1]) + t*np.array([0.5, 0.7, 1])   
    
## Creating random Scene
        
def random_scene():
    obj_list = []
    obj_list.append(sphere(hitrecord,np.array([0,-1000,0]),1000,Lambertian(np.array([0.5,0.5,0.5]))))
    m = 0
    for ii in range(-3,3,1):
        for jj in range(-3, 3, 1):
            choose_mat = np.random.rand(1)
            C = np.array([ii+0.*np.random.rand(1),0.2,jj+0.9*np.random.rand(1)])
            if LA.norm(C - np.array([4.,0.2,0.0]))>0.9:
                if choose_mat < 0.8:    # diffuse material
                    attenuation = np.power(np.random.rand(1,3),2)
                    new_obj = sphere(hitrecord,C,0.2,Lambertian(attenuation))
                    obj_list.append(new_obj) 
                elif choose_mat > 0.8 and choose_mat < 0.95:    # Metal
                    attenuation = 0.5*(np.random.rand(1,3)+1)
                    fuzz = 0.5*np.random.rand(1)
                    new_obj = sphere(hitrecord,C,0.2,Metal(attenuation,fuzz))
                    obj_list.append(new_obj) 
                else:   #Glass
                    new_obj = sphere(hitrecord,C,0.2,Dielectric_Schlick(1.5))
                    obj_list.append(new_obj)
                    
    obj_list.append(sphere(hitrecord,np.array([0,1,0]),1.0,Dielectric_Schlick(1.5))) 
    obj_list.append(sphere(hitrecord,np.array([-4,1,0]),1.0,Lambertian(np.array([0.4,0.2,0.1]))))              
    obj_list.append(sphere(hitrecord,np.array([4,1,0]),1.0,Metal(np.array([0.7,0.6,0.5]),0.0)))

    return obj_list              
                    
        
###########------------ Main -----------###############
        
## Simple dielectric without Schlick equation


nx = 200
ny = 100
ns = 50

out1 = np.zeros((ny, nx, 3))

R = np.cos(np.pi/4)
'''
obj1 = sphere(hitrecord,np.array([0,0,-1]),0.5,Lambertian(np.array([0.1,0.2,0.5])))
obj2 = sphere(hitrecord,np.array([0,-100.5,-1]),100,Lambertian(np.array([0.8,0.8,0.0])))
obj3 = sphere(hitrecord,np.array([1,0,-1]),0.5,Metal([0.8,0.6,0.2],0.3))
obj4 = sphere(hitrecord,np.array([-1,0,-1]),0.5,Dielectric_Schlick(1.5))
obj5 = sphere(hitrecord,np.array([-1,0,-1]),-0.45,Dielectric_Schlick(1.5))
list_of_obj = [obj1,obj2,obj3,obj4,obj5]
'''


list_of_obj = random_scene()
world = Hittable_list(ray, 0.0, np.inf, hitrecord, list_of_obj)
lookfrom = np.array([5,3,3])
lookat = np.array([0,0,0])
aperture = 2.0
focus_dist = LA.norm(lookfrom-lookat)
Cam = Camera(90.0,float(nx/ny),lookfrom,lookat,np.array([0,1,0]),aperture,focus_dist)

for yy in range(0,ny,1):
    for xx in range(0, nx, 1):
        P = np.array([0.0,0.0,0.0])
        for ss in range(0,ns,1):
            u = float(xx+np.random.rand(1))/float(nx)
            v = float(yy+np.random.rand(1))/float(ny) 
            RR = Cam.get_ray(u,v)
            P += np.squeeze(color(RR, world, hitrecord, 0))
            
        out1[ny-yy-1,xx,0] = np.sqrt(P[0]/ns)
        out1[ny-yy-1,xx,1] = np.sqrt(P[1]/ns)
        out1[ny-yy-1,xx,2] = np.sqrt(P[2]/ns)
                 
        
imgplot = mpimg.imshow(out1)        