#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:28:49 2021

@author: selinbac
"""

# comparision of float numbers (strict)
EPS_FSTRICT = 1e-10

# ds STEP FOR THE CALCULATION OF RETURN POINTS
DS_RPT = 1e-3

# comparison of s values of MEP (in bohr)
EPS_MEPS = 1e-7

# comparison of energies of MEP (in hartree)
EPS_MEPE = 1e-8

# comparision of float numbers
EPS_FLOAT = 1e-8

import scipy
from   scipy.optimize import newton
from   scipy.optimize import fmin

def minima_in_list(lx,ly):
    '''
    in the list, it find points
    that may be local minima
    Returns the list of x-guesses
    '''
    np = len(lx)
    # initialize guesses
    guesses = []
    # initial point
    if ly[0] <= ly[1]:
       guesses.append(lx[0])
    # mid points
    for idx in range(1,np-1):
        xi, yi = lx[idx-1],ly[idx-1]
        xj, yj = lx[idx  ],ly[idx  ]
        xk, yk = lx[idx+1],ly[idx+1]
        if yj <= yi and yj <= yk: guesses.append(xj)
    # final points
    if ly[-1] <= ly[-2]:
       guesses.append(lx[-1])
    return guesses


def sign(number):
    if number >= 0.0: return +1
    else:             return -1
    
def uniquify_flist(list_of_floats,eps=EPS_FLOAT):
    ''' removes duplicates in a list of float numbers'''
    if len(list_of_floats) < 2: return list_of_floats
    # Copy list and sort it
    list2 = list(list_of_floats)
    list2.sort()
    # Get position of duplicates
    repeated = []
    if len(list2) > 1:
       for idx in range(1,len(list2)):
           previous_float = list2[idx-1]
           current_float  = list2[idx]
           if abs(current_float - previous_float) < eps: repeated.append(idx)
    # Change values of positions to Nones
    for idx in repeated: list2[idx] = None
    # Remove Nones from list
    removals = list2.count(None)
    for removal in range(removals): list2.remove(None)
    return list2

class WrongVar(Exception)       : pass

class Spline():
      '''
      https://codeplea.com/introduction-to-splines
      https://en.wikipedia.org/wiki/Cubic_Hermite_spline
      https://www.youtube.com/watch?v=UCtmRJs726U
      '''

      def __init__(self,lx,ly,tan="findiff",tension=0.0):
          # sort points, just in case
          points = [(xi,yi) for xi,yi in zip(lx,ly)]
          points.sort()
          # save data
          self._xk      = [xi  for xi,yi in points]
          self._yk      = [yi  for xi,yi in points]
          self._mk      = [0.0 for xi,yi in points]
          self._tan     = tan
          self._tension = tension
          # number of points and number of intervals
          self._np = len(self._xk)
          self._ni = self._np - 1
          # there are m+1 intervals and n+1 points (m+1=n)
          self._x0      = self._xk[ 0] # first point
          self._xn      = self._xk[-1] # last point
          self._i0      = 0            # idx of first interval
          self._im      = self._ni - 1 # idx of last  interval
          # get derivatives according to gtype
          if self._tan not in ["findiff","pchip"]: raise WrongVar(Exception)
          if self._tan == "findiff" : self.slopes_findiff()
          if self._tan == "pchip"   : self.slopes_pchip()
          #if self._tan == "cardinal": self.slopes_cardinal()

      def xx(self): return list(self._xk)

      def yy(self): return list(self._yk)

      def point(self,idx):
          return float(self._xk[idx]), float(self._yk[idx])

      def slope(self,idx):
          return float(self._mk[idx])

      def interval(self,x):
          # first or last point
          if abs(x-self._x0) <= EPS_FSTRICT: return self._i0, True
          if abs(x-self._xn) <= EPS_FSTRICT: return self._im, True
          # outside limits
          if x < self._x0: return self._i0, False
          if x > self._xn: return self._im, False
          # in the range
          for idx in range(0,self._ni):
              xi, yi = self.point(idx  )
              xj, yj = self.point(idx+1)
              if xi - EPS_FSTRICT <= x <= xj + EPS_FSTRICT:
                 return idx, True

      def interval_data(self,idx):
          x1, y1 = self.point(idx  )
          m1     = self.slope(idx  )
          x2, y2 = self.point(idx+1)
          m2     = self.slope(idx+1)
          # correct tangent
          m1 *= (x2-x1) * (1.0-self._tension)
          m2 *= (x2-x1) * (1.0-self._tension)
          return x1,y1,m1,x2,y2,m2

      def tval(self,x):
          '''
          converts the x value to interval index + t
          '''
          # get interval
          idx, inrange = self.interval(x)
          # get t in interval
          if not inrange and idx == self._i0: return idx, 0.0
          if not inrange and idx == self._im: return idx, 1.0
          # interval data
          x1,y1,m1,x2,y2,m2 = self.interval_data(idx)
          t = 1.0*(x-x1)/(x2-x1)
          # return
          return idx, float(t)

      def get_first(self):
          return self.point(0)

      def get_last(self):
          return self.point(-1)

      def basis(self,t):
          '''
          Hermite basis functions
          [extended version]
          '''
          h1  = +2*(t**3)-3*(t**2)+1
          h2  = -2*(t**3)+3*(t**2)
          h3  =    (t**3)-2*(t**2)+t
          h4  =    (t**3)-  (t**2)
          return h1, h2, h3, h4

      def dbasis(self,t):
          '''
          derivative Hermite basis functions
          [extended version]
          '''
          dh1  = +6*(t**2)-6*t
          dh2  = -6*(t**2)+6*t
          dh3  =  3*(t**2)-4*t+1
          dh4  =  3*(t**2)-2*t
          return dh1, dh2, dh3, dh4

      def interpolate(self,x):
          idx, t = self.tval(x)
          # interval
          x1,y1,m1,x2,y2,m2 = self.interval_data(idx)
          # hermite
          h1, h2, h3, h4 = self.basis(t)
          # value
          H = y1*h1 + y2*h2 + m1*h3 + m2*h4
          return H

      def derivative(self,x):
          idx, t = self.tval(x)
          # interval
          x1,y1,m1,x2,y2,m2 = self.interval_data(idx)
          # hermite
          dh1, dh2, dh3, dh4 = self.dbasis(t)
          # dt/dx
          dtdx = 1.0/(x2-x1)
          # value
          dH = (y1*dh1 + y2*dh2 + m1*dh3 + m2*dh4)*dtdx
          return dH

      def __call__(self,x):
          return self.interpolate(x)

      def find_xtr(self,xtr="max"):
          if xtr.lower() not in  ["max","min"]: return
          if xtr == "min":
             def fx(x): return +self.interpolate(x)
             ly = [+y for y in self._yk]
          else:
             def fx(x): return -self.interpolate(x)
             ly = [-y for y in self._yk]
          # get guesses
          guesses = minima_in_list(self._xk,ly)
          # accurate search
          x_xtr, y_xtr = None, +float("inf")
          for guess in guesses:
              xm, ym = scipy.optimize.fmin(fx,guess,disp=False,full_output=True)[0:2]  #is this correct??
              # convert to float
              xm, ym = float(xm), float(ym)
              # check if it's the smallest one
              if ym < y_xtr: x_xtr,y_xtr = xm,ym
          # Save data
          if xtr == "min": self._xmin, self._ymin = (+x_xtr,+y_xtr)
          if xtr == "max": self._xmax, self._ymax = (+x_xtr,-y_xtr)

      def get_min(self): return float(self._xmin), float(self._ymin)

      def get_max(self): return float(self._xmax), float(self._ymax)

      #----------------------------------#
      # functions related to derivatives #
      #----------------------------------#
      def slopes_findiff(self):
          for k in range(0,self._np):
              xk, yk = self.point(k)
              mk_l, mk_r = None, None
              if k >= 1:
                  xj, yj = self.point(k-1)
                  mk_l   = (yk-yj)/(xk-xj)
              if k <= self._np-2:
                  xl, yl = self.point(k+1)
                  mk_r   = (yl-yk)/(xl-xk)
              if   mk_l is None: mk = mk_r
              elif mk_r is None: mk = mk_l
              else             : mk = (mk_r+mk_l)/2.0
              self._mk[k] = mk

      def slopes_pchip(self):
          # first, get slopes
          lhk = [0.0 for k in range(self._np)]
          ldk = [0.0 for k in range(self._np)]
          for k in range(0,self._np-1):
              xk, yk = self.point(k)
              xl, yl = self.point(k+1)
              hk = xl-xk
              dk = (yl-yk)/hk
              lhk[k] = hk
              ldk[k] = dk
          lhk[-1] = lhk[-2]
          ldk[-1] = ldk[-2]
          # now calculate using PCHIP
          self._mk = list(ldk)
          for k in range(1,self._np-1):
              dj = ldk[k-1]
              dk = ldk[k  ]
              hj = lhk[k-1]
              hk = lhk[k  ]
              w1 = 2*hk+hj
              w2 = hk+2*hj
              cond1 = (dj == 0.0 or dk == 0.0)
              cond2 = (sign(dj) != sign(dk))
              if cond1 or cond2: self._mk[k] = 0.0
              else             : self._mk[k] = (w1+w2)/(w1/dj+w2/dk)

     #def slopes_cardinal(self,tension=0.0):
     #    self._mk = [(0.0,0.0) for k in range(self._np)]
     #    for idx in range(0,self._ni):
     #        if idx == 0:
     #           x0,y0 = self.point(idx  )
     #           x1,y1 = self.point(idx  )
     #           x2,y2 = self.point(idx+1)
     #           x3,y3 = self.point(idx+2)
     #        elif idx == self._im:
     #           x0,y0 = self.point(idx-1)
     #           x1,y1 = self.point(idx  )
     #           x2,y2 = self.point(idx+1)
     #           x3,y3 = self.point(idx+1)
     #        else:
     #           x0,y0 = self.point(idx-1)
     #           x1,y1 = self.point(idx  )
     #           x2,y2 = self.point(idx+1)
     #           x3,y3 = self.point(idx+2)
     #        # get dx, dx1, dx2
     #        dx  = x2-x1
     #        dx1 = x1-x0
     #        dx2 = x3-x2
     #        # get s1 and s2
     #        s1 = 2.0 * (dx / (dx1+dx) )
     #        s2 = 2.0 * (dx / (dx2+dx) )
     #        # calculate slope
     #        c = 1.0-tension
     #        m1 = s1 * c * (y2-y0)
     #        m2 = s2 * c * (y3-y1)
     #        self._mk[idx  ] = (m1,m2)
#==================================================#
class VadiSpline(Spline):

      def __init__(self,xx,yy):
         self._spl = Spline.__init__(self,xx,yy,tan="findiff",tension=0.0)
         # saddle point
         self._ssad, self._Vsad = None, None
         for x,y in zip(xx,yy):
             if x == 0.0:
                self._ssad, self._Vsad = x, y
                break

      def setup(self):
          # find maximum
          self.find_xtr("max")
          self._sAG, self._VAG = self.get_max()

      def get_alpha(self) : return self.get_first()
      def get_saddle(self): return float(self._ssad), float(self._Vsad)
      def get_omega(self) : return self.get_last()
      def get_AG(self)   : return float(self._sAG), float(self._VAG)
      def relative(self,si,E=0.0): return self(si) - E

      def returnpoints(self,E):
          salpha, Valpha = self.get_first()
          somega, Vomega = self.get_last()
          # add first point
          rtn_points = [salpha]
          # find points
          si,Vi = self.get_first()
          sj    = min(si+DS_RPT,somega)
          Vj    = self( sj )
          while True:
                # break?
                if abs(sj-somega)<EPS_MEPS: break
                # Energy really close to Vj?
                if abs(Vj-E) < EPS_MEPE: rtn_points.append( sj )
                # Energy between? Then get return point
                elif (Vi < E < Vj) or (Vi > E > Vj):
                   guess = (si+sj)/2.0
                   rpoint = scipy.optimize.newton(self.relative,guess,args=(E,))
                   if si < rpoint < sj: rtn_points.append( rpoint )
                # update
                si, Vi = sj, Vj
                sj = min(si+DS_RPT,somega)
                Vj = self( sj )
          # add last point
          rtn_points += [somega]
          # remove repetitions
          rtn_points = uniquify_flist(rtn_points,EPS_MEPS)
          # convert to intervals
          intervals = []
          for idx in range(len(rtn_points)-1):
              si   = rtn_points[idx  ]
              sj   = rtn_points[idx+1]
              smid = (si+sj)/2.0
              Vmid = self(smid)
              if Vmid >= E: intervals.append( (si,sj) )
          return intervals
#==================================================#

'''
def test(case=1):
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import PchipInterpolator

    if case == 1:
       xx  = [0.1 * ii for ii in range(11)]
       yy  = [0.25,1.2,0.9,0.75,0.55,1.0,1.8,1.0,1.0,0.78,0.20]
       xx2 = [0.01*ii for ii in range(101)]

    if case == 2:
       def f(xi):
           return 0.001*xi**2
          #return 0.90*np.sin(random.random()) + 0.003*xi**2
          #return 0.001*xi**2+np.sin(np.pi*xi/20.0)
       xx  = [0.1 * ii for ii in range(11)]
       yy  = [f(x) for x in xx]
       xx2 = [0.01*ii for ii in range(101)]


    plt.plot(xx,yy ,'o')

    # univariate
    spl = UnivariateSpline(xx,yy,k=3,ext=3,s=0)
    yy2 = [ spl(xi) for xi in xx2]
    plt.plot(xx2,yy2,'k-',label="univar")

    # no tension
    spl = Spline(xx,yy,"findiff",0.0)
    yy2 = [ spl(xi) for xi in xx2]
    plt.plot(xx2,yy2,'g--',label="t=0.0")

    # tension
    spl = Spline(xx,yy,"findiff",1.0)
    yy2 = [ spl(xi) for xi in xx2]
    plt.plot(xx2,yy2,'r--',label="t=1.0")
    
    print("check derivative:")
    xi = 0.376
    dx = 1e-5
    a = spl(xi)
    b = spl(xi+dx)
    d = (b-a)/dx
    print(spl.derivative(xi), d)
    plt.legend(loc="best")
    plt.show()
    print(spl(0.35))

if __name__ == '__main__': test(1)
'''
'''
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator

case=2

if case == 1:
   xx  = [0.1 * ii for ii in range(11)]
   yy  = [0.25,1.2,0.9,0.75,0.55,1.0,1.8,1.0,1.0,0.78,0.20]
   xx2 = [0.01*ii for ii in range(101)]
   
if case == 2:
   def f(xi):
       return 0.001*xi**2
      #return 0.90*np.sin(random.random()) + 0.003*xi**2
      #return 0.001*xi**2+np.sin(np.pi*xi/20.0)
   xx  = [0.1 * ii for ii in range(11)]
   yy  = [f(x) for x in xx]
   xx2 = [0.01*ii for ii in range(101)]


plt.plot(xx,yy ,'o')

# univariate
spl = UnivariateSpline(xx,yy,k=3,ext=3,s=0)
yy2 = [ spl(xi) for xi in xx2]
plt.plot(xx2,yy2,'k-',label="univar")

print(spl(0.453))
'''

